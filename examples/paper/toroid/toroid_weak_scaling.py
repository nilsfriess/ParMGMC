"""Parallel (weak) scaling of the algebraic MGMC sampler on a twisted toroid.

Samples from the Whittle-Matern prior N(0, A^-1) with A the CG-1 discretisation of
kappa^2 - Delta (natural boundary conditions) on a uniformly refined twisted toroid (see
twisted_toroid.py), using algebraic MGMC (PCGAMGMC) with Gibbs random smoothers.  Reports
the time per MGMC iteration, setup times, the IACT of a scalar QOI and a correctness check
against the exact QOI variance.

Mesh: the coarse mesh (-tt_n cells across the cross-section, -tt_ntheta around the ring)
is built on rank 0, refined -tt_prerefine times there, then distributed and refined in
parallel until it has been refined -tt_refine times in total.  Pre-refining avoids
distributing a coarse mesh with only a handful of cells per rank.

Smoothers (-tt_smoother):
    gibbs    the correct parallel Gibbs sampler (PCSORGIBBS, which uses PCPARSOR in parallel)
    hogwild  Gibbs sweeps on each rank that ignore the coupling to other ranks during a
             sweep (-pc_sorgibbs_local_forward); does NOT sample from the target
             distribution, only a baseline for the cost of the correct parallel sweep
GAMG parameters (-tt_gamg): default (PETSc defaults) or tuned (as in the wrench study).
Any PETSc option on the command line overrides these defaults.

Timed stages (also PETSc log stages): mesh, assemble, setup (KSPSetUp + one iteration),
burnin, sampling.  Results: summary on stdout, JSON record (-tt_out), QOI chain (-tt_chain),
optionally the last sample as VTK (-tt_vtk <file.pvd>).

Example:
    mpiexec -n 8 python toroid_weak_scaling.py -tt_refine 3 -tt_prerefine 1 -tt_smoother gibbs
"""

import datetime
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from time import perf_counter

import emcee
import firedrake as fd
import numpy as np
import pymgmc
import twisted_toroid
from firedrake.petsc import PETSc
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))

# QOI: L^2 integral of the field over a ball inside the ring
QOI_CENTER = (1.0, 0.0, 0.0)
QOI_RADIUS = 0.25

GAMG_TUNED = {
    "gamgmc_pc_gamg_aggressive_coarsening": "0",
    "gamgmc_pc_gamg_agg_nsmooths": "2",
    "gamgmc_pc_gamg_threshold": "0.04",
    "gamgmc_pc_gamg_threshold_scale": "0.25",
}


def sampler_options(gamg, smoother):
    opts = {"pc_type": "gamgmc", "pc_gamgmc_mg_type": "gamg"}
    if gamg == "tuned":
        opts.update(GAMG_TUNED)
    elif gamg != "default":
        raise SystemExit("-tt_gamg must be default or tuned")
    if smoother == "hogwild":
        opts["gamgmc_mg_levels_pc_sorgibbs_local_forward"] = "true"
    elif smoother != "gibbs":
        raise SystemExit("-tt_smoother must be gibbs or hogwild")
    return opts


def progress(comm, msg):
    """Progress message from rank 0 (with -tt_verbose), to locate failures at scale."""
    if comm.rank == 0 and PETSc.Options().getBool("tt_verbose", False):
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def make_mesh(comm, n, ntheta, prerefine, refine):
    if comm.rank == 0:
        coords, tets = twisted_toroid.build(n, ntheta)
    else:
        coords, tets = None, None
    progress(comm, "coarse mesh built on rank 0")
    plex = fd.mesh.plex_from_cell_list(3, tets, coords, comm)
    for _ in range(prerefine):
        plex.setRefinementUniform(True)
        plex = plex.refine()
    progress(comm, f"pre-refined {prerefine} times on rank 0, distributing")
    coarse = fd.Mesh(plex)
    progress(comm, "coarse mesh created, refining in parallel")
    if refine > prerefine:
        return fd.MeshHierarchy(coarse, refine - prerefine)[-1]
    return coarse


def qoi_vector(V):
    x = fd.SpatialCoordinate(V.mesh())
    r_sq = sum((x[i] - QOI_CENTER[i]) ** 2 for i in range(3))
    chi = fd.conditional(fd.lt(r_sq, QOI_RADIUS ** 2), 1.0, 0.0)
    meas = fd.assemble(chi * fd.TestFunction(V) * fd.dx)
    with meas.dat.vec_ro as m:
        return m.copy()


def exact_variance(A, m):
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOptionsPrefix("tt_ref_")
    ksp.setOperators(A, A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=5000)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setErrorIfNotConverged(True)
    ksp.setFromOptions()
    z = m.duplicate()
    ksp.solve(m, z)
    its = ksp.getIterationNumber()
    ksp.destroy()
    return float(m.dot(z)), its


def md5(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


def loaded_library(name):
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                if name in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None


def git_info():
    """Commit of the checkout; read from .git directly if there is no git executable."""
    try:
        res = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return {"commit": res.stdout.strip(), "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        pass
    path = HERE
    while not os.path.isdir(os.path.join(path, ".git")):
        if os.path.dirname(path) == path:
            return {"commit": None, "dirty": None}
        path = os.path.dirname(path)
    gitdir = os.path.join(path, ".git")
    try:
        with open(os.path.join(gitdir, "HEAD")) as fh:
            head = fh.read().strip()
        if not head.startswith("ref: "):
            return {"commit": head, "dirty": None}
        ref = head[len("ref: "):]
        if os.path.isfile(os.path.join(gitdir, ref)):
            with open(os.path.join(gitdir, ref)) as fh:
                return {"commit": fh.read().strip(), "dirty": None}
        with open(os.path.join(gitdir, "packed-refs")) as fh:
            for line in fh:
                if line.rstrip().endswith(" " + ref):
                    return {"commit": line.split()[0], "dirty": None}
    except OSError:
        pass
    return {"commit": None, "dirty": None}


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    faulthandler.enable()  # Python traceback on SIGSEGV (run with -no_signal_handler)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    opts = PETSc.Options()
    cmdline_opts = opts.getAll()

    n = opts.getInt("tt_n", 4)
    ntheta = opts.getInt("tt_ntheta", 11 * n)
    prerefine = opts.getInt("tt_prerefine", 0)
    refine = opts.getInt("tt_refine", 2)
    kappa = opts.getReal("tt_kappa", 10.0)
    gamg = opts.getString("tt_gamg", "default")
    smoother = opts.getString("tt_smoother", "gibbs")
    nburnin = opts.getInt("tt_nburnin", 100)
    nsamples = opts.getInt("tt_nsamples", 1000)
    reference = opts.getBool("tt_reference", True)
    seed = opts.getInt("tt_seed", 12345)
    out_json = opts.getString("tt_out", "")
    out_chain = opts.getString("tt_chain", "")
    out_vtk = opts.getString("tt_vtk", "")
    if prerefine > refine:
        raise SystemExit("-tt_prerefine must not exceed -tt_refine")
    if not 0 <= seed < 2**20 or comm.size > 2**11:
        raise SystemExit("need 0 <= -tt_seed < 2^20 and at most 2^11 ranks (see the seeding below)")

    defaults = sampler_options(gamg, smoother)
    for key, val in defaults.items():
        if not opts.hasName(key):
            opts[key] = val

    # Independent random streams on the ranks.  ParMGMC seeds its MKL stream with
    # (seed XOR rank) truncated to 32 bits, PETSc's generator uses the seed as given; with
    # seed < 2^20 the value seed + rank * 2^20 is distinct per rank in both cases.
    pymgmc.seed(seed + rank * 2**20)

    timings = {}

    def timed(name, fn):
        stage = PETSc.Log.Stage("tt_" + name)
        stage.push()
        progress(comm, f"stage {name}")
        comm.Barrier()
        t0 = perf_counter()
        result = fn()
        comm.Barrier()
        timings[name] = comm.allreduce(perf_counter() - t0, op=MPI.MAX)
        stage.pop()
        return result

    mesh = timed("mesh", lambda: make_mesh(comm, n, ntheta, prerefine, refine))
    V = fd.FunctionSpace(mesh, "CG", 1)

    def assemble():
        u, v = fd.TrialFunction(V), fd.TestFunction(V)
        k2 = fd.Constant(kappa * kappa)
        return fd.assemble((k2 * u * v + fd.inner(fd.grad(u), fd.grad(v))) * fd.dx).M.handle

    A = timed("assemble", assemble)
    m = qoi_vector(V)
    rhs = A.createVecs()[0]
    rhs.zeroEntries()
    x = fd.Function(V, name="sample")

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A, A)
    opts["ksp_type"] = "richardson"
    opts["ksp_max_it"] = 1
    opts["ksp_convergence_test"] = "skip"
    opts["ksp_initial_guess_nonzero"] = True
    opts["ksp_norm_type"] = "none"
    ksp.setFromOptions()

    def run_iterations(k):
        ksp.setTolerances(max_it=k)
        with x.dat.vec as xv:
            ksp.solve(rhs, xv)

    timed("setup", ksp.setUp)
    if rank == 0:
        print("-" * 72 + "\nKSP/PC configuration after setup:", flush=True)
    ksp.view()
    t_setup_kspsetup = timings["setup"]
    timed("first_iteration", lambda: run_iterations(1))
    pc = ksp.getPC()
    try:
        mg_levels = pc.getMGLevels()
    except PETSc.Error:
        mg_levels = None

    if nburnin > 0:
        timed("burnin", lambda: run_iterations(nburnin))

    chain = np.zeros(nsamples)
    count = [0]

    def callback(it, xv):
        chain[count[0]] = m.dot(xv)
        count[0] += 1

    pymgmc.PCSetSampleCallback(pc, callback)
    timed("sampling", lambda: run_iterations(nsamples))
    if count[0] != nsamples:
        raise RuntimeError(f"sample callback called {count[0]} times for {nsamples} iterations")

    tau_raw = float(emcee.autocorr.integrated_time(chain, quiet=True)[0])
    tau = max(tau_raw, 1.0)
    qoi_mean = float(np.mean(chain))
    qoi_var = float(np.var(chain, ddof=1))
    t_iter = timings["sampling"] / nsamples

    ref = None
    if reference:
        t0 = perf_counter()
        var_ex, its = exact_variance(A, m)
        ref = {"qoi_var": var_ex, "cg_iterations": its, "time": perf_counter() - t0,
               "var_ratio": qoi_var / var_ex, "var_ratio_se": float(np.sqrt(2.0 * tau / nsamples)),
               "mean_z": qoi_mean / float(np.sqrt(var_ex * tau / nsamples))}

    if out_vtk:
        fd.VTKFile(out_vtk).write(x)

    nlocal = A.getLocalSize()[0]
    local_sizes = comm.gather(nlocal, root=0)
    ncells = int(comm.allreduce(int(mesh.cell_set.size), op=MPI.SUM))
    nnz = int(A.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"])
    cpus = None
    try:
        with open("/proc/self/status") as fh:
            cpus = next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("Cpus_allowed_list")), None)
    except OSError:
        pass
    hosts = comm.gather((socket.gethostname(), cpus), root=0)
    if rank != 0:
        return

    ndofs = V.dim()
    record = {
        "driver": "toroid_weak_scaling.py",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "problem": {"n": n, "ntheta": ntheta, "prerefine": prerefine, "refine": refine,
                    "kappa": kappa, "ndofs": ndofs, "ncells": ncells, "nnz": nnz,
                    "qoi": {"center": QOI_CENTER, "radius": QOI_RADIUS}},
        "sampler": {"gamg": gamg, "smoother": smoother, "mg_levels": mg_levels, "options": defaults},
        "parallel": {"nranks": comm.size, "nnodes": len({h for h, _ in hosts}),
                     "dofs_per_rank_min": min(local_sizes), "dofs_per_rank_max": max(local_sizes),
                     "dofs_per_rank_mean": ndofs / comm.size, "ranks": [list(h) for h in hosts]},
        "run": {"seed": seed, "nburnin": nburnin, "nsamples": nsamples},
        "timings": {**timings, "kspsetup": t_setup_kspsetup, "per_iteration": t_iter,
                    "per_indep_sample": t_iter * tau},
        "chain": {"iact_raw": tau_raw, "iact": tau, "n_over_iact": nsamples / tau,
                  "reliable_iact": nsamples >= 50 * tau, "qoi_mean": qoi_mean, "qoi_var": qoi_var},
        "reference": ref,
        "provenance": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git": git_info(), "driver_md5": md5(os.path.abspath(__file__)),
            "libparmgmc": loaded_library("libparmgmc"),
            "libparmgmc_md5": md5(loaded_library("libparmgmc") or ""),
            "petsc": ".".join(str(v) for v in PETSc.Sys.getVersion()),
            "petsc4py": version("petsc4py"), "firedrake": version("firedrake"),
            "emcee": version("emcee"), "python": sys.version.split()[0],
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "argv": sys.argv, "cmdline_options": cmdline_opts,
        },
    }

    print("=" * 72)
    print(f"toroid weak scaling: AMGMC ({gamg} GAMG, {smoother} smoothers, {mg_levels} levels)")
    print(f"  ranks / nodes          : {comm.size} / {record['parallel']['nnodes']}")
    print(f"  refine (pre) / kappa   : {refine} ({prerefine}) / {kappa}")
    print(f"  DOFs, per rank         : {ndofs}, {min(local_sizes)}..{max(local_sizes)}")
    print("  times [s]              : " + ", ".join(f"{k} {v:.3f}" for k, v in timings.items()))
    print(f"  time per iteration [s] : {t_iter:.6e}")
    print(f"  IACT (raw) / N/IACT    : {tau:.3f} ({tau_raw:.3f}) / {nsamples / tau:.1f}")
    if ref is not None:
        print(f"  var ratio / mean z     : {ref['var_ratio']:.4f} +- {ref['var_ratio_se']:.4f} / "
              f"{ref['mean_z']:+.2f} (CG its {ref['cg_iterations']})")
    print("=" * 72, flush=True)

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            # numpy scalars -> Python numbers
            json.dump(record, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if out_chain:
        os.makedirs(os.path.dirname(os.path.abspath(out_chain)), exist_ok=True)
        np.save(out_chain, chain)


if __name__ == "__main__":
    main()
