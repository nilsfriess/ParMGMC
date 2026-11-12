"""Mesh-refinement study on the wrench: Gibbs vs. algebraic/geometric MGMC vs. Cholesky.

Samples from the Whittle-Matern prior N(0, A^-1), or from the posterior of a linear
Bayesian inverse problem, on a uniformly refined wrench mesh, records a scalar QOI
along the chain and reports timings, the integrated autocorrelation time (IACT) and
a correctness check against the exact QOI mean/variance.

Prior precision (-wr_alpha):
    1: kappa^2 u v + grad u . grad v, CG-1, natural (Neumann) boundary conditions
    2: C0 interior-penalty discretisation of (kappa^2 - Delta)^2, CG-2 (as in ex13)

Samplers (-wr_sampler), each a fixed set of PETSc options (see SAMPLERS below):
    gibbs        single-site Gibbs (PCSORGIBBS)
    amgmc        algebraic MGMC, PETSc's default GAMG parameters
    amgmc_tuned  algebraic MGMC, GAMG parameters tuned in earlier experiments
    gmgmc        geometric MGMC on the MeshHierarchy (Gibbs smoothers for alpha=1,
                 ASMStarPC patch samplers for alpha=2)
    cholesky     Cholesky sampler (prior) / Woodbury + Cholesky (posterior)
Any PETSc option given on the command line overrides the sampler defaults.

Timed stages (also pushed as PETSc log stages, so -log_view breaks them down):
    setup     KSPSetUp plus one sampler iteration (triggers lazy initialisation,
              e.g. the low-rank posterior hierarchy in PCGAMGMC)
    burnin    -wr_nburnin iterations (QOI not recorded)
    sampling  up to -wr_nsamples iterations in chunks of -wr_chunk, stopped early once
              -wr_max_time seconds of sampling have elapsed (the chain is truncated
              to the iterations actually performed)

The exact QOI mean m^T A^-1 f and variance m^T A^-1 m are computed with CG + GAMG
(untimed) unless -wr_reference 0 is given.

Results: a human-readable summary on stdout, a JSON record (-wr_out) and the QOI chain
as a .npy file (-wr_chain).

Example:
    python wrench_refinement.py -wr_refine 3 -wr_kappa 0.1 -wr_sampler amgmc \
        -wr_posterior -wr_out run.json -wr_chain run.npy
"""

import datetime
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

# Monkeypatches Firedrake so that the DM provides assembled (AIJ) prolongations, which
# the Galerkin coarse-operator assembly of the geometric sampler needs.  Must be
# imported before the MeshHierarchy is built.
import assembled_transfer  # noqa: F401
import emcee
import firedrake as fd
import numpy as np
import pymgmc
from firedrake import dmhooks, solving_utils
from firedrake.petsc import PETSc
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_FILE = os.path.join(HERE, "..", "..", "..", "data", "wrench.msh")

# QOI: L^2 integral of the field over a ball around the ring end of the wrench
QOI_CENTER = (0.0, 10.0, 0.0)
QOI_RADIUS = 70.0

# Posterior observations: volume-averages over small balls along the shaft, with
# independent Gaussian noise of variance OBS_SIGMA2.
OBS_CENTERS = np.array([
    (18.0, 15.0, 0.0),
    (2.0, 40.0, 0.0),
    (-2.0, 90.0, 0.0),
    (2.0, 140.0, 0.0),
    (-2.0, 190.0, 0.0),
    (0.0, 210.0, 0.0),
])
OBS_RADII = np.full(len(OBS_CENTERS), 6.0)
OBS_VALUES = np.array([-0.3, 0.10, -0.15, 0.05, -0.10, 0.12])
OBS_SIGMA2 = 2.5e-3

# Tuned GAMG parameters ("optimised algebraic sampler" in the earlier experiments,
# masterthesis/_ParMGMC.old/jobs/{2d,3d}_alg_geo_comp.sh on Helix).
GAMG_TUNED = {
    "gamgmc_pc_gamg_aggressive_coarsening": "0",
    "gamgmc_pc_gamg_agg_nsmooths": "2",
    "gamgmc_pc_gamg_threshold": "0.04",
    "gamgmc_pc_gamg_threshold_scale": "0.25",
}

# Patch (overlapping Schwarz) random smoothers for the fourth-order problem (alpha=2)
ASMSTAR_SMOOTHER = {
    "gamgmc_mg_levels_ksp_type": "richardson",
    "gamgmc_mg_levels_ksp_max_it": "1",
    "gamgmc_mg_levels_pc_type": "python",
    "gamgmc_mg_levels_pc_python_type": "firedrake.ASMStarPC",
    "gamgmc_mg_levels_pc_star_construct_dim": "0",
    "gamgmc_mg_levels_pc_star_sub_pc_asm_local_type": "multiplicative",
    "gamgmc_mg_levels_pc_star_sub_sub_ksp_type": "richardson",
    "gamgmc_mg_levels_pc_star_sub_sub_ksp_max_it": "1",
    "gamgmc_mg_levels_pc_star_sub_sub_pc_type": "cholsampler",
}


def sampler_options(name, alpha, posterior):
    """PETSc options (without leading '-') that define the sampler ``name``."""
    if name == "gibbs":
        return {"pc_type": "sorgibbs"}
    if name == "amgmc":
        return {"pc_type": "gamgmc", "pc_gamgmc_mg_type": "gamg"}
    if name == "amgmc_tuned":
        return {"pc_type": "gamgmc", "pc_gamgmc_mg_type": "gamg", **GAMG_TUNED}
    if name == "gmgmc":
        opts = {"pc_type": "gamgmc", "pc_gamgmc_mg_type": "mg",
                "gamgmc_pc_mg_galerkin": "both"}
        if alpha == 2:
            opts.update(ASMSTAR_SMOOTHER)
        return opts
    if name == "cholesky":
        if posterior:
            return {"pc_type": "woodbury", "pc_woodbury_sampler": "cholsampler",
                    "pc_woodbury_solver": "cholesky"}
        return {"pc_type": "cholsampler"}
    raise SystemExit(f"unknown sampler '{name}'")


def prior_form(V, kappa, alpha):
    """Bilinear form of the prior precision operator."""
    w = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    kappa_sq = fd.Constant(kappa * kappa)
    if alpha == 1:
        return (kappa_sq * w * v + fd.inner(fd.grad(w), fd.grad(v))) * fd.dx
    # alpha == 2: C0 interior-penalty form of the squared shifted Laplacian (as in ex13)
    mesh = V.mesh()
    penalty = fd.Constant(8.0)
    n = fd.FacetNormal(mesh)
    h = 2.0 * fd.avg(fd.CellVolume(mesh)) / fd.avg(fd.FacetArea(mesh))
    lap_w = fd.div(fd.grad(w))
    lap_v = fd.div(fd.grad(v))
    return (
        fd.inner(lap_w, lap_v) * fd.dx
        + (
            penalty / h * fd.inner(fd.jump(fd.grad(w), n), fd.jump(fd.grad(v), n))
            - fd.avg(lap_w) * fd.jump(fd.grad(v), n)
            - fd.jump(fd.grad(w), n) * fd.avg(lap_v)
        ) * fd.dS
        + 2 * kappa_sq * fd.inner(fd.grad(w), fd.grad(v)) * fd.dx
        + kappa_sq ** 2 * w * v * fd.dx
    )


def qoi_vector(V):
    """PETSc Vec m with m^T x = integral of the field x over the QOI ball."""
    x = fd.SpatialCoordinate(V.mesh())
    r_sq = sum((x[i] - QOI_CENTER[i]) ** 2 for i in range(len(QOI_CENTER)))
    chi = fd.conditional(fd.lt(r_sq, QOI_RADIUS ** 2), 1.0, 0.0)
    meas = fd.assemble(chi * fd.TestFunction(V) * fd.dx)
    with meas.dat.vec_ro as m:
        return m.copy()


def setup_posterior(V, A):
    """Build ``A_post = A + B Sigma^-1 B^T`` (MATLRC) and RHS ``f = B Sigma^-1 y``.

    Returns ``(A_post, f, B, S)``; the caller must keep ``B`` and ``S`` alive for as
    long as the LRC matrix is used.
    """
    mesh = V.mesh()
    comm = mesh.comm
    gdim = mesh.coordinates.ufl_shape[0]
    nobs = len(OBS_CENTERS)

    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    M = fd.assemble(fd.inner(u, v) * fd.dx).M.handle
    vols = (4.0 / 3.0) * np.pi * OBS_RADII ** 3

    n_local, n_global = A.getLocalSize()[0], A.getSize()[0]
    B = PETSc.Mat().createDense(((n_local, n_global), (PETSc.DECIDE, nobs)), comm=comm)
    B.setUp()

    # Column i of B is M @ I[1_ball_i / vol_i], so (B^T u)_i is the L^2 average of u
    # over ball i.
    xsym = fd.SpatialCoordinate(mesh)
    ind = fd.Function(V)
    col = fd.Function(V)
    for i in range(nobs):
        c = fd.as_vector([fd.Constant(float(OBS_CENTERS[i, k])) for k in range(gdim)])
        r = float(OBS_RADII[i])
        ind.interpolate(fd.conditional(fd.lt(fd.dot(xsym - c, xsym - c), r * r),
                                       1.0 / float(vols[i]), 0.0))
        with ind.dat.vec_ro as iv, col.dat.vec as cv:
            M.mult(iv, cv)
        with col.dat.vec_ro as src:
            cvec = B.getDenseColumnVec(i, mode="w")
            src.copy(cvec)
            B.restoreDenseColumnVec(i, mode="w")
    B.assemble()

    S, f = B.createVecs()  # S: size nobs, f: size n_dofs
    S.set(1.0 / OBS_SIGMA2)

    y, _ = B.createVecs()
    lo, hi = y.getOwnershipRange()
    y.array[:] = OBS_VALUES[lo:hi]
    Sy = y.duplicate()
    Sy.pointwiseMult(S, y)
    B.mult(Sy, f)

    A_post = PETSc.Mat().createLRC(A, B, S, None)
    return A_post, f, B, S


def exact_reference(operator, A, m, f, posterior):
    """Exact QOI mean m^T operator^-1 f and variance m^T operator^-1 m (CG + GAMG on A)."""
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOptionsPrefix("wr_ref_")
    ksp.setOperators(operator, A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=1e-12, atol=0.0, max_it=5000)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setErrorIfNotConverged(True)
    ksp.setFromOptions()

    z = m.duplicate()
    ksp.solve(m, z)
    var = m.dot(z)
    its = [ksp.getIterationNumber()]
    mean = 0.0
    if posterior:
        ksp.solve(f, z)
        mean = m.dot(z)
        its.append(ksp.getIterationNumber())
    ksp.destroy()
    return float(mean), float(var), its


def md5(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


def loaded_library(name):
    """Path of a shared library loaded into this process (from /proc/self/maps)."""
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                if name in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None


def proc_status(key):
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def git_info():
    def run(*args):
        try:
            res = subprocess.run(["git", "-C", HERE, *args], capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return res.stdout.strip() if res.returncode == 0 else None
    status = run("status", "--porcelain", "--untracked-files=no")
    info = {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}
    if not info["commit"]:
        # No git executable (e.g. on the compute nodes): read the checkout's HEAD directly.
        # The dirty state is unknown in this case.
        info.update(git_head_from_files())
    return info


def git_head_from_files():
    path = HERE
    while not os.path.isdir(os.path.join(path, ".git")):
        parent = os.path.dirname(path)
        if parent == path:
            return {"commit": None, "branch": None}
        path = parent
    gitdir = os.path.join(path, ".git")
    try:
        with open(os.path.join(gitdir, "HEAD")) as fh:
            head = fh.read().strip()
        if not head.startswith("ref: "):
            return {"commit": head, "branch": None}
        ref = head[len("ref: "):]
        branch = ref.removeprefix("refs/heads/")
        ref_file = os.path.join(gitdir, ref)
        if os.path.isfile(ref_file):
            with open(ref_file) as fh:
                return {"commit": fh.read().strip(), "branch": branch}
        with open(os.path.join(gitdir, "packed-refs")) as fh:
            for line in fh:
                if line.rstrip().endswith(" " + ref):
                    return {"commit": line.split()[0], "branch": branch}
    except OSError:
        pass
    return {"commit": None, "branch": None}


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    opts = PETSc.Options()
    cmdline_opts = opts.getAll()  # before the driver adds its own

    mesh_file = opts.getString("wr_mesh", MESH_FILE)
    alpha = opts.getInt("wr_alpha", 1)
    refine = opts.getInt("wr_refine", 2)
    kappa = opts.getReal("wr_kappa", 0.1)
    sampler = opts.getString("wr_sampler", "amgmc")
    posterior = opts.getBool("wr_posterior", False)
    nburnin = opts.getInt("wr_nburnin", 500)
    nsamples = opts.getInt("wr_nsamples", 10000)
    chunk = opts.getInt("wr_chunk", nsamples)
    max_time = opts.getReal("wr_max_time", float("inf"))
    reference = opts.getBool("wr_reference", True)
    seed = opts.getInt("wr_seed", 20261112)
    out_json = opts.getString("wr_out", "")
    out_chain = opts.getString("wr_chain", "")

    if alpha not in (1, 2):
        raise SystemExit("-wr_alpha must be 1 or 2")
    if chunk <= 0 or nsamples <= 0:
        raise SystemExit("-wr_nsamples and -wr_chunk must be positive")

    # Sampler options; anything given explicitly on the command line wins.
    defaults = sampler_options(sampler, alpha, posterior)
    for key, val in defaults.items():
        if not opts.hasName(key):
            opts[key] = val

    pymgmc.seed(seed)

    t0 = perf_counter()
    coarse = fd.Mesh(mesh_file)
    mesh = fd.MeshHierarchy(coarse, refine)[-1]
    degree = 1 if alpha == 1 else 2
    V = fd.FunctionSpace(mesh, "CG", degree)

    a = prior_form(V, kappa, alpha)
    L = fd.Constant(0) * fd.TestFunction(V) * fd.dx
    x = fd.Function(V, name="sample")
    A = fd.assemble(a).M.handle
    t_assemble = perf_counter() - t0

    # DM + appctx wiring so that PCGAMGMC's internal PCMG can coarsen through the DM
    # and Galerkin-assemble the coarse operators (only used by the geometric sampler).
    problem = fd.LinearVariationalProblem(a, L, x, constant_jacobian=True)
    ctx = solving_utils._SNESContext(problem, mat_type="aij", pmat_type="aij",
                                     appctx={}, options_prefix="")
    dm = problem.dm
    owner = SimpleNamespace()

    keep = ()  # keep posterior B/S alive for the lifetime of the LRC operator
    if posterior:
        operator, rhs, B, S = setup_posterior(V, A)
        keep = (B, S)
    else:
        operator = A
        rhs = A.createVecs()[0]
        rhs.zeroEntries()

    m = qoi_vector(V)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("")
    ksp.setOperators(operator, operator)
    ksp.setDM(dm)
    ksp.setDMActive(PETSc.KSP.DMActive.OPERATOR, False)

    # The outer KSP is a fixed Richardson driver; the PC is the sampler.
    opts["ksp_type"] = "richardson"
    opts["ksp_max_it"] = 1
    opts["ksp_convergence_test"] = "skip"
    opts["ksp_initial_guess_nonzero"] = True
    opts["ksp_norm_type"] = "none"
    with dmhooks.add_hooks(dm, owner, appctx=ctx, save=False):
        ksp.setFromOptions()

    def run_iterations(n):
        ksp.setTolerances(max_it=n)
        with dmhooks.add_hooks(dm, owner, appctx=ctx):
            with x.dat.vec as xv:
                ksp.solve(rhs, xv)

    # --- setup: KSPSetUp + one iteration -------------------------------------------
    stage = PETSc.Log.Stage("wr_setup")
    stage.push()
    comm.Barrier()
    t0 = perf_counter()
    with dmhooks.add_hooks(dm, owner, appctx=ctx):
        ksp.setUp()
    t_setup = comm.allreduce(perf_counter() - t0, op=MPI.MAX)
    stage.pop()

    # Untimed view of the configuration.  This has to happen before the first
    # iteration: for the posterior, PCGAMGMC then swaps the level operators for MATLRC
    # matrices, for which PCView_GAMG's complexity computation is not supported.
    if rank == 0:
        print("-" * 72)
        print("KSP/PC configuration after setup:")
        sys.stdout.flush()
    try:
        ksp.view()
    except PETSc.Error as err:
        if rank == 0:
            print(f"(KSPView failed: {err})")
    sys.stdout.flush()

    stage.push()
    comm.Barrier()
    t0 = perf_counter()
    run_iterations(1)
    t_setup += comm.allreduce(perf_counter() - t0, op=MPI.MAX)
    stage.pop()

    pc = ksp.getPC()
    pc_type = pc.getType()
    try:
        mg_levels = pc.getMGLevels() if pc_type == "gamgmc" else None
    except PETSc.Error:
        mg_levels = None

    # --- burn-in ----------------------------------------------------------------------
    stage = PETSc.Log.Stage("wr_burnin")
    stage.push()
    comm.Barrier()
    t0 = perf_counter()
    if nburnin > 0:
        run_iterations(nburnin)
    t_burnin = comm.allreduce(perf_counter() - t0, op=MPI.MAX)
    stage.pop()

    # --- sampling ---------------------------------------------------------------------
    # The samplers do not agree on the index they pass to the callback (some count per
    # KSPSolve, some since creation), so we keep our own counter.
    chain = np.zeros(nsamples)
    count = [0]

    def callback(it, xv):
        chain[count[0]] = m.dot(xv)
        count[0] += 1

    pymgmc.PCSetSampleCallback(pc, callback)

    stage = PETSc.Log.Stage("wr_sampling")
    stage.push()
    comm.Barrier()
    t0 = perf_counter()
    done = 0
    stopped_early = False
    while done < nsamples:
        n = min(chunk, nsamples - done)
        run_iterations(n)
        done += n
        elapsed = comm.allreduce(perf_counter() - t0, op=MPI.MAX)
        if done < nsamples and elapsed >= max_time:
            stopped_early = True
            break
    t_sampling = comm.allreduce(perf_counter() - t0, op=MPI.MAX)
    stage.pop()
    if count[0] != done:
        raise RuntimeError(f"sample callback called {count[0]} times for {done} iterations")
    chain = chain[:done]

    # --- diagnostics ------------------------------------------------------------------
    tau_raw = float(emcee.autocorr.integrated_time(chain, quiet=True)[0])
    tau = max(tau_raw, 1.0)
    qoi_mean = float(np.mean(chain))
    qoi_var = float(np.var(chain, ddof=1))
    t_iter = t_sampling / done
    t_indep = t_iter * tau

    ref = None
    if reference:
        t0 = perf_counter()
        mean_ex, var_ex, ref_its = exact_reference(operator, A, m, rhs, posterior)
        ref = {
            "qoi_mean": mean_ex, "qoi_var": var_ex, "cg_iterations": ref_its,
            "time": perf_counter() - t0,
            # Monte Carlo checks, using the estimated IACT for the effective sample size
            "var_ratio": qoi_var / var_ex,
            "var_ratio_se": float(np.sqrt(2.0 * tau / done)),
            "mean_z": (qoi_mean - mean_ex) / float(np.sqrt(var_ex * tau / done)),
        }

    del keep  # posterior B/S referenced until here

    if rank != 0:
        return

    record = {
        "driver": "wrench_refinement.py",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "problem": {
            "mesh": os.path.abspath(mesh_file), "alpha": alpha, "degree": degree,
            "refine": refine, "kappa": kappa,
            "mode": "posterior" if posterior else "prior",
            "ndofs": V.dim(), "ncells": mesh.num_cells(),
            "nnz": int(A.getInfo()["nz_used"]),
            "qoi": {"center": QOI_CENTER, "radius": QOI_RADIUS},
            "observations": {"centers": OBS_CENTERS.tolist(), "radii": OBS_RADII.tolist(),
                             "values": OBS_VALUES.tolist(), "sigma2": OBS_SIGMA2}
            if posterior else None,
        },
        "sampler": {"name": sampler, "pc_type": pc_type, "mg_levels": mg_levels,
                    "options": defaults},
        "run": {
            "seed": seed, "nburnin": nburnin, "nsamples_requested": nsamples,
            "nsamples": done, "chunk": chunk, "max_time": max_time,
            "stopped_early": stopped_early, "nranks": comm.size,
        },
        "timings": {
            "assemble": t_assemble, "setup": t_setup, "burnin": t_burnin,
            "sampling": t_sampling, "per_iteration": t_iter, "per_indep_sample": t_indep,
        },
        "chain": {
            "iact_raw": tau_raw, "iact": tau, "n_over_iact": done / tau,
            "reliable_iact": done >= 50 * tau,
            "qoi_mean": qoi_mean, "qoi_var": qoi_var,
        },
        "reference": ref,
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "cpus_allowed": proc_status("Cpus_allowed_list"),
            "mems_allowed": proc_status("Mems_allowed_list"),
            "git": git_info(),
            "driver_md5": md5(os.path.abspath(__file__)),
            "pymgmc": pymgmc.__file__, "pymgmc_md5": md5(pymgmc.__file__),
            "libparmgmc": loaded_library("libparmgmc"),
            "libparmgmc_md5": md5(loaded_library("libparmgmc") or ""),
            "petsc": ".".join(str(v) for v in PETSc.Sys.getVersion()),
            "petsc4py": version("petsc4py"), "firedrake": version("firedrake"),
            "emcee": version("emcee"), "numpy": np.__version__,
            "python": sys.version.split()[0],
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "argv": sys.argv, "cmdline_options": cmdline_opts,
        },
    }

    print("=" * 72)
    print(f"wrench refinement: {record['problem']['mode']}, sampler = {sampler} "
          f"(pc_type {pc_type}, mg levels {mg_levels})")
    print(f"  alpha / kappa / refine : {alpha} / {kappa} / {refine}")
    print(f"  DOFs / nnz             : {V.dim()} / {record['problem']['nnz']}")
    print(f"  burn-in / samples      : {nburnin} / {done} (requested {nsamples}"
          f"{', stopped early' if stopped_early else ''})")
    print(f"  setup / burn-in [s]    : {t_setup:.4f} / {t_burnin:.4f}")
    print(f"  sampling [s]           : {t_sampling:.4f}")
    print(f"  time per iteration [s] : {t_iter:.6e}")
    print(f"  IACT (raw) / N/IACT    : {tau:.4f} ({tau_raw:.4f}) / {done / tau:.1f}"
          f"{'' if done >= 50 * tau else '  [UNRELIABLE: N < 50 IACT]'}")
    print(f"  time / indep. sample   : {t_indep:.6e} s")
    print(f"  QOI mean / var         : {qoi_mean:.6e} / {qoi_var:.6e}")
    if ref is not None:
        print(f"  exact mean / var       : {ref['qoi_mean']:.6e} / {ref['qoi_var']:.6e}"
              f" (CG its {ref['cg_iterations']})")
        print(f"  var ratio / mean z     : {ref['var_ratio']:.4f} +- "
              f"{ref['var_ratio_se']:.4f} / {ref['mean_z']:+.2f}")
    print("=" * 72)
    sys.stdout.flush()

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(record, fh, indent=2)
    if out_chain:
        os.makedirs(os.path.dirname(os.path.abspath(out_chain)), exist_ok=True)
        np.save(out_chain, chain)


if __name__ == "__main__":
    main()
