"""SPDE sampler on the wrench, for comparison with the alpha=2 MGMC samplers of wrench_refinement.py.

Standard SPDE approach (Lindgren et al.) for the Matern field with precision (kappa^2 - Delta)^2:
P1 finite elements with lumped mass matrix M_L on the wrench refined -wr_refine times, natural
(Neumann) boundary conditions, L = K + kappa^2 M.  The discrete prior has precision
Q = L M_L^{-1} L, i.e. covariance Q^{-1} = L^{-1} M_L L^{-1}, and independent prior samples are
x0 = L^{-1} (M_L^{1/2} z), z ~ N(0, I): one solve with L per sample (CG + GAMG, relative tolerance
-wr_rtol on the unpreconditioned residual).

Posterior (-wr_posterior; same observations as wrench_refinement.py): conditioning by kriging,
    x = x0 + W S^{-1} (y + eps - B^T x0),  W = Q^{-1} B,  S = B^T W + Sigma,  eps ~ N(0, Sigma),
which is Algorithm (SplittingPosteriorSampler) with the exact prior sampler (M = A).  W needs two
solves per observation once (part of the setup); a posterior sample then costs one solve plus a
rank-m update.  Samples are independent (IACT 1 up to the solver tolerance).

The exact QOI moments of this discretisation are computed with solves to relative tolerance 1e-12:
prior Var[q] = m^T w with w = Q^{-1} m; posterior Var[q] = m^T w - g^T S^{-1} g, E[q] = g^T S^{-1} y
with g = B^T w.

Writes the same JSON record format as wrench_refinement.py (collect.py reads both).

Example:
    python wrench_spde.py -wr_refine 3 -wr_kappa 0.1 -wr_posterior -wr_nsamples 500 -wr_out spde.json
"""

import datetime
import os
import socket
import sys
from time import perf_counter

import emcee
import firedrake as fd
import numpy as np
import pymgmc  # noqa: F401  (for provenance: the same library as the other runs)
import wrench_refinement as wr
from firedrake.petsc import PETSc
from mpi4py import MPI


def make_solver(L, rtol, prefix):
    ksp = PETSc.KSP().create(comm=L.getComm())
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(L, L)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=rtol, atol=0.0, max_it=5000)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setErrorIfNotConverged(True)
    ksp.setFromOptions()
    return ksp


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise SystemExit("wrench_spde.py is sequential")
    opts = PETSc.Options()
    cmdline_opts = opts.getAll()

    mesh_file = opts.getString("wr_mesh", wr.MESH_FILE)
    refine = opts.getInt("wr_refine", 2)
    kappa = opts.getReal("wr_kappa", 0.1)
    posterior = opts.getBool("wr_posterior", False)
    obs_sigma2 = opts.getReal("wr_obs_sigma2", wr.OBS_SIGMA2)
    nburnin = opts.getInt("wr_nburnin", 2)
    nsamples = opts.getInt("wr_nsamples", 500)
    rtol = opts.getReal("wr_rtol", 1e-8)
    reference = opts.getBool("wr_reference", True)
    seed = opts.getInt("wr_seed", 20261112)
    label = opts.getString("wr_label", "spde")
    out_json = opts.getString("wr_out", "")
    out_chain = opts.getString("wr_chain", "")

    rng = np.random.default_rng(seed)

    t0 = perf_counter()
    mesh = fd.MeshHierarchy(fd.Mesh(mesh_file), refine)[-1]
    V = fd.FunctionSpace(mesh, "CG", 1)
    L = fd.assemble(wr.prior_form(V, kappa, 1)).M.handle  # kappa^2 M + K
    ml = fd.assemble(fd.TestFunction(V) * fd.dx)  # lumped mass (row sums of M)
    with ml.dat.vec_ro as v:
        ml_vec = v.copy()
    sqrt_ml = ml_vec.copy()
    sqrt_ml.sqrtabs()
    m = wr.qoi_vector(V)
    t_assemble = perf_counter() - t0

    nobs = len(wr.OBS_CENTERS)
    B = None
    keep = ()
    if posterior:
        _, _, B, Sinv = wr.setup_posterior(V, L, obs_sigma2)  # B: n x nobs dense, Sinv = Sigma^{-1} (diagonal)
        keep = (Sinv,)
        sigma2 = obs_sigma2
        y_obs = np.array(wr.OBS_VALUES)

    x0, b, t = L.createVecs()[0], L.createVecs()[0], L.createVecs()[0]
    x = x0.duplicate()
    tmp_obs = B.createVecs()[0] if posterior else None  # length nobs
    upd = x0.duplicate()

    def apply_cov(v, out, ksp):
        """out = Q^{-1} v = L^{-1} M_L L^{-1} v"""
        ksp.solve(v, t)
        t.pointwiseMult(t, ml_vec)
        ksp.solve(t, out)

    # --- setup: GAMG setup and, for the posterior, W = Q^{-1} B and the Cholesky factor of S ----
    t0 = perf_counter()
    ksp = make_solver(L, rtol, "wr_spde_")
    ksp.setUp()
    W = None
    if posterior:
        W = B.duplicate()
        col_out = x0.duplicate()
        for k in range(nobs):
            bk = B.getDenseColumnVec(k, mode="r")
            apply_cov(bk, col_out, ksp)
            B.restoreDenseColumnVec(k, mode="r")
            wk = W.getDenseColumnVec(k, mode="w")
            col_out.copy(wk)
            W.restoreDenseColumnVec(k, mode="w")
        W.assemble()
        BtW = np.zeros((nobs, nobs))
        for k in range(nobs):
            wk = W.getDenseColumnVec(k, mode="r")
            B.multTranspose(wk, tmp_obs)
            BtW[:, k] = tmp_obs.array
            W.restoreDenseColumnVec(k, mode="r")
        S_mat = 0.5 * (BtW + BtW.T) + sigma2 * np.eye(nobs)
        S_chol = np.linalg.cholesky(S_mat)
    t_setup = perf_counter() - t0
    if not posterior:
        S_chol = None

    def solve_S(d):
        return np.linalg.solve(S_chol.T, np.linalg.solve(S_chol, d))

    c_vec = B.createVecs()[0] if posterior else None  # length nobs (right vector), W.mult input
    its_total = [0]

    def draw():
        b.array[:] = rng.standard_normal(b.getLocalSize())
        b.pointwiseMult(b, sqrt_ml)
        ksp.solve(b, x0)
        its_total[0] += ksp.getIterationNumber()
        if not posterior:
            return x0
        B.multTranspose(x0, tmp_obs)
        d = y_obs + np.sqrt(sigma2) * rng.standard_normal(nobs) - tmp_obs.array
        c_vec.array[:] = solve_S(d)
        W.mult(c_vec, upd)
        x0.copy(x)
        x.axpy(1.0, upd)
        return x

    for _ in range(nburnin):
        draw()
    its_total[0] = 0

    chain = np.zeros(nsamples)
    comm.Barrier()
    t0 = perf_counter()
    for i in range(nsamples):
        chain[i] = m.dot(draw())
    t_sampling = perf_counter() - t0

    tau_raw = float(emcee.autocorr.integrated_time(chain, quiet=True)[0])
    tau = max(tau_raw, 1.0)
    qoi_mean = float(np.mean(chain))
    qoi_var = float(np.var(chain, ddof=1))
    t_iter = t_sampling / nsamples

    ref = None
    if reference:
        t0 = perf_counter()
        ksp_ref = make_solver(L, 1e-12, "wr_spde_ref_")
        w = m.duplicate()
        apply_cov(m, w, ksp_ref)
        var_ex = m.dot(w)
        mean_ex = 0.0
        if posterior:
            B.multTranspose(w, tmp_obs)
            g = tmp_obs.array.copy()
            var_ex -= float(g @ solve_S(g))
            mean_ex = float(g @ solve_S(y_obs))
        ksp_ref.destroy()
        ref = {"qoi_mean": float(mean_ex), "qoi_var": float(var_ex), "cg_iterations": None,
               "time": perf_counter() - t0,
               "var_ratio": qoi_var / var_ex, "var_ratio_se": float(np.sqrt(2.0 * tau / nsamples)),
               "mean_z": (qoi_mean - mean_ex) / float(np.sqrt(var_ex * tau / nsamples))}

    try:
        mg_levels = ksp.getPC().getMGLevels()
    except PETSc.Error:
        mg_levels = None
    del keep

    record = {
        "driver": "wrench_spde.py",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "problem": {
            "mesh": os.path.abspath(mesh_file), "alpha": 2, "degree": 1,
            "discretisation": "SPDE: P1, lumped mass, Q = L M_L^-1 L",
            "refine": refine, "kappa": kappa, "mode": "posterior" if posterior else "prior",
            "ndofs": V.dim(), "ncells": mesh.num_cells(), "nnz": int(L.getInfo()["nz_used"]),
            "qoi": {"center": wr.QOI_CENTER, "radius": wr.QOI_RADIUS},
            "observations": {"centers": wr.OBS_CENTERS.tolist(), "radii": wr.OBS_RADII.tolist(),
                             "values": wr.OBS_VALUES.tolist(), "sigma2": obs_sigma2}
            if posterior else None,
        },
        "sampler": {"name": "spde", "label": label, "pc_type": "spde (CG + GAMG)",
                    "mg_levels": mg_levels, "options": {"rtol": rtol}},
        "run": {"seed": seed, "nburnin": nburnin, "nsamples_requested": nsamples,
                "nsamples": nsamples, "chunk": nsamples, "max_time": None,
                "stopped_early": False, "nranks": comm.size},
        "timings": {"assemble": t_assemble, "setup": t_setup, "burnin": None,
                    "sampling": t_sampling, "per_iteration": t_iter, "per_indep_sample": t_iter * tau,
                    "cg_iterations_per_sample": its_total[0] / nsamples},
        "chain": {"iact_raw": tau_raw, "iact": tau, "n_over_iact": nsamples / tau,
                  "reliable_iact": nsamples >= 50 * tau, "qoi_mean": qoi_mean, "qoi_var": qoi_var},
        "reference": ref,
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "cpus_allowed": wr.proc_status("Cpus_allowed_list"),
            "git": wr.git_info(), "driver_md5": wr.md5(os.path.abspath(__file__)),
            "libparmgmc": wr.loaded_library("libparmgmc"),
            "petsc": ".".join(str(v) for v in PETSc.Sys.getVersion()),
            "petsc4py": wr.version("petsc4py"), "firedrake": wr.version("firedrake"),
            "numpy": np.__version__, "python": sys.version.split()[0],
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "argv": sys.argv, "cmdline_options": cmdline_opts,
        },
    }

    print("=" * 72)
    print(f"wrench SPDE sampler: {record['problem']['mode']}, P1 + lumped mass, kappa {kappa}, refine {refine}")
    print(f"  DOFs / GAMG levels     : {V.dim()} / {mg_levels}")
    print(f"  setup / sampling [s]   : {t_setup:.4f} / {t_sampling:.4f}")
    print(f"  time per sample [s]    : {t_iter:.6e}  (CG its per sample {its_total[0] / nsamples:.1f}, rtol {rtol:g})")
    print(f"  IACT (raw)             : {tau:.4f} ({tau_raw:.4f})")
    print(f"  QOI mean / var         : {qoi_mean:.6e} / {qoi_var:.6e}")
    if ref is not None:
        print(f"  exact mean / var       : {ref['qoi_mean']:.6e} / {ref['qoi_var']:.6e}")
        print(f"  var ratio / mean z     : {ref['var_ratio']:.4f} +- {ref['var_ratio_se']:.4f} / {ref['mean_z']:+.2f}")
    print("=" * 72, flush=True)

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            import json
            json.dump(record, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if out_chain:
        os.makedirs(os.path.dirname(os.path.abspath(out_chain)), exist_ok=True)
        np.save(out_chain, chain)


if __name__ == "__main__":
    main()
