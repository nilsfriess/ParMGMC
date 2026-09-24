"""Sample the posterior of a linear Bayesian inverse problem on the vessel graph.

Prior: the Whittle-Matérn field of graph_prior.py with precision Q. Observations: the length-weighted mean of the
field over each brain region of the atlas, B^T u. The observed values are arbitrary but fixed: y_k = obs_scale * xi_k
with xi_k ~ N(0, 1) from a seeded generator (``-obs_scale``, ``-obs_seed``), the same on every rank; the noise has a
fixed standard deviation ``-sigma``. So every run (any kappa, any number of ranks) solves the same problem; the data
do not affect the cost of sampling anyway. The posterior N(A^{-1} f, A^{-1}) has
precision A = Q + B Sigma^{-1} B^T (a low-rank update, MATLRC) and f = B Sigma^{-1} y.

The QoI is the same ball average as for the prior; its exact posterior mean and variance are computed for comparison.

    python preprocess_graph.py [-fraction 0.0625]   # once, writes vessel.dat (or vessel_1-16.dat)
    mpirun -n 4 python graph_posterior.py [-graph vessel.dat] [-pc_type gamgmc|sorgibbs] [-kappas 0.1,0.01]
"""

import sys
import time

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
import pymgmc  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402
from graph_prior import iact, sample  # noqa: E402

UNOBSERVED = ("bgr", "root")  # background / unassigned atlas labels


def observation_operator(G: graph.DistributedGraph) -> PETSc.Mat:
    """Dense n x m matrix B whose columns are the mean-over-region weights of the observed regions."""
    columns = []
    inside = G.regions.duplicate()
    for k, name in enumerate(G.names):
        if name in UNOBSERVED:
            continue
        inside.getArray()[:] = G.regions.getArray() == k
        w = graph.mean_weights_petsc(G.L, inside)
        if w is not None:  # the region contains at least one whole edge
            columns.append(w)
    B = PETSc.Mat().createDense(((G.L.getLocalSize()[0], G.n), (PETSc.DECIDE, len(columns))), comm=G.L.getComm())
    B.setUp()
    for c, w in enumerate(columns):
        col = B.getDenseColumnVec(c, mode="w")
        w.copy(col)
        B.restoreDenseColumnVec(c, mode="w")
        w.destroy()
    B.assemble()
    return B


def solve(A: PETSc.Mat, P: PETSc.Mat, b: PETSc.Vec) -> PETSc.Vec:
    """A^{-1} b with a tight CG solve preconditioned by GAMG on P (options prefix exact_)."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOptionsPrefix("exact_")
    ksp.setOperators(A, P)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=1e-10, divtol=1e100)  # the residual can grow a lot before CG converges
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)  # GAMG(Q) ignores the low-rank term: use the true residual
    ksp.setErrorIfNotConverged(True)
    ksp.setFromOptions()
    x = b.duplicate()
    ksp.solve(b, x)
    ksp.destroy()
    return x


def main() -> None:
    opts = PETSc.Options()
    kappas = opts.getRealArray("kappas", np.array([1e-1, 1e-2, 1e-3]))
    filename = opts.getString("graph", "vessel.dat")  # written by preprocess_graph.py
    radius = opts.getReal("radius", 200.0)  # QoI ball radius in voxels
    obs_scale = opts.getReal("obs_scale", 0.05)  # observed values y_k = obs_scale * N(0, 1), fixed by obs_seed
    obs_seed = opts.getInt("obs_seed", 0)
    sigma = opts.getReal("sigma", 0.01)  # observation noise standard deviation
    nburnin = opts.getInt("nburnin", 100)
    nsamples = opts.getInt("nsamples", 1000)
    seed = opts.getInt("seed", 1)
    pymgmc.seed(seed)
    comm = PETSc.COMM_WORLD

    graph.log(f"loading {filename} ...")
    t = time.perf_counter()
    G = graph.load(filename)
    t_load = time.perf_counter() - t
    w = graph.ball_mean_weights_petsc(G, radius)
    ginfo = G.info()
    graph.log("building the observation operator ...")
    t = time.perf_counter()
    B = observation_operator(G)
    t_obs = time.perf_counter() - t
    nobs = B.getSize()[1]
    graph.log(f"{filename}: n = {G.n}, {nobs} observed regions, {comm.getSize()} ranks")

    rows = []
    for kappa in kappas:
        graph.log(f"kappa {kappa:.0e}:")
        Q = graph.precision_petsc(G.L, kappa)

        # Posterior precision A = Q + B Sigma^{-1} B^T and right-hand side f = B Sigma^{-1} y
        y, _ = B.createVecs()
        lo, hi = y.getOwnershipRange()
        y.array[:] = obs_scale * np.random.default_rng(obs_seed).standard_normal(y.getSize())[lo:hi]
        S = y.duplicate()
        S.set(1 / sigma**2)
        A = PETSc.Mat().createLRC(Q, B, S, None)
        f = Q.createVecLeft()
        Sy = y.duplicate()
        Sy.pointwiseMult(S, y)
        B.mult(Sy, f)

        graph.log("  sampling the posterior ...")
        qoi, stats, _ = sample(A, f, w, nburnin, nsamples)
        tau_raw, tau = iact(qoi)
        graph.log("  computing the exact posterior mean and variance ...")
        t = time.perf_counter()
        mean = w.dot(solve(A, Q, f))
        var = w.dot(solve(A, Q, w))
        var_prior = w.dot(solve(Q, Q, w))
        t_exact = time.perf_counter() - t
        t_indep = stats["t_per_sample"] * tau
        err = 2 * np.sqrt(var * tau / nsamples)
        graph.log(
            f"kappa {kappa:.0e}: {1e3 * stats['t_per_sample']:.1f} ms/sample, IACT {tau:.1f}, "
            f"{1e3 * t_indep:.1f} ms/independent sample\n"
            f"    QoI: posterior mean {qoi.mean():+.4f} "
            f"(+- {err:.4f}, exact {mean:+.4f}), "
            f"var {qoi.var():.2e} (exact {var:.2e}, prior {var_prior:.2e})"
        )
        rows.append(
            {
                **graph.run_info("graph_posterior.py"),  # after sampling, so it includes the sampler's default options
                "graph": filename,
                **ginfo,
                "nnz_Q": int(Q.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"]),
                "kappa": kappa,
                "seed": seed,
                "nburnin": nburnin,
                "nsamples": nsamples,
                "qoi_radius": radius,
                "qoi_nodes": graph.count_nonzero(w),
                "observations": nobs,
                "obs_scale": obs_scale,
                "obs_seed": obs_seed,
                "sigma": sigma,
                "t_load": t_load,
                "t_observation_operator": t_obs,
                **stats,
                "iact_raw": tau_raw,
                "iact": tau,
                "t_per_indep_sample": t_indep,
                "qoi_mean": qoi.mean(),
                "qoi_mean_err_2sigma": err,
                "qoi_mean_exact": mean,
                "qoi_var": qoi.var(),
                "qoi_var_exact": var,
                "qoi_var_prior_exact": var_prior,
                "t_exact": t_exact,
            }
        )
        A.destroy()
        Q.destroy()
    graph.print_csv(rows)


if __name__ == "__main__":
    main()
