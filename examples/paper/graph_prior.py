"""Sample the Whittle-Matérn prior on the vessel graph for several correlation lengths 1/kappa.

The sampler is a Richardson iteration whose preconditioner is the actual sampler, configured from the command
line (default: algebraic MGMC). The QoI is the length-weighted mean of the field over the vessels inside a ball
around the centre of the brain; its exact variance w^T Q^{-1} w is computed for comparison.

    python preprocess_graph.py [-fraction 0.0625]   # once, writes vessel.dat (or vessel_1-16.dat)
    mpirun -n 4 python graph_prior.py [-graph vessel.dat] [-pc_type gamgmc|sorgibbs|cholsampler] [-kappas 0.1,0.01]
"""

import logging
import sys
import time

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
import pymgmc  # noqa: E402
from emcee.autocorr import integrated_time  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402

if PETSc.COMM_WORLD.getRank() != 0:  # emcee's warnings about short chains once, not once per rank
    logging.getLogger("emcee").setLevel(logging.ERROR)


def make_sampler(A: PETSc.Mat) -> PETSc.KSP:
    """Richardson iteration with a sampler as preconditioner; every iteration produces one sample."""
    opts = PETSc.Options()
    for key, value in {"ksp_type": "richardson", "pc_type": "gamgmc", "ksp_convergence_test": "skip"}.items():
        if not opts.hasName(key):
            opts[key] = value
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setInitialGuessNonzero(True)  # continue the chain from the current sample
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    ksp.setFromOptions()
    return ksp


def run_chain(ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec, n: int, label: str, w: PETSc.Vec | None) -> np.ndarray:
    """Advance the chain in x by n steps, printing progress every 10 %; returns the QoI w^T x of each step."""
    qoi = np.zeros(n)
    step = max(n // 10, 1)

    def callback(it: int, sample: PETSc.Vec) -> None:
        if w is not None:
            qoi[it] = w.dot(sample)
        if (it + 1) % step == 0:
            graph.log(f"    {label}: {100 * (it + 1) // n:3d}% ...")

    pymgmc.PCSetSampleCallback(ksp.getPC(), callback)
    ksp.setTolerances(max_it=n)
    ksp.solve(b, x)
    return qoi


def sample(A: PETSc.Mat, b: PETSc.Vec, w: PETSc.Vec | None, nburnin: int, nsamples: int) -> tuple:
    """Sample N(A^{-1} b, A^{-1}) starting from zero.

    Returns the QoI w^T x of each sample after burn-in (if w is given), a dict with the sampler type and the
    timings (KSP setup, burn-in, sampling, per sample) and the last sample.
    """
    comm = A.getComm()
    ksp = make_sampler(A)
    x = b.duplicate()
    x.zeroEntries()
    stats = {"sampler": ksp.getPC().getType()}

    comm.barrier()
    t = time.perf_counter()
    ksp.setUp()
    comm.barrier()
    stats["t_ksp_setup"] = time.perf_counter() - t
    graph.log(f"  sampler ({stats['sampler']}) set up in {stats['t_ksp_setup']:.1f} s")

    t = time.perf_counter()
    if nburnin > 0:
        run_chain(ksp, b, x, nburnin, f"burn-in ({nburnin} steps)", None)
    comm.barrier()
    stats["t_burnin"] = time.perf_counter() - t

    qoi = np.zeros(nsamples)
    t = time.perf_counter()
    if nsamples > 0:
        qoi = run_chain(ksp, b, x, nsamples, f"sampling ({nsamples} samples)", w)
    comm.barrier()
    stats["t_sampling"] = time.perf_counter() - t
    stats["t_per_sample"] = stats["t_sampling"] / max(nsamples, 1)
    if nsamples > 0:
        graph.log(f"  {nsamples} samples in {stats['t_sampling']:.1f} s")
    ksp.destroy()
    return qoi, stats, x


def iact(qoi: np.ndarray) -> tuple[float, float]:
    """Integrated autocorrelation time of the QoI chain: (raw estimate, estimate clamped to >= 1).

    Values below 1 are estimation artefacts; the clamped value is used for derived quantities.
    """
    tau = integrated_time(qoi, quiet=True)[0]
    return tau, max(tau, 1.0)


def exact_variance(A: PETSc.Mat, w: PETSc.Vec) -> float:
    """w^T A^{-1} w with a tight CG solve (options prefix exact_)."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOptionsPrefix("exact_")
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=1e-10, divtol=1e100)  # the residual can grow a lot before CG converges
    ksp.setErrorIfNotConverged(True)
    ksp.setFromOptions()
    z = w.duplicate()
    ksp.solve(w, z)
    ksp.destroy()
    return w.dot(z)


def main() -> None:
    opts = PETSc.Options()
    kappas = opts.getRealArray("kappas", np.array([1e-1, 1e-2, 1e-3]))
    filename = opts.getString("graph", "vessel.dat")  # written by preprocess_graph.py
    radius = opts.getReal("radius", 200.0)  # QoI ball radius in voxels
    nburnin = opts.getInt("nburnin", 100)
    nsamples = opts.getInt("nsamples", 1000)
    seed = opts.getInt("seed", 1)
    pymgmc.seed(seed)

    graph.log(f"loading {filename} ...")
    t = time.perf_counter()
    G = graph.load(filename)
    t_load = time.perf_counter() - t
    w = graph.ball_mean_weights_petsc(G, radius)
    ginfo = G.info()
    graph.log(f"{filename}: n = {G.n}, {G.L.getComm().getSize()} ranks, {ginfo['edges_cut']:.2%} of edges cut")

    rows = []
    for kappa in kappas:
        graph.log(f"kappa {kappa:.0e}:")
        A = graph.precision_petsc(G.L, kappa)

        b = A.createVecLeft()
        b.zeroEntries()  # prior: mean zero
        qoi, stats, _ = sample(A, b, w, nburnin, nsamples)
        tau_raw, tau = iact(qoi)
        graph.log("  computing the exact QoI variance ...")
        t = time.perf_counter()
        var = exact_variance(A, w)
        t_exact = time.perf_counter() - t
        t_indep = stats["t_per_sample"] * tau
        err = 2 * np.sqrt(var * tau / nsamples)
        graph.log(
            f"kappa {kappa:.0e} (1/kappa = {1 / kappa:.0f}): {1e3 * stats['t_per_sample']:.1f} ms/sample, "
            f"IACT {tau:.1f}, {1e3 * t_indep:.1f} ms/independent sample | QoI mean {qoi.mean():+.3f} "
            f"(+- {err:.3f}), var {qoi.var():.4f} (exact {var:.4f})"
        )
        rows.append(
            {
                **graph.run_info("graph_prior.py"),  # after sampling, so it includes the sampler's default options
                "graph": filename,
                **ginfo,
                "nnz_Q": int(A.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"]),
                "kappa": kappa,
                "seed": seed,
                "nburnin": nburnin,
                "nsamples": nsamples,
                "qoi_radius": radius,
                "qoi_nodes": graph.count_nonzero(w),
                "t_load": t_load,
                **stats,
                "iact_raw": tau_raw,
                "iact": tau,
                "t_per_indep_sample": t_indep,
                "qoi_mean": qoi.mean(),
                "qoi_mean_err_2sigma": err,
                "qoi_var": qoi.var(),
                "qoi_var_exact": var,
                "t_exact": t_exact,
            }
        )
        A.destroy()
    graph.print_csv(rows)


if __name__ == "__main__":
    main()
