"""Sample the Whittle-Matérn prior on the vessel graph for several correlation lengths 1/kappa.

The sampler is a Richardson iteration whose preconditioner is the actual sampler, configured from the command
line (default: algebraic MGMC). The QoI is the length-weighted mean of the field over the vessels inside a ball
around the centre of the brain; its exact variance w^T Q^{-1} w is computed for comparison.

    python preprocess_graph.py [-fraction 0.0625]   # once, writes vessel.dat (or vessel_1-16.dat)
    mpirun -n 4 python graph_prior.py [-graph vessel.dat] [-pc_type gamgmc|sorgibbs|cholsampler] [-kappas 0.1,0.01]
"""

import sys
import time

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
import pymgmc  # noqa: E402
from emcee.autocorr import integrated_time  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402


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


def sample(A: PETSc.Mat, b: PETSc.Vec, w: PETSc.Vec, nburnin: int, nsamples: int) -> tuple:
    """Sample N(A^{-1} b, A^{-1}) starting from zero.

    Returns the QoI w^T x of each sample after burn-in, the time per sample and the last sample.
    """
    ksp = make_sampler(A)
    x = b.duplicate()
    x.zeroEntries()

    ksp.setTolerances(max_it=nburnin)
    ksp.solve(b, x)

    qoi = np.zeros(nsamples)

    def callback(it: int, sample: PETSc.Vec) -> None:
        qoi[it] = w.dot(sample)

    pymgmc.PCSetSampleCallback(ksp.getPC(), callback)
    ksp.setTolerances(max_it=nsamples)
    A.getComm().barrier()
    t = time.perf_counter()
    ksp.solve(b, x)
    A.getComm().barrier()
    t_sample = (time.perf_counter() - t) / max(nsamples, 1)
    ksp.destroy()
    return qoi, t_sample, x


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
    pymgmc.seed(opts.getInt("seed", 1))

    G = graph.load(filename)
    w = graph.ball_mean_weights_petsc(G, radius)
    size = G.L.getComm().getSize()
    PETSc.Sys.Print(f"{filename}: n = {G.n}, {size} ranks, {graph.cut_fraction(G.L):.2%} of edges cut")

    for kappa in kappas:
        A = graph.precision_petsc(G.L, kappa)

        b = A.createVecLeft()
        b.zeroEntries()  # prior: mean zero
        qoi, t_sample, _ = sample(A, b, w, nburnin, nsamples)
        tau = integrated_time(qoi, quiet=True)[0]
        var = exact_variance(A, w)
        PETSc.Sys.Print(
            f"kappa {kappa:.0e} (1/kappa = {1 / kappa:.0f}): {1e3 * t_sample:.1f} ms/sample, IACT {tau:.1f}, "
            f"{1e3 * t_sample * tau:.1f} ms/independent sample | QoI mean {qoi.mean():+.3f} "
            f"(+- {2 * np.sqrt(var * tau / nsamples):.3f}), var {qoi.var():.4f} (exact {var:.4f})"
        )
        A.destroy()


if __name__ == "__main__":
    main()
