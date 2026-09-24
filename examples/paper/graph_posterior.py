"""Sample the posterior of a linear Bayesian inverse problem on the vessel graph.

Prior: the Whittle-Matérn field of graph_prior.py with precision Q. Observations: the length-weighted mean of the
field over each brain region of the atlas, y = B^T u* + noise, for a "true" field u* drawn from the prior and
Gaussian noise with standard deviation ``-noise`` times the RMS of B^T u*. The posterior N(A^{-1} f, A^{-1}) has
precision A = Q + B Sigma^{-1} B^T (a low-rank update, MATLRC) and f = B Sigma^{-1} y.

The QoI is the same ball average as for the prior; its exact posterior mean and variance are computed for comparison.

    python preprocess_graph.py [-fraction 0.0625]   # once, writes vessel.dat (or vessel_1-16.dat)
    mpirun -n 4 python graph_posterior.py [-graph vessel.dat] [-pc_type gamgmc|sorgibbs] [-kappas 0.1,0.01]
"""

import sys

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
import pymgmc  # noqa: E402
from emcee.autocorr import integrated_time  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402
from graph_prior import sample  # noqa: E402

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
    noise = opts.getReal("noise", 0.1)  # noise std relative to the RMS of the noise-free observations
    nburnin = opts.getInt("nburnin", 100)
    nsamples = opts.getInt("nsamples", 1000)
    seed = opts.getInt("seed", 1)
    pymgmc.seed(seed)
    comm = PETSc.COMM_WORLD

    G = graph.load(filename)
    w = graph.ball_mean_weights_petsc(G, radius)
    B = observation_operator(G)
    PETSc.Sys.Print(f"{filename}: n = {G.n}, {B.getSize()[1]} observed regions, {comm.getSize()} ranks")

    for kappa in kappas:
        Q = graph.precision_petsc(G.L, kappa)

        # Synthetic data from a prior sample
        zero = Q.createVecLeft()
        zero.zeroEntries()
        _, _, truth = sample(Q, zero, w, nburnin, 0)
        y, _ = B.createVecs()
        B.multTranspose(truth, y)
        sigma = noise * np.sqrt(y.dot(y) / y.getSize())
        rng = np.random.default_rng(seed + comm.getRank())
        y.array[:] += sigma * rng.standard_normal(y.getLocalSize())

        # Posterior precision A = Q + B Sigma^{-1} B^T and right-hand side f = B Sigma^{-1} y
        S = y.duplicate()
        S.set(1 / sigma**2)
        A = PETSc.Mat().createLRC(Q, B, S, None)
        f = Q.createVecLeft()
        Sy = y.duplicate()
        Sy.pointwiseMult(S, y)
        B.mult(Sy, f)

        qoi, t_sample, _ = sample(A, f, w, nburnin, nsamples)
        tau = integrated_time(qoi, quiet=True)[0]
        mean = w.dot(solve(A, Q, f))
        var = w.dot(solve(A, Q, w))
        var_prior = w.dot(solve(Q, Q, w))
        PETSc.Sys.Print(
            f"kappa {kappa:.0e}: {1e3 * t_sample:.1f} ms/sample, IACT {tau:.1f}, "
            f"{1e3 * t_sample * tau:.1f} ms/independent sample\n"
            f"    QoI: truth {w.dot(truth):+.4f}, posterior mean {qoi.mean():+.4f} "
            f"(+- {2 * np.sqrt(var * tau / nsamples):.4f}, exact {mean:+.4f}), "
            f"var {qoi.var():.2e} (exact {var:.2e}, prior {var_prior:.2e})"
        )
        A.destroy()
        Q.destroy()


if __name__ == "__main__":
    main()
