"""Squared shifted-Laplace prior on the wrench: MGMC vs Cholesky, with weak scaling.

Draws samples from the smoother Whittle-Matern prior ``(kappa^2 - Delta)^2``
(with a C0 interior-penalty CG-2 discretisation).

The sampler is chosen through PETSc command-line options:

    # Cholesky
    python ex13.py -pc_type cholsampler

    # Geometric MGMC (ASMStarPC patch-Cholesky smoothers)
    python ex13.py -pc_type gamgmc -gamgmc_pc_mg_galerkin both \
        -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_ksp_max_it 1 \
        -gamgmc_mg_levels_pc_type python \
        -gamgmc_mg_levels_pc_python_type firedrake.ASMStarPC \
        -gamgmc_mg_levels_pc_star_construct_dim 0 \
        -gamgmc_mg_levels_pc_star_sub_pc_asm_local_type multiplicative \
        -gamgmc_mg_levels_pc_star_sub_sub_ksp_type richardson \
        -gamgmc_mg_levels_pc_star_sub_sub_ksp_max_it 1 \
        -gamgmc_mg_levels_pc_star_sub_sub_pc_type cholsampler \
        -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_coarse_ksp_max_it 1 \
        -gamgmc_mg_coarse_pc_type cholsampler


Can also sample from the posterior of a Bayesian linear inverse problem via the
command line option `-ex13_posterior`. The observations are integrals over balls
with given centres and radii. Can be used to test the Woodbury sampler, e.g.,

    python ex13.py -ex13_posterior -pc_type woodbury \
        -pc_woodbury_sampler cholsampler -pc_woodbury_solver cholesky

Options:
    -kappa <float>     inverse correlation length      (default 6.0)
    -base_refine <int> refinements on a single rank    (default 2)
    -nburnin <int>     burn-in samples discarded       (default 1000)
    -nsamples <int>    production samples timed         (default 10000)
    -posterior         switch to posterior sampling     (default off)
    -seed <int>        RNG seed                         (default 20260625)
"""

from time import perf_counter
from types import SimpleNamespace

import numpy as np

import firedrake as fd
from firedrake import dmhooks, solving_utils
from firedrake.petsc import PETSc
from mpi4py import MPI

import emcee
import pymgmc
import assembled_transfer


MESH_FILE = "../data/wrench2.msh"

# QOI: L^2 integral of the field over a ball near one end of the wrench (wrench
# coordinates span x in [-18, 15], y in [-24, 252], z in [-5, 5]).
QOI_CENTER = (0.0, 116.0, 0.0)
QOI_RADIUS = 5.0

# Posterior observations: an abstract list of balls (centre, radius) along the
# shaft, each with an observed value -- the field's volume-average over the ball
# -- and a shared Gaussian noise variance.  The prior field has std ~0.5, so the
# O(0.1) values are sensible volume-averages, and the noise std (sqrt(sigma2) =
# 0.05) sits well below the signal, making the observations informative.
OBS_CENTERS = np.array([
    (0.0,  40.0, 0.0),
    (0.0,  90.0, 0.0),
    (0.0, 140.0, 0.0),
    (0.0, 190.0, 0.0),
    (0.0, 228.0, 0.0),
])
OBS_RADII = np.full(len(OBS_CENTERS), 6.0)
OBS_VALUES = np.array([0.10, -0.15, 0.05, -0.10, 0.12])
OBS_SIGMA2 = 2.5e-3


def weak_scaling_refinements(size, base_refine):
    """ How many levels of refinement we need for the given MPI size to make this a weak-scaling run  """
    m = round(size ** (1.0 / 3.0))
    if m ** 3 != size or (m & (m - 1)) != 0:
        raise SystemExit(
            f"ex13 weak scaling requires nranks in {{1, 8, 64, 512, ...}}, got {size}"
        )
    return base_refine + (m.bit_length() - 1)


def prior_form(V, kappa):
    """C0 interior-penalty bilinear form for the squared shifted Laplace."""
    w = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    mesh = V.mesh()
    kappa_sq = fd.Constant(kappa * kappa)
    alpha = fd.Constant(8.0)
    n = fd.FacetNormal(mesh)
    # h_F = 2|T|/|F| is a true facet length scale (robust on stretched tets).
    h = 2.0 * fd.avg(fd.CellVolume(mesh)) / fd.avg(fd.FacetArea(mesh))
    lap_w = fd.div(fd.grad(w))
    lap_v = fd.div(fd.grad(v))
    return (
        fd.inner(lap_w, lap_v) * fd.dx
        + (
            alpha / h * fd.inner(fd.jump(fd.grad(w), n), fd.jump(fd.grad(v), n))
            - fd.avg(lap_w) * fd.jump(fd.grad(v), n)
            - fd.jump(fd.grad(w), n) * fd.avg(lap_v)
        ) * fd.dS
        + 2 * kappa_sq * fd.inner(fd.grad(w), fd.grad(v)) * fd.dx
        + kappa_sq ** 2 * w * v * fd.dx
    )


def make_qoi(V):
    """Return ``qoi(x_vec)`` computing the L^2 integral of x over the QOI ball."""
    x = fd.SpatialCoordinate(V.mesh())
    r_sq = sum((x[i] - QOI_CENTER[i]) ** 2 for i in range(len(QOI_CENTER)))
    chi = fd.conditional(fd.lt(r_sq, QOI_RADIUS ** 2), 1.0, 0.0)
    meas = fd.assemble(chi * fd.TestFunction(V) * fd.dx)

    def qoi(x_vec):
        with meas.dat.vec_ro as m:
            return m.dot(x_vec)

    return qoi


def setup_posterior(V, A):
    """Build ``A_post = A + B Sigma^-1 B^T`` (MATLRC) and RHS ``f = B Sigma^-1 y``.

    Returns ``(A_post, f, B, S)``; the caller must keep ``B`` and ``S`` alive for
    as long as the LRC matrix is used.
    """
    mesh = V.mesh()
    comm = mesh.comm
    gdim = mesh.geometric_dimension
    nobs = len(OBS_CENTERS)

    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    M = fd.assemble(fd.inner(u, v) * fd.dx).M.handle
    vols = (4.0 / 3.0) * np.pi * OBS_RADII ** 3  # 3D ball volumes

    n_local, n_global = A.getLocalSize()[0], A.getSize()[0]
    B = PETSc.Mat().createDense(
        ((n_local, n_global), (PETSc.DECIDE, nobs)), comm=comm
    )
    B.setUp()

    # Column i of B is M @ I[1_ball_i / vol_i], so (B^T u)_i is the L^2 average of
    # u over ball i.
    xsym = fd.SpatialCoordinate(mesh)
    ind = fd.Function(V)
    col = fd.Function(V)
    for i in range(nobs):
        c = fd.as_vector([fd.Constant(float(OBS_CENTERS[i, k])) for k in range(gdim)])
        r = float(OBS_RADII[i])
        ind.interpolate(
            fd.conditional(
                fd.lt(fd.dot(xsym - c, xsym - c), r * r),
                1.0 / float(vols[i]), 0.0,
            )
        )
        with ind.dat.vec_ro as iv, col.dat.vec as cv:
            M.mult(iv, cv)
        with col.dat.vec_ro as src:
            cvec = B.getDenseColumnVec(i, mode="w")
            src.copy(cvec)
            B.restoreDenseColumnVec(i, mode="w")
    B.assemble()

    S, f = B.createVecs()  # S: size nobs, f: size n_dofs
    S.set(1.0 / OBS_SIGMA2)

    # RHS f = B Sigma^-1 y from the hard-coded observed values (the values only
    # shift the posterior mean; the sampler cost and IACT depend on A_post alone).
    y, _ = B.createVecs()
    lo, hi = y.getOwnershipRange()
    y.array[:] = OBS_VALUES[lo:hi]

    Sy = y.duplicate()
    Sy.pointwiseMult(S, y)
    B.mult(Sy, f)

    A_post = PETSc.Mat().createLRC(A, B, S, None)
    return A_post, f, B, S


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    opts = PETSc.Options()

    mesh_file = opts.getString("mesh", MESH_FILE)
    kappa     = opts.getReal("kappa", 6.0)
    base_ref  = opts.getInt("base_refine", 2)
    nburnin   = opts.getInt("nburnin", 1000)
    nsamples  = opts.getInt("nsamples", 10000)
    posterior = opts.getBool("posterior", False)
    seed      = opts.getInt("seed", 20260625)

    pymgmc.seed(seed)

    total_ref = weak_scaling_refinements(comm.size, base_ref)

    coarse = fd.Mesh(mesh_file)
    mesh = fd.MeshHierarchy(coarse, total_ref)[-1]
    V = fd.FunctionSpace(mesh, "CG", 2)

    a = prior_form(V, kappa)
    L = fd.Constant(0) * fd.TestFunction(V) * fd.dx
    x = fd.Function(V, name="sample")
    A = fd.assemble(a).M.handle

    # DM + appctx wiring so PCGAMGMC's internal PCMG can coarsen through the DM
    # and Galerkin-assemble the coarse operators.  The LinearVariationalProblem is
    # built only to construct the _SNESContext (it is never solved); it is unused
    # by -pc_type cholsampler but harmless, so we attach it unconditionally.
    problem = fd.LinearVariationalProblem(a, L, x, constant_jacobian=True)
    ctx = solving_utils._SNESContext(
        problem, mat_type="aij", pmat_type="aij", appctx={}, options_prefix=""
    )
    dm = problem.dm
    owner = SimpleNamespace()  # carries Firedrake's per-solve setup hooks

    keep = ()  # keep posterior B/S alive for the lifetime of the LRC operator
    if posterior:
        operator, rhs, B, S = setup_posterior(V, A)
        keep = (B, S)
    else:
        operator = A
        rhs = A.createVecs()[0]
        rhs.zeroEntries()

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("")
    ksp.setOperators(operator, operator)
    ksp.setDM(dm)
    ksp.setDMActive(PETSc.KSP.DMActive.OPERATOR, False)

    # Outer KSP is a fixed Richardson driver; the PC (the sampler) is configured
    # entirely from the command line.
    opts["ksp_type"] = "richardson"
    opts["ksp_max_it"] = 0
    opts["ksp_convergence_test"] = "skip"
    opts["ksp_initial_guess_nonzero"] = True
    opts["ksp_norm_type"] = "none"
    with dmhooks.add_hooks(dm, owner, appctx=ctx, save=False):
        ksp.setFromOptions()

    qoi = make_qoi(V)
    ndofs = V.dim()
    pc_type = ksp.getPC().getType()

    # Burn-in (untimed).
    ksp.setTolerances(max_it=nburnin)
    with dmhooks.add_hooks(dm, owner, appctx=ctx):
        with x.dat.vec as xv:
            ksp.solve(rhs, xv)

    # Production: record the scalar QOI at every sample.
    qois = np.zeros(nsamples)

    def callback(it, xv):
        qois[it] = qoi(xv)

    pymgmc.PCSetSampleCallback(ksp.getPC(), callback)
    ksp.setTolerances(max_it=nsamples)

    comm.Barrier()
    t0 = perf_counter()
    with dmhooks.add_hooks(dm, owner, appctx=ctx):
        with x.dat.vec as xv:
            ksp.solve(rhs, xv)
    elapsed = comm.allreduce(perf_counter() - t0, op=MPI.MAX)

    tau = float(emcee.autocorr.integrated_time(qois, quiet=True)[0])
    qoi_mean = float(np.mean(qois))
    qoi_var = float(np.var(qois))

    del keep  # references held until here

    if rank != 0:
        return

    mode = "posterior" if posterior else "prior"
    print("=" * 60)
    print(f"ex13: {mode} sampling, pc_type = {pc_type}")
    print(f"  ranks               : {comm.size}")
    print(f"  refinements (total) : {total_ref}")
    print(f"  global DOFs         : {ndofs}")
    print(f"  kappa               : {kappa}")
    print(f"  burn-in / samples   : {nburnin} / {nsamples}")
    print(f"  QOI mean / var      : {qoi_mean:.6e} / {qoi_var:.6e}")
    print(f"  wall time (max rank): {elapsed:.4f} s")
    print(f"  IACT (tau)          : {tau:.4f}")
    print(f"  time / indep. sample: {elapsed * tau / nsamples:.6e} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
