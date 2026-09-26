"""Solve Q x = b on the vessel graph with a random b, to time a solver independently of any sampler.

Q is the Whittle-Matérn precision for one kappa, assembled as in graph_prior.py. The KSP is configured entirely from
the command line (prefix none), e.g. for a direct solve with MKL Pardiso / MUMPS:

    mpirun -n 8 python graph_solve.py -kappa 0.01 -mat_type sbaij -ksp_type preonly -pc_type cholesky \\
        -pc_factor_mat_solver_type mkl_cpardiso -nsolves 10 -log_view

-mat_type converts Q after assembly (it is built as AIJ); MKL Pardiso needs sbaij for a parallel Cholesky.

Setup (e.g. the factorisation) and the solves are timed separately; with -nsolves > 1 the same KSP solves again
with new random right-hand sides, which is what a direct sampler does once per sample.
"""

import sys
import time

import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402


def main() -> None:
    opts = PETSc.Options()
    kappa = opts.getReal("kappa", 1e-2)
    filename = opts.getString("graph", "vessel.dat")  # written by preprocess_graph.py
    nsolves = opts.getInt("nsolves", 1)

    graph.log(f"loading {filename} ...")
    G = graph.load(filename)
    comm = G.L.getComm()
    graph.log(f"{filename}: n = {G.n}, {comm.getSize()} ranks, {G.info()['edges_cut']:.2%} of edges cut")
    A = graph.precision_petsc(G.L, kappa)
    mat_type = opts.getString("mat_type", "")  # e.g. sbaij, which mkl_cpardiso needs for a parallel Cholesky
    if mat_type:
        A = A.convert(mat_type)
        A.setOption(PETSc.Mat.Option.SPD, True)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setFromOptions()
    x, b = A.createVecs()
    rng = PETSc.Random().create(comm)
    rng.setFromOptions()

    comm.barrier()
    t = time.perf_counter()
    ksp.setUp()
    comm.barrier()
    t_setup = time.perf_counter() - t
    graph.log(f"kappa {kappa:g}: KSP ({ksp.getType()}, {ksp.getPC().getType()}) set up in {t_setup:.3f} s")

    t_solve = 0.0
    for i in range(nsolves):
        b.setRandom(rng)
        comm.barrier()
        t = time.perf_counter()
        ksp.solve(b, x)
        comm.barrier()
        dt = time.perf_counter() - t
        t_solve += dt
        graph.log(f"  solve {i + 1}: {dt:.4f} s, {ksp.getIterationNumber()} iterations, {ksp.getConvergedReason()}")
    graph.log(f"setup {t_setup:.3f} s, {t_solve / nsolves:.4f} s per solve ({nsolves} solves)")


if __name__ == "__main__":
    main()
