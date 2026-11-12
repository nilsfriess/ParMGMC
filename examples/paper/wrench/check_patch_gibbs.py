"""Consistency checks for PCPATCHGIBBS / patch_gibbs.StarPatchGibbsPC (sequential).

1. Single-DOF patches 0..n-1 in deterministic mode reproduce MatSOR (forward, backward and
   symmetric sweeps with zero initial guess) on the alpha=1 (CG-1) wrench matrix.
2. StarPatchGibbsPC in deterministic mode reproduces ASMStarPC (PCASM multiplicative, exact patch
   solves) on the alpha=2 (CG-2, C0IP) wrench matrix: one smoother application, and one V-cycle of
   a geometric multigrid solver (MeshHierarchy, Galerkin coarse operators) with either smoother.

Usage: python check_patch_gibbs.py [-refine 1]
Prints the relative differences and exits with status 1 if any exceeds the tolerance.
"""

import sys
from types import SimpleNamespace

import assembled_transfer  # noqa: F401  (assembled prolongations, needed for the Galerkin V-cycle)
import firedrake as fd
import numpy as np
import pymgmc  # noqa: F401  (registers the ParMGMC PCs)
import wrench_refinement as wr
from firedrake import dmhooks, solving_utils
from firedrake.petsc import PETSc

TOL = 1e-9
failures = []


def report(name, ref, val):
    diff = ref.copy()
    diff.axpy(-1.0, val)
    rel = diff.norm() / ref.norm()
    ok = rel < TOL
    if not ok:
        failures.append(name)
    print(f"  {name:60s} relative difference {rel:.3e}  {'ok' if ok else 'FAILED'}")


def mesh_space(refine, alpha):
    mesh = fd.MeshHierarchy(fd.Mesh(wr.MESH_FILE), refine)[-1]
    V = fd.FunctionSpace(mesh, "CG", 1 if alpha == 1 else 2)
    return V


def check_sor(refine):
    print(f"1. single-DOF patches vs MatSOR (alpha=1, refine {refine})")
    V = mesh_space(refine, 1)
    A = fd.assemble(wr.prior_form(V, 0.1, 1)).M.handle
    n = A.getSize()[0]
    rng = np.random.default_rng(1)
    b = A.createVecs()[0]
    b.array[:] = rng.standard_normal(n)
    sweeps = {"forward": PETSc.Mat.SORType.FORWARD_SWEEP,
              "backward": PETSc.Mat.SORType.BACKWARD_SWEEP,
              "symmetric": PETSc.Mat.SORType.SYMMETRY_SWEEP}
    for name, sortype in sweeps.items():
        x_sor = b.duplicate()
        x_sor.zeroEntries()
        A.SOR(b, x_sor, omega=1.0, sortype=sortype | PETSc.Mat.SORType.ZERO_INITIAL_GUESS)

        pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
        pc.setOptionsPrefix(f"chk_sor_{name}_")
        pc.setType("patchgibbs")
        pc.setOperators(A, A)
        pymgmc.PCPatchGibbsSetPatches(pc, np.arange(n + 1, dtype=PETSc.IntType), np.arange(n, dtype=PETSc.IntType))
        opts = PETSc.Options(f"chk_sor_{name}_")
        opts["pc_patchgibbs_sweep"] = name
        opts["pc_patchgibbs_deterministic"] = True
        pc.setFromOptions()
        x_pg = b.duplicate()
        pc.apply(b, x_pg)
        report(f"{name} sweep", x_sor, x_pg)
        pc.destroy()


def star_options(prefix, smoother):
    opts = PETSc.Options(prefix)
    opts["pc_type"] = "python"
    opts["pc_star_construct_dim"] = 0
    if smoother == "asm":
        opts["pc_python_type"] = "firedrake.ASMStarPC"
        opts["pc_star_sub_pc_asm_local_type"] = "multiplicative"
        opts["pc_star_sub_sub_ksp_type"] = "preonly"
        opts["pc_star_sub_sub_pc_type"] = "cholesky"
    else:
        opts["pc_python_type"] = "patch_gibbs.StarPatchGibbsPC"
        opts["pc_star_pgibbs_pc_patchgibbs_deterministic"] = True


def check_star(refine):
    print(f"2. StarPatchGibbsPC vs ASMStarPC (alpha=2, refine {refine})")
    V = mesh_space(refine, 2)
    a = wr.prior_form(V, 0.1, 2)
    A = fd.assemble(a).M.handle
    n = A.getSize()[0]
    rng = np.random.default_rng(2)
    b = A.createVecs()[0]
    b.array[:] = rng.standard_normal(n)

    # one smoother application on the finest level
    ys = {}
    for smoother in ("asm", "patch"):
        pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
        pc.setOptionsPrefix(f"chk_one_{smoother}_")
        pc.setDM(V.dm)
        pc.setOperators(A, A)
        star_options(f"chk_one_{smoother}_", smoother)
        pc.setFromOptions()
        pc.setUp()
        ys[smoother] = b.duplicate()
        pc.apply(b, ys[smoother])
        pc.destroy()
    report("one smoother application (multiplicative, exact patch solves)", ys["asm"], ys["patch"])

    # one V-cycle of a geometric multigrid solver with either smoother
    x = fd.Function(V)
    L = fd.Constant(0) * fd.TestFunction(V) * fd.dx
    problem = fd.LinearVariationalProblem(a, L, x, constant_jacobian=True)
    ctx = solving_utils._SNESContext(problem, mat_type="aij", pmat_type="aij", appctx={}, options_prefix="")
    dm = problem.dm
    owner = SimpleNamespace()
    for smoother in ("asm", "patch"):
        prefix = f"chk_mg_{smoother}_"
        opts = PETSc.Options(prefix)
        opts["pc_type"] = "mg"
        opts["pc_mg_levels"] = refine + 1
        opts["pc_mg_galerkin"] = "both"
        opts["mg_levels_ksp_type"] = "richardson"
        opts["mg_levels_ksp_max_it"] = 1
        opts["mg_coarse_ksp_type"] = "preonly"
        opts["mg_coarse_pc_type"] = "cholesky"
        star_options(prefix + "mg_levels_", smoother)
        pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
        pc.setOptionsPrefix(prefix)
        pc.setDM(dm)
        pc.setOperators(A, A)
        with dmhooks.add_hooks(dm, owner, appctx=ctx, save=(smoother != "asm")):
            pc.setFromOptions()
        with dmhooks.add_hooks(dm, owner, appctx=ctx):
            pc.setUp()
            ys[smoother] = b.duplicate()
            pc.apply(b, ys[smoother])
        pc.destroy()
    report("one V-cycle, geometric MG with Galerkin coarse operators", ys["asm"], ys["patch"])


if __name__ == "__main__":
    refine = PETSc.Options().getInt("refine", 1)
    check_sor(refine + 1)
    check_star(refine)
    print("all checks passed" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)
