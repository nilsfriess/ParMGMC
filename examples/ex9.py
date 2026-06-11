"""Squared shifted-Laplace sampler in Firedrake, driven by ParMGMC.

Draws samples from (kappa^2 - Delta)^2 -- a C0 interior-penalty CG-2
discretisation, SPD with natural boundary conditions -- with a geometric MGMC
sampler: ASMStarPC patch-Cholesky smoothers on every level and Galerkin coarse
operators A_c = P^T A P.

Importing ``assembled_transfer`` makes Firedrake's geometric-MG interpolation
assembled (AIJ) rather than matrix-free, so ``-pc_mg_galerkin both`` forms the
coarse operators by itself

    python ex9.py
"""

import firedrake as fd
import assembled_transfer
import pymgmc

coarse = fd.Mesh("../data/lshape.msh")
mesh = fd.MeshHierarchy(coarse, 2)[-1]

degree = 2
V = fd.FunctionSpace(mesh, "CG", degree)
w = fd.TrialFunction(V)
v = fd.TestFunction(V)

kappa_sq = fd.Constant(100)
alpha = fd.Constant(8.0)
n = fd.FacetNormal(mesh)
h = 2.0 * fd.avg(fd.CellVolume(mesh)) / fd.avg(fd.FacetArea(mesh))
lap_w = fd.div(fd.grad(w))
lap_v = fd.div(fd.grad(v))
a = (fd.inner(lap_w, lap_v) * fd.dx
    + (alpha / h * fd.inner(fd.jump(fd.grad(w), n), fd.jump(fd.grad(v), n))
       - fd.avg(lap_w) * fd.jump(fd.grad(v), n)
       - fd.jump(fd.grad(w), n) * fd.avg(lap_v)
      ) * fd.dS
    + 2 * kappa_sq * fd.inner(fd.grad(w), fd.grad(v)) * fd.dx
    + kappa_sq ** 2 * w * v * fd.dx
)
f = fd.Constant(0) * v * fd.dx
y = fd.Function(V, name="sample")

problem = fd.LinearVariationalProblem(a, f, y)
solver = fd.LinearVariationalSolver(
    problem,
    options_prefix="",
    solver_parameters={
        "ksp_type": "richardson", "ksp_max_it": 10,
        "pc_type": "gamgmc", "pc_gamgmc_mg_type": "mg",
        "gamgmc": {
            "mg_levels": {
                "ksp_type": "richardson", "ksp_max_it": 1,
                "pc_type": "python", "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_construct_dim": 0,
                "pc_star_sub_pc_asm_local_type": "multiplicative",
                "pc_star_sub_sub_ksp_type": "richardson",
                "pc_star_sub_sub_ksp_max_it": 1,
                "pc_star_sub_sub_pc_type": "cholsampler",
            },
            "mg_coarse": {
                "ksp_type": "richardson", "ksp_max_it": 1,
                "pc_type": "cholsampler",
            }
        }
    },
    # solver_parameters={
    #     "ksp_type": "richardson",
    #     "ksp_max_it": 100,
    #     "pc_type": "gamgmc",
    #     "gamgmc" : {
    #         "pc_mg_type": "full",
    #         "pc_mg_cycle_type": "v",
    #         "mg_levels_ksp_max_it": 100,
    #     }
    # }
)
solver.solve()

fd.VTKFile("sample.pvd").write(y)
