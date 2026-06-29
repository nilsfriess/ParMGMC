"""Shifted-Laplace sampler in Firedrake, driven by ParMGMC.

Draws samples from (kappa^2 - Delta) -- a standard CG-1 discretisation, SPD with
natural boundary conditions -- using an algebraic MGMC sampler. PETSc's GAMG
builds the multigrid hierarchy, so no MeshHierarchy or assembled transfers are
needed: a SOR-Gibbs random smoother runs on every level and a Cholesky sampler
on the coarse level.

    python ex10.py
"""

import firedrake as fd
import pymgmc

coarse = fd.Mesh("../data/bunny.msh")
mesh = fd.MeshHierarchy(coarse, 0)[-1]

V = fd.FunctionSpace(mesh, "CG", 1)
w = fd.TrialFunction(V)
v = fd.TestFunction(V)

kappa_sq = fd.Constant(100)
a = fd.inner(fd.grad(w), fd.grad(v)) * fd.dx + kappa_sq * w * v * fd.dx
f = fd.Constant(0) * v * fd.dx
y = fd.Function(V, name="sample")

problem = fd.LinearVariationalProblem(a, f, y)
solver = fd.LinearVariationalSolver(
    problem,
    solver_parameters={
      "ksp_type": "richardson", "ksp_max_it": 10,
      "pc_type": "gamg",
      "mg_levels": {
        "ksp_type": "richardson", "ksp_max_it": 1,
        "pc_type": "sorgibbs",
      },
      "mg_coarse": {
        "ksp_type": "richardson", "ksp_max_it": 1,
        "pc_type": "cholsampler",
      },
    },
)
solver.solve()

rank = fd.Function(fd.FunctionSpace(mesh, "DG", 0), name="rank")
rank.dat.data[:] = mesh.comm.rank
fd.VTKFile("sample.pvd").write(y, rank)
