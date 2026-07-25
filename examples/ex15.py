"""Poisson Gibbs sampler in Firedrake, driven by ParMGMC.

python ex15.py
"""

import firedrake as fd
import pymgmc
import pandas as pd
import numpy as np
from petsc4py import PETSc

mesh = fd.Mesh("../data/rongelap.msh", dim=2)

V = fd.FunctionSpace(mesh, "CG", 1)
w = fd.TrialFunction(V)
v = fd.TestFunction(V)

kappa_sq = fd.Constant(1.0e-3)
a = fd.inner(fd.grad(w), fd.grad(v)) * fd.dx + kappa_sq * w * v * fd.dx
f = fd.Constant(0) * v * fd.dx
y = fd.Function(V, name="sample")

df = pd.read_csv("../data/rongelap_gamma.csv")
points = np.stack([df[dim].to_numpy() for dim in ("x", "y")]).T
n_obs = points.shape[0]
print(f"Number of measurements = {n_obs}")
vom = fd.VertexOnlyMesh(mesh, points)
W = fd.FunctionSpace(vom, "DG", 0)

interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
B = interp.petscmat.transpose()
measured_rates = PETSc.Vec().createWithArray(
    df["gamma_counts"].to_numpy() / df["measurement_time"].to_numpy()
)
nu = PETSc.Vec().createWithArray(np.zeros_like(df["gamma_counts"].to_numpy()))


problem = fd.LinearVariationalProblem(a, f, y)
solver = fd.LinearVariationalSolver(
    problem,
    solver_parameters={
        "ksp_type": "richardson",
        "ksp_min_it": 10,
        "ksp_max_it": 10,
        "ksp_convergence_test": "skip",
        "pc_type": "gamgmc",
        "pc_gamgmc_mg_type": "gamg",
        "gamgmc_mg_levels_pc_type": "poissongibbs",
        "gamgmc_mg_coarse_pc_type": "poissongibbs",
        "dm_refine_hierarchy": "2",
        "ksp_view": ":ksp_view.txt",
    },
)


pc = solver.snes.getKSP().getPC()
pymgmc.PCPoissonSetAppCtx(pc, measured_rates, nu, B)

solver.solve()

fd.VTKFile("sample.pvd").write(y)
