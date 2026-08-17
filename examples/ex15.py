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

correlation_length = 100  # m
kappa_sq = fd.Constant(1 / correlation_length**2)
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
B_dense = PETSc.Mat()
measurement_interval = 300
measured_counts = PETSc.Vec().createWithArray(
    np.round(
        measurement_interval
        * df["gamma_counts"].to_numpy()
        / df["measurement_time"].to_numpy()
    )
)
print(np.asarray(measured_counts))
background_rate = 0.4
nu = PETSc.Vec().createWithArray(
    np.log(measurement_interval * background_rate)
    * np.ones_like(df["gamma_counts"].to_numpy())
)

problem = fd.LinearVariationalProblem(a, f, y)
solver = fd.LinearVariationalSolver(
    problem,
    solver_parameters={
        # "snes_type": "ksponly",
        "ksp_type": "richardson",
        "ksp_min_it": 1,
        "ksp_max_it": 1,
        "ksp_convergence_test": "skip",
        "pc_type": "poissongibbs",
        # "pc_type": "gamgmc",
        # "gamgmc_mg_levels_pc_type": "poissongibbs",
        # "pc_gamgmc_mg_type": "gamg",
        # "gamgmc_mg_coarse_pc_type": "poissongibbs",
        # "dm_refine_hierarchy": "2",
        "ksp_view": ":ksp_view.txt",
    },
)


pc = solver.snes.getKSP().getPC()
pymgmc.PCPoissonSetAppCtx(pc, measured_counts, nu, B)

n_samples = 32
observation_points = [[-5545.0, -3160.0]]
vom_obs = fd.VertexOnlyMesh(mesh, observation_points)
W_qoi = fd.FunctionSpace(vom_obs, "DG", 0)

obs = []

for k in range(n_samples):
    print(f"sample {k + 1:6d} of {n_samples:6d}")
    solver.solve()
    # y_obs = fd.assemble(fd.interpolate(y, W_qoi))
    # obs.append(y_obs.dat.data[0])

y_meas = fd.assemble(fd.interpolate(y, W))
print(-y_meas.dat.data + nu)
from matplotlib import pyplot as plt

plt.clf()
plt.plot(obs, linewidth=2, marker="o")
plt.savefig("observations.pdf", bbox_inches="tight")

w = fd.Function(V, name="exp_sample").interpolate(y)

fd.VTKFile("sample.pvd").write(w)
