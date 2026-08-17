"""Poisson Gibbs sampler in Firedrake, driven by ParMGMC.

python ex15.py
"""

import firedrake as fd
import pymgmc
import pandas as pd
import numpy as np
from petsc4py import PETSc


def get_rongelap_measurements(filename, measurement_interval=300):
    df = pd.read_csv(filename)
    points = np.stack([df[dim].to_numpy() for dim in ("x", "y")]).T
    vom = fd.VertexOnlyMesh(mesh, points)
    W = fd.FunctionSpace(vom, "DG", 0)

    interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
    B = interp.petscmat.transpose()
    counts = np.round(
        measurement_interval
        * df["gamma_counts"].to_numpy()
        / df["measurement_time"].to_numpy()
    )
    return B, counts


def get_synthetic_measurements():
    points = [[0.4, 0.6], [0.8, 0.1]]
    vom = fd.VertexOnlyMesh(mesh, points)
    W = fd.FunctionSpace(vom, "DG", 0)

    interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
    B = interp.petscmat.transpose()
    counts = np.array([100, 100], dtype=np.float64)
    return B, counts


setup = "rongelap"
setup = "synthetic"

if setup == "rongelap":
    mesh = fd.Mesh("../data/rongelap.msh", dim=2)
    correlation_length = 100.0  # m
else:
    n = 128
    mesh = fd.UnitSquareMesh(n, n)
    correlation_length = 0.1

V = fd.FunctionSpace(mesh, "CG", 1)
w = fd.TrialFunction(V)
v = fd.TestFunction(V)


kappa_sq = fd.Constant(1 / correlation_length**2)
a = fd.inner(fd.grad(w), fd.grad(v)) * fd.dx + kappa_sq * w * v * fd.dx
f = fd.Constant(0) * v * fd.dx
y = fd.Function(V, name="sample")


problem = fd.LinearVariationalProblem(a, f, y)

solver = fd.LinearVariationalSolver(
    problem,
    solver_parameters={
        "snes_type": "ksponly",
        "ksp_initial_guess_nonzero": True,
        "ksp_type": "richardson",
        "ksp_min_it": 1,
        "ksp_max_it": 1,
        "ksp_convergence_test": "skip",
        "pc_type": "poissongibbs",
        "ksp_view": ":ksp_view.txt",
    },
)

ksp = solver.snes.getKSP()
pc = ksp.getPC()

# attach measurements
if setup == "rongelap":
    B, measured_counts = get_rongelap_measurements(
        "../data/rongelap_gamma.csv", measurement_interval=300
    )
else:
    B, measured_counts = get_synthetic_measurements()
    B_dense = PETSc.Mat()
    B.convert("dense", B_dense)
    print(B_dense.getDenseArray())

event_counts = PETSc.Vec().createWithArray(measured_counts)

nu = PETSc.Vec().createWithArray(np.zeros_like(event_counts))


pymgmc.PCPoissonSetAppCtx(pc, event_counts, nu, B)

n_samples = 128

vom_qoi = fd.VertexOnlyMesh(mesh, [[0.8, 0.1]])
W_qoi = fd.FunctionSpace(vom_qoi, "DG", 0)
solver.solve()
obs = []
for k in range(n_samples):
    print(f"sample {k + 1:6d} of {n_samples:6d}")
    with y.dat.vec as u, fd.assemble(f).dat.vec_ro as b:
        ksp.solve(b, u)

    y_obs = fd.assemble(fd.interpolate(y, W_qoi))
    print(np.exp(y_obs.dat.data))
    obs.append(np.exp(y_obs.dat.data))
print(np.average(np.asarray(obs)))

w = fd.Function(V, name="exp_sample").interpolate(y)

fd.VTKFile("sample.pvd").write(y)
