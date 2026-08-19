"""Poisson Gibbs sampler in Firedrake, driven by ParMGMC.

python ex15.py
"""

import firedrake as fd
import pymgmc
import pandas as pd
import numpy as np
from petsc4py import PETSc
from matplotlib import pyplot as plt
import tqdm
import emcee


def get_rongelap_measurements(filename, measurement_interval=300):
    df = pd.read_csv(filename)
    points = np.stack([df[dim].to_numpy() for dim in ("x", "y")]).T
    vom = fd.VertexOnlyMesh(mesh, points, reorder=False)
    W = fd.FunctionSpace(vom, "DG", 0)

    interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
    B = interp.petscmat.transpose()
    counts = np.round(
        measurement_interval
        * df["gamma_counts"].to_numpy()
        / df["measurement_time"].to_numpy()
    )
    background_counts = 0.25 * measurement_interval
    return B, counts, background_counts


def get_synthetic_measurements():
    points = [[0.4, 0.6], [0.8, 0.1], [0.5, 0.5]]
    vom = fd.VertexOnlyMesh(mesh, points, reorder=False)
    W = fd.FunctionSpace(vom, "DG", 0)

    interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
    B = interp.petscmat.transpose()
    counts = np.array([100, 50, 20], dtype=np.float64)
    background_counts = 1
    return B, counts, background_counts


# setup = "rongelap"
setup = "synthetic"

if setup == "rongelap":
    mesh = fd.Mesh("../data/rongelap.msh", dim=2)
    correlation_length = 1000.0  # m
else:
    n = 64
    mesh = fd.UnitSquareMesh(n, n)
    correlation_length = 1.0

V = fd.FunctionSpace(mesh, "CG", 1)
w = fd.TrialFunction(V)
v = fd.TestFunction(V)


kappa_sq = fd.Constant(1 / correlation_length**2)
a = fd.inner(fd.grad(w), fd.grad(v)) * fd.dx + kappa_sq * w * v * fd.dx
y = fd.Function(V, name="sample")

# Attach measurements
if setup == "rongelap":
    B, measured_counts, background_counts = get_rongelap_measurements(
        "../data/rongelap_gamma.csv", measurement_interval=300
    )
else:
    B, measured_counts, background_counts = get_synthetic_measurements()

event_counts = PETSc.Vec().createWithArray(measured_counts)
nu = PETSc.Vec().createWithArray(np.zeros_like(event_counts))

# Construct mean field for RHS
mu_rhs = fd.Function(V).interpolate(fd.Constant(np.log(background_counts)))
f_rhs = fd.action(a, mu_rhs)

# Assemble system matrix

if True:
    A = fd.assemble(a).M.handle
    ksp = PETSc.KSP().create()
    # Construct KSP
    ksp.setOperators(A, A)
    ksp.setOptionsPrefix("")
    opts = PETSc.Options()
    solver_parameters = {
        "ksp_initial_guess_nonzero": True,
        "ksp_type": "richardson",
        "ksp_min_it": 1,
        "ksp_max_it": 1,
        "ksp_convergence_test": "skip",
        "pc_type": "poissongibbs",
        "ksp_view": ":ksp_view.txt",
    }
    for key, value in solver_parameters.items():
        opts[key] = value
    ksp.setFromOptions()
    pc = ksp.getPC()
else:
    problem = fd.LinearVariationalProblem(a, f_rhs, y)
    solver = fd.LinearVariationalSolver(
        problem,
        solver_parameters={
            "ksp_initial_guess_nonzero": True,
            "ksp_type": "richardson",
            "ksp_min_it": 1,
            "ksp_max_it": 1,
            "ksp_convergence_test": "skip",
            "pc_type": "poissongibbs",
            "ksp_view": ":ksp_view.txt",
        },
    )

pymgmc.PCPoissonSetAppCtx(pc, event_counts, nu, B)

n_samples = 1024

if setup == "rongelap":
    points_qoi = [[-3000, -1000]]
else:
    points_qoi = [[0.4, 0.6]]

vom_qoi = fd.VertexOnlyMesh(mesh, points_qoi, reorder=False)
W_qoi = fd.FunctionSpace(vom_qoi, "DG", 0)
chain = []
for k in tqdm.tqdm(range(n_samples)):
    if True:
        with y.dat.vec as u, fd.assemble(f_rhs).dat.vec_ro as b:
            ksp.solve(b, u)
    else:
        solver.solve()

    y_obs = fd.assemble(fd.interpolate(y, W_qoi))
    z = float(np.exp(y_obs.dat.data)[0])
    chain.append(z)
chain = np.asarray(chain)
mean = np.average(chain)
std = np.std(chain)

iact = emcee.autocorr.integrated_time(chain)[0]
print(f"mean = {mean:8.4f}")
print(f"std  = {std:8.4f}")
print(f"iact = {iact:8.4f}")

w = fd.Function(V, name="exp_sample").interpolate(fd.exp(y))
fd.VTKFile("sample.pvd").write(y, w)

plt.clf()
plt.plot(chain, linewidth=2, marker="o", markersize=4)
plt.savefig("observations.pdf", bbox_inches="tight")
