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
    B_meas = interp.petscmat.transpose()
    counts = np.round(
        measurement_interval
        * df["gamma_counts"].to_numpy()
        / df["measurement_time"].to_numpy()
    )
    background_counts = 0.25 * measurement_interval
    return B_meas, counts, background_counts


def get_synthetic_measurements():
    points = [[0.4, 0.6], [0.8, 0.1], [0.7, 0.3]]
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
    correlation_length = 500.0  # m
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
    B_meas, measured_counts, background_counts = get_rongelap_measurements(
        "../data/rongelap_gamma.csv", measurement_interval=300
    )
else:
    B_meas, measured_counts, background_counts = get_synthetic_measurements()

event_counts = PETSc.Vec().createWithArray(measured_counts)
nu = PETSc.Vec().createWithArray(np.zeros_like(event_counts))

# Construct mean field for RHS
mu_rhs = fd.Function(V).interpolate(fd.Constant(np.log(background_counts)))


# Assemble system matrix


Q_prec = fd.assemble(a).M.handle
# Construct SNES
snes = PETSc.SNES().create()
snes.setOptionsPrefix("")
opts = PETSc.Options()
solver_parameters = {
    "snes_type": "poissongibbs",
    "poissongibbs_its": 1,
    "snes_view": ":snes_view.txt",
}
for key, value in solver_parameters.items():
    opts[key] = value
snes.setFromOptions()

with fd.assemble(fd.action(a, mu_rhs)).dat.vec_ro as f_rhs:
    pymgmc.SNESPoissonSetAppCtx(snes, event_counts, Q_prec, B_meas, f_rhs, nu)

n_samples = 1024

if setup == "rongelap":
    points_qoi = [[-3000, -1000]]
else:
    points_qoi = [[0.5, 0.5]]

vom_qoi = fd.VertexOnlyMesh(mesh, points_qoi, reorder=False)
W_qoi = fd.FunctionSpace(vom_qoi, "DG", 0)
chain = []
for k in tqdm.tqdm(range(n_samples)):
    with y.dat.vec as u:
        snes.solve(None, u)

    y_obs = fd.assemble(fd.interpolate(y, W_qoi))
    z = float(np.exp(y_obs.dat.data)[0])
    chain.append(z)
chain = np.asarray(chain)
mean = np.average(chain)
std = np.std(chain)

iact = emcee.autocorr.integrated_time(chain, quiet=True)[0]
print(f"mean = {mean:8.4f}")
print(f"std  = {std:8.4f}")
print(f"iact = {iact:8.4f}")

w = fd.Function(V, name="exp_sample").interpolate(fd.exp(y))
fd.VTKFile("sample.pvd").write(y, w)

plt.clf()
plt.plot(chain, linewidth=2, marker="o", markersize=4)
plt.savefig("observations.pdf", bbox_inches="tight")
