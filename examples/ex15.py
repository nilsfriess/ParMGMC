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


def get_rongelap_measurements(filename):
    df = pd.read_csv(filename)
    points = np.stack([df[dim].to_numpy() for dim in ("x", "y")]).T
    vom = fd.VertexOnlyMesh(mesh, points, reorder=False)
    W = fd.FunctionSpace(vom, "DG", 0)

    interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
    B_meas = interp.petscmat.transpose()
    event_counts = df["gamma_counts"].to_numpy()
    measurement_times = df["measurement_time"].to_numpy()
    background_rate = 1

    return B_meas, event_counts, measurement_times, background_rate


def get_synthetic_measurements():
    points = [[0.4, 0.6], [0.8, 0.1], [0.7, 0.3]]
    vom = fd.VertexOnlyMesh(mesh, points, reorder=False)
    W = fd.FunctionSpace(vom, "DG", 0)

    interp = fd.assemble(fd.interpolate(fd.TrialFunction(V), W))
    B_meas = interp.petscmat.transpose()
    event_counts = np.array([400, 300, 250], dtype=np.float64)
    measurement_times = np.array([300, 300, 300], dtype=np.float64)
    background_rate = 1
    return B_meas, event_counts, measurement_times, background_rate


setup = "rongelap"
# setup = "synthetic"

if setup == "rongelap":
    mesh = fd.Mesh("../data/rongelap.msh", dim=2)
    correlation_length = 500.0  # m
else:
    n = 64
    mesh = fd.UnitSquareMesh(n, n)
    correlation_length = 0.1

V = fd.FunctionSpace(mesh, "CG", 1)
w = fd.TrialFunction(V)
v = fd.TestFunction(V)


kappa_sq = fd.Constant(1 / correlation_length**2)
a = fd.inner(fd.grad(w), fd.grad(v)) * fd.dx + kappa_sq * w * v * fd.dx
y = fd.Function(V, name="sample")

# Attach measurements
if setup == "rongelap":
    B_meas, measured_counts, measurement_times, background_rate = (
        get_rongelap_measurements("../data/rongelap_gamma.csv")
    )
else:
    B_meas, measured_counts, measurement_times, background_rate = (
        get_synthetic_measurements()
    )

event_counts = PETSc.Vec().createWithArray(measured_counts)
measurement_times = PETSc.Vec().createWithArray(np.zeros_like(measurement_times))

# Construct mean field for RHS
mu_rhs = fd.Function(V).interpolate(fd.Constant(0))


# Assemble system matrix
Q_prec = fd.assemble(a).M.handle
# Construct SNES
snes = PETSc.SNES().create()
snes.setOptionsPrefix("")
snes.setFromOptions()
beta = np.log(background_rate)
pymgmc.SNESPoissonSetAppCtx(snes, event_counts, measurement_times, beta, Q_prec, B_meas)

n_samples = 128

if setup == "rongelap":
    points_qoi = [[-3000, -1000]]
else:
    points_qoi = [[0.5, 0.5]]

vom_qoi = fd.VertexOnlyMesh(mesh, points_qoi, reorder=False)
W_qoi = fd.FunctionSpace(vom_qoi, "DG", 0)
chain = []
y_samples = []
for k in tqdm.tqdm(range(n_samples)):
    with fd.assemble(fd.action(a, mu_rhs)).dat.vec_ro as f_rhs, y.dat.vec as u:
        snes.solve(f_rhs, u)

    y_obs = fd.assemble(fd.interpolate(y, W_qoi))
    z = float(np.exp(y_obs.dat.data)[0])
    chain.append(z)
    _y = y.copy(deepcopy=True)
    _y.rename(f"sample_{k:04d}")
    y_samples.append(_y)
chain = np.asarray(chain)
mean = np.average(chain)
std = np.std(chain)

iact = emcee.autocorr.integrated_time(chain, quiet=True)[0]
print(f"mean = {mean:8.4f}")
print(f"std  = {std:8.4f}")
print(f"iact = {iact:8.4f}")

w = fd.Function(V, name="exp_sample").interpolate(fd.exp(y))
fd.VTKFile("sample.pvd").write(*y_samples, y, w)

plt.clf()
plt.plot(chain, linewidth=2, marker="o", markersize=4)
plt.savefig("observations.pdf", bbox_inches="tight")
