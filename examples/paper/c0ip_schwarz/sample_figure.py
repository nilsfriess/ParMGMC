"""Figure with one sample each of the alpha = 1 (P1, precision K + kappa^2 M) and alpha = 2 (C0IP P2,
precision (kappa^2 - Delta)^2) GMRFs on the unit square, drawn with the exact Cholesky sampler
(PCCHOLSAMPLER) from the same seed. Serial.

    python sample_figure.py -N 128 -kappa 20 -out samples_unit_square.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc
import firedrake as fd
import pymgmc

opts = PETSc.Options()
N, kappa = opts.getInt("N", 128), opts.getReal("kappa", 20.0)
pymgmc.seed(opts.getInt("seed", 1))
mesh = fd.UnitSquareMesh(N, N)
k2 = fd.Constant(kappa ** 2)


def draw(V, a):
    A = fd.assemble(a, mat_type="aij").M.handle
    ksp = PETSc.KSP().create()
    ksp.setOperators(A, A)
    ksp.setType("richardson")
    ksp.setTolerances(max_it=1)
    ksp.getPC().setType("cholsampler")
    b, x = A.createVecs()
    b.zeroEntries()
    ksp.solve(b, x)
    f = fd.Function(V)
    with f.dat.vec_wo as fv:
        x.copy(fv)
    return f


V1 = fd.FunctionSpace(mesh, "CG", 1)
u, v = fd.TrialFunction(V1), fd.TestFunction(V1)
f1 = draw(V1, (k2 * u * v + fd.inner(fd.grad(u), fd.grad(v))) * fd.dx)

V2 = fd.FunctionSpace(mesh, "CG", 2)
u, v = fd.TrialFunction(V2), fd.TestFunction(V2)
n, h = fd.FacetNormal(mesh), fd.avg(fd.CellDiameter(mesh))


def d2n(w):
    H = fd.grad(fd.grad(w))
    return 0.5 * (fd.dot(fd.dot(H("+"), n("+")), n("+")) + fd.dot(fd.dot(H("-"), n("+")), n("+")))


a2 = ((fd.inner(fd.grad(fd.grad(u)), fd.grad(fd.grad(v))) + 2 * k2 * fd.inner(fd.grad(u), fd.grad(v)) + k2 ** 2 * u * v) * fd.dx
      - (d2n(u) * fd.jump(fd.grad(v), n) + d2n(v) * fd.jump(fd.grad(u), n)) * fd.dS
      + fd.Constant(20.0) / h * fd.jump(fd.grad(u), n) * fd.jump(fd.grad(v), n) * fd.dS)
f2 = fd.Function(V1).interpolate(draw(V2, a2))  # vertex values for plotting

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True)
for ax, f, title in zip(axes, (f1, f2), (r"$\alpha=1$", r"$\alpha=2$")):
    c = fd.tripcolor(f, axes=ax, cmap="viridis", rasterized=True)
    fig.colorbar(c, ax=ax, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(title)
fig.savefig(opts.getString("out", "samples_unit_square.pdf"), dpi=200)
