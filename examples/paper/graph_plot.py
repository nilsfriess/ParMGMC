"""Render one prior sample on the vessel graph: the vessels in a thin slab, coloured by the field.

The field is sampled on the whole (cropped) graph with the same sampler as graph_prior.py; only the slab is drawn.

    mpirun -n 4 python graph_plot.py [-kappa 0.01] [-fraction 0.0625] [-axis 2] [-thickness 100] [-output sample.png]
"""

import os
import sys

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
import pymgmc  # noqa: E402
import pyvista as pv  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402
from graph_prior import make_sampler  # noqa: E402


def draw_sample(A: PETSc.Mat, nsteps: int) -> PETSc.Vec:
    """State of the chain after nsteps iterations started from zero."""
    ksp = make_sampler(A)
    b, x = A.createVecs()
    b.zeroEntries()
    x.zeroEntries()
    ksp.setTolerances(max_it=nsteps)
    ksp.solve(b, x)
    ksp.destroy()
    return x


def gather(x: PETSc.Vec, rows: PETSc.IS, n: int) -> np.ndarray | None:
    """The redistributed vector x in the original numbering, on rank 0 (None elsewhere)."""
    comm = rows.getComm().tompi4py()
    idx = comm.gather(rows.getIndices(), root=0)
    val = comm.gather(x.getArray(), root=0)
    if comm.rank != 0:
        return None
    out = np.empty(n)
    out[np.concatenate(idx)] = np.concatenate(val)
    return out


def render(coords: graph.Coords, L, u: np.ndarray, axis: int, thickness: float, output: str) -> None:
    """Draw the edges with both endpoints within thickness/2 of the middle plane normal to ``axis``."""
    mid = (coords[:, axis].min() + coords[:, axis].max()) / 2
    inside = np.abs(coords[:, axis] - mid) <= thickness / 2
    E = L.tocoo()
    keep = (E.row < E.col) & inside[E.row] & inside[E.col]
    nodes = np.flatnonzero(inside)
    new = np.full(len(coords), -1)
    new[nodes] = np.arange(len(nodes))
    lines = np.column_stack([np.full(keep.sum(), 2), new[E.row[keep]], new[E.col[keep]]]).ravel()

    mesh = pv.PolyData(coords[nodes], lines=lines)
    mesh.point_data["u"] = u[nodes]
    lim = np.quantile(np.abs(u[nodes]), 0.99)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(window_size=(2000, 2000))
    plotter.add_mesh(
        mesh,
        scalars="u",
        cmap="RdBu_r",
        clim=(-lim, lim),
        line_width=2,
        render_lines_as_tubes=True,
        show_scalar_bar=False,
    )
    view = ["yz", "xz", "xy"][axis]
    getattr(plotter, f"view_{view}")()
    plotter.screenshot(output, transparent_background=True)
    print(f"{keep.sum()} edges in the slab, written to {output}")


def main() -> None:
    opts = PETSc.Options()
    kappa = opts.getReal("kappa", 1e-2)
    fraction = opts.getReal("fraction", 1 / 16)
    axis = opts.getInt("axis", 2)
    thickness = opts.getReal("thickness", 100.0)
    nsteps = opts.getInt("nsteps", 20)
    output = opts.getString("output", "sample.png")
    sample_file = opts.getString("sample_file", output.rsplit(".", 1)[0] + ".npy")  # reused if it exists
    pymgmc.seed(opts.getInt("seed", 1))

    coords, L = graph.load_graph()
    if fraction < 1:
        coords, L = graph.crop(coords, L, fraction)
    if os.path.exists(sample_file):
        u = np.load(sample_file) if PETSc.COMM_WORLD.getRank() == 0 else None
    else:
        Q = graph.to_petsc(graph.precision(L, kappa))
        rows = graph.partition(Q)
        A = graph.redistribute(Q, rows)
        Q.destroy()
        u = gather(draw_sample(A, nsteps), rows, L.shape[0])
        if u is not None:
            np.save(sample_file, u)
    if u is not None:
        render(coords, L, u, axis, thickness, output)


if __name__ == "__main__":
    main()
