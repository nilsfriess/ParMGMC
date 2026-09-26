"""Render one prior sample on the vessel graph: the vessels in a thin slab, coloured by the field.

The field is sampled on the whole (cropped) graph with the same sampler as graph_prior.py; only the slab is drawn.

    mpirun -n 4 python graph_plot.py [-kappa 0.01] [-fraction 0.0625] [-axis 2] [-thickness 100]
        [-cmap viridis] [-line_width 3] [-output sample.png]
"""

import os
import sys

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
import pymgmc  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402
import plotting  # noqa: E402
from graph_prior import sample  # noqa: E402


def draw_sample(A: PETSc.Mat, nsteps: int) -> PETSc.Vec:
    """State of the chain after nsteps iterations started from zero."""
    b = A.createVecLeft()
    b.zeroEntries()
    return sample(A, b, None, nsteps, 0)[2]


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


def render(coords: graph.Coords, L, u: np.ndarray, axis: int, thickness: float, output: str, opts) -> None:
    """Draw the edges with both endpoints within thickness/2 of the middle plane normal to ``axis`` (see plotting.py).

    The colours show the nodal values of u (blended linearly along each edge), symmetric about zero and clipped at
    the -clip quantile (0.95) of |u| in the slab; the colour bar goes to <output>_colorbar.pdf.
    """
    mid = (coords[:, axis].min() + coords[:, axis].max()) / 2
    inside = np.abs(coords[:, axis] - mid) <= thickness / 2
    E = L.tocoo()
    keep = (E.row < E.col) & inside[E.row] & inside[E.col]
    lim = np.quantile(np.abs(u[inside]), opts.getReal("clip", 0.95))
    cmap = opts.getString("cmap", "viridis")
    plotting.render_lines(
        coords,
        np.column_stack([E.row[keep], E.col[keep]]),
        u,
        output,
        cmap=cmap,
        clim=(-lim, lim),
        view=["yz", "xz", "xy"][axis],
        line_width=opts.getReal("line_width", 3.0),
        scale=opts.getReal("scale", 2.0),
    )
    plotting.colorbar(output.rsplit(".", 1)[0] + "_colorbar.pdf", cmap, (-lim, lim), "$u$", extend="both")


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
        render(coords, L, u, axis, thickness, output, opts)


if __name__ == "__main__":
    main()
