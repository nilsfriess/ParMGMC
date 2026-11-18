"""Write the vessel graph (optionally a crop) into a PETSc binary file for the parallel scripts (run serially, once).

python preprocess_graph.py [-fraction 0.0625] [-output vessel_1-16.dat]
"""

import sys

import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc  # noqa: E402

import graph  # noqa: E402


def main() -> None:
    opts = PETSc.Options()
    fraction = opts.getReal("fraction", 1.0)
    default = "vessel.dat" if fraction == 1 else f"vessel_1-{round(1 / fraction)}.dat"
    output = opts.getString("output", default)

    coords, L, regions, names = graph.load_graph(regions=True)
    if fraction < 1:
        coords, L, regions = graph.crop(coords, L, fraction, regions)
    graph.write(output, coords, L, regions, names)
    print(f"{L.shape[0]} nodes, {L.nnz // 2} edges written to {output}")


if __name__ == "__main__":
    main()
