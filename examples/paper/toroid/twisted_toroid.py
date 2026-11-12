"""Tetrahedral mesh of a twisted toroid (the geometry of MFEM's data/toroid-hex.mesh).

Solid torus with major radius R and a square cross-section of half-diagonal A whose
orientation rotates by TWIST degrees (a multiple of 90) once around the ring.  The mesh is
a structured n_theta x n x n hexahedral grid in (theta, s, t), each hexahedron split into
24 tetrahedra through its centre and face centres.  This split is invariant under rotations
of the cross-section, so the mesh is conforming across the seam where the cross-section
grid is identified with a rotated copy of itself.

build(n, ntheta) returns (coordinates, tetrahedra); toroid_weak_scaling.py uses it to
create the coarse mesh in memory.  Run as a script to write a gmsh (format 2.2) file:
    python twisted_toroid.py out.msh [n] [n_theta]
"""

import sys

import numpy as np

R = 1.0
A = 0.4  # half-diagonal of the square cross-section
TWIST = 270  # degrees, multiple of 90


def build(n, ntheta):
    """Vertex coordinates (nv x 3) and positively oriented tetrahedra (nt x 4)."""
    if TWIST % 90 != 0:
        raise ValueError("TWIST must be a multiple of 90 degrees")
    c = A / np.sqrt(2.0)  # half side of the square
    quarter_turns = (TWIST // 90) % 4

    def point(theta, s, t):
        phi = np.deg2rad(TWIST) * theta / (2 * np.pi)
        u = np.cos(phi) * s - np.sin(phi) * t
        v = np.sin(phi) * s + np.cos(phi) * t
        return ((R + u) * np.cos(theta), (R + u) * np.sin(theta), v)

    coords = []
    vid = {}

    def vertex(key, theta, s, t):
        if key not in vid:
            vid[key] = len(coords)
            coords.append(point(theta, s, t))
        return vid[key]

    def seam_index(i, j):
        """Index of the layer-0 grid point that coincides with grid point (i, j) after one
        full turn around the ring (where the cross-section is rotated by TWIST)."""
        for _ in range(quarter_turns):
            i, j = n - j, i  # rotation of the index grid by +90 degrees
        return i, j

    def grid(i):
        return -c + 2 * c * i / n

    def node(k, i, j):
        if k == ntheta:
            k = 0
            i, j = seam_index(i, j)
        return vertex(("n", k, i, j), 2 * np.pi * k / ntheta, grid(i), grid(j))

    def theta_face(k, i, j):
        """Centre of the face in theta-layer k spanned by grid points (i..i+1, j..j+1)."""
        if k == ntheta:
            corners = [seam_index(i + di, j + dj) for di in (0, 1) for dj in (0, 1)]
            i = min(p[0] for p in corners)
            j = min(p[1] for p in corners)
            k = 0
        return vertex(("ft", k, i, j), 2 * np.pi * k / ntheta, grid(i + 0.5), grid(j + 0.5))

    tets = []
    for k in range(ntheta):
        thm = 2 * np.pi * (k + 0.5) / ntheta
        for i in range(n):
            for j in range(n):
                cn = {(dk, di, dj): node(k + dk, i + di, j + dj)
                      for dk in (0, 1) for di in (0, 1) for dj in (0, 1)}
                centre = vertex(("c", k, i, j), thm, grid(i + 0.5), grid(j + 0.5))
                faces = [  # (face centre, face corners in cyclic order)
                    (theta_face(k, i, j), [cn[0, 0, 0], cn[0, 1, 0], cn[0, 1, 1], cn[0, 0, 1]]),
                    (theta_face(k + 1, i, j), [cn[1, 0, 0], cn[1, 1, 0], cn[1, 1, 1], cn[1, 0, 1]]),
                    (vertex(("fs", k, i, j), thm, grid(i), grid(j + 0.5)),
                     [cn[0, 0, 0], cn[1, 0, 0], cn[1, 0, 1], cn[0, 0, 1]]),
                    (vertex(("fs", k, i + 1, j), thm, grid(i + 1), grid(j + 0.5)),
                     [cn[0, 1, 0], cn[1, 1, 0], cn[1, 1, 1], cn[0, 1, 1]]),
                    (vertex(("fu", k, i, j), thm, grid(i + 0.5), grid(j)),
                     [cn[0, 0, 0], cn[1, 0, 0], cn[1, 1, 0], cn[0, 1, 0]]),
                    (vertex(("fu", k, i, j + 1), thm, grid(i + 0.5), grid(j + 1)),
                     [cn[0, 0, 1], cn[1, 0, 1], cn[1, 1, 1], cn[0, 1, 1]]),
                ]
                for fc, cyc in faces:
                    for m in range(4):
                        tets.append((centre, fc, cyc[m], cyc[(m + 1) % 4]))
    coords = np.array(coords)
    tets = np.array(tets)
    x = coords[tets]
    vol = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    return coords, tets


def write_msh(path, coords, tets):
    with open(path, "w") as fh:
        fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        fh.write(f"$Nodes\n{len(coords)}\n")
        for i, (x, y, z) in enumerate(coords, start=1):
            fh.write(f"{i} {x:.16g} {y:.16g} {z:.16g}\n")
        fh.write("$EndNodes\n")
        fh.write(f"$Elements\n{len(tets)}\n")
        for i, tet in enumerate(tets, start=1):
            fh.write(f"{i} 4 2 1 1 {' '.join(str(v + 1) for v in tet)}\n")
        fh.write("$EndElements\n")


if __name__ == "__main__":
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ntheta = int(sys.argv[3]) if len(sys.argv) > 3 else 11 * n
    coords, tets = build(n, ntheta)
    write_msh(out, coords, tets)
    print(f"{out}: {len(coords)} vertices, {len(tets)} tetrahedra (n={n}, ntheta={ntheta})")
