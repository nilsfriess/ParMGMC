"""Pictures of the twisted toroid: one alpha = 1 sample (P1, precision K + kappa^2 M, natural BCs, exact
Cholesky sample) and the partition into -nparts parts. The partition is the one the -nparts-rank runs of
toroid_weak_scaling.py start from: the mesh after -refine uniform refinements (= their -tt_prerefine),
partitioned with ParMETIS on the cell graph (cells sharing a face), before the parallel refinements.
Serial; renders with pyvista (offscreen).

    python toroid_figure.py -refine 2 -nparts 16 -out toroid
    -> toroid_sample.png, toroid_partition_np16.png
"""

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from firedrake.petsc import PETSc
import firedrake as fd
import pymgmc
import twisted_toroid

opts = PETSc.Options()
refine, nparts = opts.getInt("refine", 2), opts.getInt("nparts", 16)
kappa, out = opts.getReal("kappa", 10.0), opts.getString("out", "toroid")
pymgmc.seed(opts.getInt("seed", 1))

coords, tets = twisted_toroid.build(4, opts.getInt("tt_ntheta", 33))
plex = fd.mesh.plex_from_cell_list(3, tets, coords, PETSc.COMM_WORLD)
for _ in range(refine):
    plex.setRefinementUniform(True)
    plex = plex.refine()
mesh = fd.Mesh(plex)

# alpha = 1 sample with the exact Cholesky sampler
V = fd.FunctionSpace(mesh, "CG", 1)
u, v = fd.TrialFunction(V), fd.TestFunction(V)
A = fd.assemble((fd.Constant(kappa ** 2) * u * v + fd.inner(fd.grad(u), fd.grad(v))) * fd.dx, mat_type="aij").M.handle
ksp = PETSc.KSP().create()
ksp.setOperators(A, A)
ksp.setType("richardson")
ksp.setTolerances(max_it=1)
ksp.getPC().setType("cholsampler")
b, x = A.createVecs()
b.zeroEntries()
ksp.solve(b, x)

cells = V.cell_node_map().values  # (ncells, 4) vertex numbers, same numbering as X
X = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 1)).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro
# Firedrake reorders the vertices of each cell; orient all tetrahedra positively (VTK convention), otherwise the
# extracted surface has mixed normals and the shading shows dark patches that look like holes
neg = np.einsum("ij,ij->i", np.cross(X[cells[:, 1]] - X[cells[:, 0]], X[cells[:, 2]] - X[cells[:, 0]]), X[cells[:, 3]] - X[cells[:, 0]]) < 0
cells = cells.copy()
cells[neg] = cells[neg][:, [1, 0, 2, 3]]

# partition of the cell graph (two cells are adjacent if they share a face)
faces = np.sort(np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 3]], cells[:, [1, 2, 3]]]), axis=1)
owner = np.tile(np.arange(len(cells)), 4)
_, inv, cnt = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
order = np.argsort(inv, kind="stable")
pairs = owner[order].reshape(-1)[np.repeat(cnt == 2, cnt)].reshape(-1, 2)  # the two cells of each interior face
G = sp.coo_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                  shape=(len(cells), len(cells))).tocsr()
G.sort_indices()
adj = PETSc.Mat().createAIJ(size=G.shape, csr=(G.indptr.astype(PETSc.IntType), G.indices.astype(PETSc.IntType), G.data))
part = PETSc.MatPartitioning().create()
part.setAdjacency(adj)
part.setType("parmetis")
opts["mat_partitioning_nparts"] = nparts  # read by setFromOptions (no petsc4py setter)
part.setFromOptions()
isp = PETSc.IS()
part.apply(isp)
parts = isp.getIndices().copy()

grid = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, X)
grid.point_data["sample"] = x.array_r.copy()
grid.cell_data["part"] = parts
surf = grid.extract_surface(algorithm="dataset_surface")
# the four corner ridges of the twisted square tube, drawn as lines to show the shape
ridges = surf.extract_feature_edges(feature_angle=40, boundary_edges=False, manifold_edges=False, non_manifold_edges=False)
# 16 well-distinguishable colours (cycled if there are more parts)
palette = (np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
                     [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
                     [152, 223, 138], [255, 152, 150], [197, 176, 213], [219, 219, 141]]))[np.arange(nparts) % 16]
surf.cell_data["part_rgb"] = palette[surf.cell_data["part"]]


def render(name, scalars, **kw):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1100))
    p.set_background("white")
    p.add_mesh(surf, scalars=scalars, ambient=0.25, specular=0.0, **kw)
    p.add_mesh(ridges, color="black", line_width=3)
    p.camera_position = "xy"
    p.camera.elevation = 15  # almost from above: from lower angles the twisted faces fold over at the outline
    p.camera.zoom(1.2)
    img = p.screenshot(None, return_img=True)
    rows, cols = np.where(np.any(img < 250, axis=2))  # crop the white border
    from PIL import Image
    Image.fromarray(img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]).save(name)


c = round(3 * float(np.std(x.array_r)))  # symmetric colour range of about 3 standard deviations
render(f"{out}_sample.png", "sample", cmap="viridis", clim=[-c, c], show_scalar_bar=True, smooth_shading=True,
       scalar_bar_args=dict(title="", vertical=False, color="black", position_x=0.25, width=0.5, label_font_size=28,
                            fmt="%.0f", n_labels=3))
render(f"{out}_partition_np{nparts}.png", "part_rgb", rgb=True)
print(f"{len(cells)} cells, {V.dim()} vertices, {nparts} parts of {np.bincount(parts).min()}-{np.bincount(parts).max()} cells")
