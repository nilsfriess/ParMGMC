"""VesselGraph BALBc_no1 whole-brain vessel graph and its Whittle-Matérn precision.

Nodes are vessel branch points, edges are vessel segments; positions and lengths
are in voxels of the 3um scan.
"""

import csv
import datetime
import io
import os
import socket
import subprocess
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.sparse.csgraph import connected_components

# Data downloaded from https://github.com/jocpae/VesselGraph
NAME = "BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0"
PREFIX = f"BALBc_no1/{NAME}/{NAME}"
NODES_CSV = PREFIX + "_nodes_processed.csv"
EDGES_CSV = PREFIX + "_edges_processed.csv"
ATLAS_CSV = PREFIX + "_atlas_processed.csv"  # one-hot brain region (Allen atlas) of each node

Coords = npt.NDArray[np.float64]


def log(msg: str) -> None:
    """Print on rank 0 and flush immediately (progress output should appear even when redirected)."""
    if PETSc.COMM_WORLD.getRank() == 0:
        print(msg, flush=True)


def git_commit() -> str:
    """Short hash of the ParMGMC checkout this script lives in, with +dirty if tracked files are modified."""
    repo = os.path.dirname(os.path.abspath(__file__))

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True)

    try:
        dirty = run("status", "--porcelain", "--untracked-files=no").stdout.strip()
        return run("rev-parse", "--short", "HEAD").stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_info(script: str) -> dict:
    """Context of this run for the CSV output."""
    options = PETSc.Options().getAll()
    return {
        "script": script,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "parmgmc_commit": git_commit(),
        "petsc_version": ".".join(str(v) for v in PETSc.Sys.getVersion()),
        "ranks": PETSc.COMM_WORLD.getSize(),
        "petsc_options": " ".join(f"-{k}" if v in (None, "") else f"-{k} {v}" for k, v in sorted(options.items())),
    }


def count_nonzero(v: PETSc.Vec) -> int:
    """Number of nonzero entries of a distributed vector."""
    return v.getComm().tompi4py().allreduce(int(np.count_nonzero(v.getArray())))


def print_csv(rows: list[dict]) -> None:
    """Print the rows as CSV (header from the keys of the first row) on rank 0, after a marker line."""
    if not rows or PETSc.COMM_WORLD.getRank() != 0:
        return
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    print("\n# CSV", flush=True)
    print(out.getvalue(), end="", flush=True)


def load_graph(regions: bool = False, weight: str = "length") -> tuple:
    """Return node coordinates and the symmetric edge-length matrix L (L_ij = length of edge ij).

    With ``regions``, also return the brain region index of each node and the list of region names. ``weight`` is
    the column of the edge CSV stored in L instead of the length (e.g. avgRadiusAvg), with the same sparsity pattern.
    """
    nodes = pd.read_csv(NODES_CSV, sep=";", usecols=["id", "pos_x", "pos_y", "pos_z"])
    edges = pd.read_csv(EDGES_CSV, sep=";", usecols=["node1id", "node2id", weight])
    assert np.array_equal(nodes["id"], np.arange(len(nodes)))

    coords = nodes[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    i, j = edges["node1id"].to_numpy(), edges["node2id"].to_numpy()
    length = edges[weight].to_numpy(dtype=np.float64)
    assert np.all(i != j) and np.all(length > 0)

    n = len(coords)
    L = sp.coo_array((length, (i, j)), shape=(n, n)).tocsr()
    L = (L + L.T).tocsr()
    assert L.nnz == 2 * len(length), "duplicate edges"
    if not regions:
        return largest_component(coords, L, np.ones(n, dtype=bool))

    atlas = pd.read_csv(ATLAS_CSV, sep=";", dtype=np.uint8)
    assert len(atlas) == n and np.all(atlas.to_numpy().sum(axis=1) == 1), "expected exactly one region per node"
    names = [c.removeprefix("Region_Acronym_") for c in atlas.columns]
    return *largest_component(coords, L, np.ones(n, dtype=bool), atlas.to_numpy().argmax(axis=1)), names


def largest_component(coords: Coords, L: sp.csr_array, mask: npt.NDArray[np.bool_], *extra: np.ndarray) -> tuple:
    """Restrict to the nodes in ``mask``, then to the largest connected component of that subgraph.

    Further per-node arrays in ``extra`` are restricted in the same way and returned after coords and L.
    """
    coords, L, extra = coords[mask], L[mask][:, mask], [e[mask] for e in extra]
    _, label = connected_components(L, directed=False)
    keep = label == np.argmax(np.bincount(label))
    return coords[keep], L[keep][:, keep].tocsr(), *(e[keep] for e in extra)


def crop(coords: Coords, L: sp.csr_array, fraction: float, *extra: np.ndarray) -> tuple:
    """Largest component inside a box around the centre holding roughly ``fraction`` of the volume."""
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo) / 2 * fraction ** (1 / 3)
    mask = np.all(np.abs(coords - centre) <= half, axis=1)
    return largest_component(coords, L, mask, *extra)


def precision(L: sp.csr_array, kappa: float) -> sp.csr_array:
    """Precision of the alpha=1 Whittle-Matérn field on the metric graph (Bolin, Simas, Wallin).

    Each edge of length l contributes kappa tau^2 / sinh(kappa l) [[cosh(kappa l), -1], [-1, cosh(kappa l)]]
    on its two endpoints. We take tau^2 = 1 / (2 kappa), i.e. unit marginal variance on the real line.
    """
    x = kappa * L.data
    off = L.copy()
    off.data = 1 / (2 * np.sinh(x))
    diag = L.copy()
    diag.data = 1 / (2 * np.tanh(x))
    return (sp.diags_array(diag.sum(axis=1)) - off).tocsr()


def to_petsc(Q: sp.csr_array, comm: PETSc.Comm = PETSc.COMM_WORLD) -> PETSc.Mat:
    """Distribute Q in PETSc's default layout; every rank holds all of Q and passes its own block of rows."""
    n = Q.shape[0]
    size, rank = comm.getSize(), comm.getRank()
    rstart, rend = rank * n // size, (rank + 1) * n // size
    indptr = Q.indptr[rstart : rend + 1]
    csr = (
        (indptr - indptr[0]).astype(PETSc.IntType),
        Q.indices[indptr[0] : indptr[-1]].astype(PETSc.IntType),
        Q.data[indptr[0] : indptr[-1]],
    )
    A = PETSc.Mat().createAIJ(((rend - rstart, n), (rend - rstart, n)), csr=csr, comm=comm)
    A.setOption(PETSc.Mat.Option.SPD, True)
    return A


def mean_weights(L: sp.csr_array, inside: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Weights w such that w^T u is the length-weighted mean of u over the edges with both endpoints ``inside``.

    Discretises (1 / |S|) int_S u ds over that part S of the graph; on each edge u is replaced by the mean of its
    endpoint values.
    """
    E = sp.triu(L, k=1).tocoo()  # each edge once
    keep = inside[E.row] & inside[E.col]
    assert keep.any(), "no edges inside"
    w = np.zeros(L.shape[0])
    np.add.at(w, E.row[keep], E.data[keep] / 2)
    np.add.at(w, E.col[keep], E.data[keep] / 2)
    return w / E.data[keep].sum()


def ball_mean_weights(coords: Coords, L: sp.csr_array, radius: float) -> npt.NDArray[np.float64]:
    """QoI weights: length-weighted mean over the vessels inside a ball of ``radius`` around the centre."""
    centre = (coords.min(axis=0) + coords.max(axis=0)) / 2
    return mean_weights(L, np.linalg.norm(coords - centre, axis=1) <= radius)


def to_petsc_vec(v: npt.NDArray[np.float64], rows: PETSc.IS) -> PETSc.Vec:
    """v (in the original numbering, on every rank) as a vector in the layout of redistribute(..., rows)."""
    comm = rows.getComm()
    n, size, rank = len(v), comm.getSize(), comm.getRank()
    rstart, rend = rank * n // size, (rank + 1) * n // size
    v0 = PETSc.Vec().createWithArray(v[rstart:rend].copy(), size=(rend - rstart, n), comm=comm)
    sub = v0.getSubVector(rows)
    w = sub.copy()
    v0.restoreSubVector(rows, sub)
    v0.destroy()
    return w


def partition(A: PETSc.Mat) -> PETSc.IS:
    """Rows each rank should own for a good parallel layout of A (graph partitioner, ParMETIS by default).

    The partition only depends on the sparsity pattern (symmetric here), so it can be reused for all kappa.
    """
    part = PETSc.MatPartitioning().create(A.getComm())
    part.setAdjacency(A)
    part.setType(PETSc.MatPartitioning.Type.PARTITIONINGPARMETIS)
    part.setFromOptions()  # e.g. -mat_partitioning_type ptscotch
    owner = PETSc.IS()
    part.apply(owner)  # new owning rank of each local row
    part.destroy()
    rows = owner.buildTwoSided()  # rows (in the original numbering) this rank owns after repartitioning
    owner.destroy()
    return rows


def redistribute(A: PETSc.Mat, rows: PETSc.IS) -> PETSc.Mat:
    """A with its rows and columns renumbered so that each rank owns ``rows``."""
    B = A.createSubMatrix(rows, rows)
    B.setOption(PETSc.Mat.Option.SPD, True)
    return B


def cut_fraction(A: PETSc.Mat) -> float:
    """Fraction of the off-diagonal nonzeros of A that couple rows owned by different ranks."""
    local = A.getInfo(PETSc.Mat.InfoType.LOCAL)["nz_used"]
    diag_block = A.getDiagonalBlock().getInfo(PETSc.Mat.InfoType.LOCAL)["nz_used"]
    offproc, total = A.getComm().tompi4py().allreduce(np.array([local - diag_block, local - A.getLocalSize()[0]]))
    return offproc / total


# ---------------------------------------------------------------------------------------------------------------
# Preprocessed binary file (written once by preprocess_graph.py) and its parallel counterparts of the above.
# The file holds the edge-length matrix L followed by the x, y, z coordinates and the atlas region index of each
# node as vectors; the region names are stored in <file>.regions.


@dataclass
class DistributedGraph:
    L: PETSc.Mat  # symmetric edge-length matrix, rows distributed by the graph partitioner
    coords: list[PETSc.Vec]  # x, y, z of the nodes in the same layout
    regions: PETSc.Vec  # atlas region index of each node (index into names)
    names: list[str]
    rows: PETSc.IS  # the owned nodes in the numbering of the file

    @property
    def n(self) -> int:
        return self.L.getSize()[0]

    def info(self) -> dict:
        """Size and partition of the graph for the CSV output."""
        nnz = self.L.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"]
        return {"n": self.n, "edges": int(nnz) // 2, "edges_cut": cut_fraction(self.L)}


def write(filename: str, coords: Coords, L: sp.csr_array, regions: np.ndarray, names: list[str]) -> None:
    """Write the graph (serially) in the format read by ``load``."""
    comm = PETSc.COMM_SELF
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    csr = (L.indptr.astype(PETSc.IntType), L.indices.astype(PETSc.IntType), L.data)
    PETSc.Mat().createAIJ(L.shape, csr=csr, comm=comm).view(viewer)
    for values in (*coords.T, regions.astype(np.float64)):
        PETSc.Vec().createWithArray(np.ascontiguousarray(values), comm=comm).view(viewer)
    viewer.destroy()
    with open(filename + ".regions", "w") as f:
        f.write("\n".join(names) + "\n")


def load(filename: str, comm: PETSc.Comm = PETSc.COMM_WORLD) -> DistributedGraph:
    """Read the graph in parallel and redistribute it with the graph partitioner (see ``partition``)."""
    viewer = PETSc.Viewer().createBinary(filename, "r", comm=comm)
    L0 = PETSc.Mat().create(comm)
    L0.setType(PETSc.Mat.Type.AIJ)
    L0.load(viewer)
    vecs0 = []
    for _ in range(4):
        v = PETSc.Vec().create(comm)
        v.load(viewer)
        vecs0.append(v)
    viewer.destroy()

    rows = partition(L0)
    L = L0.createSubMatrix(rows, rows)
    vecs = []
    for v0 in vecs0:
        sub = v0.getSubVector(rows)
        vecs.append(sub.copy())
        v0.restoreSubVector(rows, sub)
    with open(filename + ".regions") as f:
        names = f.read().split("\n")[:-1]
    return DistributedGraph(L, vecs[:3], vecs[3], names, rows)


def precision_petsc(L: PETSc.Mat, kappa: float) -> PETSc.Mat:
    """Parallel version of ``precision``: each rank transforms its own rows of the distributed L."""
    indptr, cols, lengths = L.getValuesCSR()
    rstart, rend = L.getOwnershipRange()
    nloc, n = rend - rstart, L.getSize()[0]
    x = kappa * lengths
    local_rows = np.repeat(np.arange(nloc), np.diff(indptr))
    diag = np.bincount(local_rows, weights=1 / (2 * np.tanh(x)), minlength=nloc)
    Q = sp.csr_array((-1 / (2 * np.sinh(x)), cols, indptr), shape=(nloc, n))
    Q = (Q + sp.csr_array((diag, (np.arange(nloc), rstart + np.arange(nloc))), shape=(nloc, n))).tocsr()
    Q.sort_indices()
    csr = (Q.indptr.astype(PETSc.IntType), Q.indices.astype(PETSc.IntType), Q.data)
    A = PETSc.Mat().createAIJ(((nloc, n), (nloc, n)), csr=csr, comm=L.getComm())
    A.setOption(PETSc.Mat.Option.SPD, True)
    return A


def mean_weights_petsc(L: PETSc.Mat, inside: PETSc.Vec) -> PETSc.Vec | None:
    """Parallel version of ``mean_weights`` for the node set given by the 0/1 vector ``inside``.

    Up to normalisation, w_i = inside_i * sum_j l_ij inside_j = inside * (L inside). None if no edge lies inside.
    """
    w = inside.duplicate()
    L.mult(inside, w)
    w.pointwiseMult(w, inside)
    total = w.sum()
    if total == 0:
        w.destroy()
        return None
    w.scale(1 / total)
    return w


def ball_mean_weights_petsc(G: DistributedGraph, radius: float) -> PETSc.Vec:
    """Parallel version of ``ball_mean_weights``."""
    centre = [(c.min()[1] + c.max()[1]) / 2 for c in G.coords]
    dist = np.sqrt(sum((c.getArray() - m) ** 2 for c, m in zip(G.coords, centre, strict=True)))
    inside = G.coords[0].duplicate()
    inside.getArray()[:] = dist <= radius
    w = mean_weights_petsc(G.L, inside)
    assert w is not None, "no edges inside the QoI ball"
    return w


if __name__ == "__main__":
    coords, L = load_graph()
    PETSc.Sys.Print(f"nodes {L.shape[0]}, edges {L.nnz // 2}")
    A = to_petsc(precision(L, 0.01))
    rows = partition(A)
    B = redistribute(A, rows)
    size = A.getComm().getSize()
    PETSc.Sys.Print(f"{size} ranks, edges cut: {cut_fraction(A):.2%} in file order, {cut_fraction(B):.2%} partitioned")
