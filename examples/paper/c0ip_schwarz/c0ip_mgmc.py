"""Two-level MGMC for a C0 interior penalty (P2) discretisation of (kappa^2 - Delta)^2 on the unit
square with natural BCs (alpha = 2).

Fine level (-fine):
  schwarzgibbs  PCSCHWARZGIBBS: parallel multiplicative overlapping block Gibbs over algebraic
                subdomains (graph partitioning of the local rows into -nblocks parts per rank,
                -overlap layers), coloured so that the sweep is exact across ranks
  patchgibbs    PCPATCHGIBBS (serial only): the same sweep over given patches. Without -patches
                the patches are the M x M boxes of the coarse space (each extended by delta on every
                side, lexicographic order); with -patches file.npz, the subdomains exported by an
                -mode exact run, ordered by colour.
Coarse level: Galerkin projection onto Z = {chi_j p}, chi_j a smooth (C2, quintic smoothstep)
partition of unity over M x M boxes and p in {1, x, y, x^2, xy, y^2} (centred and scaled per box);
sampled exactly with PCCHOLSAMPLER. All columns of Z belong to rank 0. Both levels run inside
PCGAMGMC with a user-configured PCMG (2 levels, interpolation Z).

a(u,v) = sum_T (D2u, D2v) + 2k^2 (grad u, grad v) + k^4 (u, v)
         - <{d2u/dn2}, [dv/dn]> - <{d2v/dn2}, [du/dn]> + sigma/h <[du/dn], [dv/dn]>   (interior edges)

Options
  -N cells per side, -M coarse boxes per side, -delta PoU overlap (default H/4), -kappa, -sigma (20),
  -ns affine|quad (default quad), -fine schwarzgibbs|patchgibbs, -nblocks, -overlap, -seed
  -mode rho       deterministic smoother and coarse solve: convergence factor of the two-level cycle
                  (power iteration in the energy norm, -rho_its steps)
  -mode exact     one deterministic cycle from a start vector given by the coordinates; writes the
                  result and the subdomains with their colours to -out file.npz. With
                  -fine patchgibbs -patches file.npz (serial): the same cycle with those subdomains as
                  patches in colour order, compared with the stored result.
  -mode sample    -nsamples samples after -nburnin; QoI and pointwise variances vs. exact (computed on
                  rank 0); in serial also an exact Cholesky sampler of the same length as reference.
  -coarse_noise 0 negative control: deterministic coarse step (the chain is then biased)
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
from mpi4py import MPI
from firedrake.petsc import PETSc
import firedrake as fd
import pymgmc

opts = PETSc.Options()
N = opts.getInt("N", 32)
M = opts.getInt("M", 4)
delta = opts.getReal("delta", 1.0 / (4 * M))
kappa = opts.getReal("kappa", 1.0)
sigma = opts.getReal("sigma", 20.0)
ns = opts.getString("ns", "quad")
fine = opts.getString("fine", "schwarzgibbs")
nblocks = opts.getInt("nblocks", 1)
overlap = opts.getInt("overlap", 2)
mode = opts.getString("mode", "rho")
nsamples = opts.getInt("nsamples", 20000)
nburnin = opts.getInt("nburnin", 100)
rho_its = opts.getInt("rho_its", 50)
coarse_noise = opts.getBool("coarse_noise", True)
seed = opts.getInt("seed", 1)
outfile = opts.getString("out", "exact.npz")
patchfile = opts.getString("patches", "")
pymgmc.seed(seed)
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
assert fine in ("schwarzgibbs", "patchgibbs")
assert fine == "schwarzgibbs" or size == 1, "PCPATCHGIBBS is sequential"

# --- discretisation ---------------------------------------------------------------------------
mesh = fd.UnitSquareMesh(N, N)
V = fd.FunctionSpace(mesh, "CG", 2)
u, v = fd.TrialFunction(V), fd.TestFunction(V)
n = fd.FacetNormal(mesh)
h = fd.avg(fd.CellDiameter(mesh))
ks = fd.Constant(kappa ** 2)


def d2n(w):
    H = fd.grad(fd.grad(w))
    return 0.5 * (fd.dot(fd.dot(H("+"), n("+")), n("+")) + fd.dot(fd.dot(H("-"), n("+")), n("+")))


a = ((fd.inner(fd.grad(fd.grad(u)), fd.grad(fd.grad(v))) + 2 * ks * fd.inner(fd.grad(u), fd.grad(v)) + ks ** 2 * u * v) * fd.dx
     - (d2n(u) * fd.jump(fd.grad(v), n) + d2n(v) * fd.jump(fd.grad(u), n)) * fd.dS
     + fd.Constant(sigma) / h * fd.jump(fd.grad(u), n) * fd.jump(fd.grad(v), n) * fd.dS)
A = fd.assemble(a, mat_type="aij").M.handle
rstart, rend = A.getOwnershipRange()
nown = rend - rstart
X = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro[:nown].copy()
keys = np.round(X * 2 * N).astype(np.int64)  # P2 nodes lie on the 1/(2N) grid: exact, partition-independent keys
ndof = A.getSize()[0]


def key_id(k):
    return k[:, 0] * (2 * N + 1) + k[:, 1]


def coord_noise(k):
    """Pseudo-random start vector as a function of the coordinates (same on any number of ranks)."""
    t = np.sin(key_id(k).astype(float) * 12.9898 + 78.233) * 43758.5453
    return t - np.floor(t) - 0.5


# --- coarse space: smooth PoU over M x M boxes times polynomials --------------------------------


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


def bump1d(x, i):
    """1D PoU factor of interval i: transitions over [i/M - delta, i/M + delta] (none at 0 and 1)."""
    f = np.ones_like(x)
    if i > 0:
        f *= smoothstep((x - (i / M - delta)) / (2 * delta))
    if i < M - 1:
        f *= smoothstep(((i + 1) / M + delta - x) / (2 * delta))
    return f


eps = 1e-12
boxes, cols = [], []
for i in range(M):
    for j in range(M):
        lo = np.array([i / M - delta, j / M - delta]) - eps
        hi = np.array([(i + 1) / M + delta, (j + 1) / M + delta]) + eps
        boxes.append(np.where(np.all((X >= lo) & (X <= hi), 1))[0])
        chi = bump1d(X[:, 0], i) * bump1d(X[:, 1], j)
        xr, yr = (X[:, 0] - (i + 0.5) / M) * M, (X[:, 1] - (j + 0.5) / M) * M
        polys = [np.ones(nown), xr, yr] + ([xr * xr, xr * yr, yr * yr] if ns == "quad" else [])
        cols += [sp.csr_matrix(chi * p).T for p in polys]
Zloc = sp.hstack(cols).tocsr()
Zloc.eliminate_zeros()
nc = Zloc.shape[1]
Z = PETSc.Mat().createAIJ(size=((nown, ndof), (nc if rank == 0 else 0, nc)),
                          csr=(Zloc.indptr.astype(PETSc.IntType), Zloc.indices.astype(PETSc.IntType), Zloc.data),
                          comm=PETSc.COMM_WORLD)

# --- patches for the serial PCPATCHGIBBS path ----------------------------------------------------
if fine == "patchgibbs":
    if patchfile:
        f = np.load(patchfile, allow_pickle=True)
        pos = {kid: i for i, kid in enumerate(key_id(keys))}
        # colour order (exact equivalent of the parallel sweep); -ignore_colors keeps the exported order
        order = np.arange(len(f["colors"])) if opts.getBool("ignore_colors", False) else np.argsort(f["colors"], kind="stable")
        patches = [np.sort(np.array([pos[kid] for kid in f["subdomains"][p]])) for p in order]
    else:
        patches = boxes
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in patches])]).astype(PETSc.IntType)
    indices = np.concatenate(patches).astype(PETSc.IntType)
    cover = np.zeros(nown)
    for p in patches:
        cover[p] += 1
    assert cover.min() >= 1, "every DOF must lie in a patch"


# --- two-level MGMC inside PCGAMGMC --------------------------------------------------------------
def make_ksp(deterministic, coarse_sample=True):
    for k in [k for k in opts.getAll() if k.startswith("gamgmc_") or k.startswith("pc_gamgmc")]:
        opts.delValue(k)
    opts["pc_gamgmc_mg_type"] = "mg"
    opts["gamgmc_mg_levels_pc_type"] = fine  # otherwise PCSetUp_GAMGMC injects sorgibbs
    if fine == "schwarzgibbs":
        opts["gamgmc_mg_levels_pc_schwarzgibbs_local_blocks"] = nblocks
        opts["gamgmc_mg_levels_pc_schwarzgibbs_overlap"] = overlap
    if deterministic:
        opts[f"gamgmc_mg_levels_pc_{fine}_deterministic"] = "1"
    if deterministic or not coarse_sample:
        opts["gamgmc_mg_coarse_ksp_type"] = "preonly"
        opts["gamgmc_mg_coarse_pc_type"] = "cholesky"
        opts["gamgmc_mg_coarse_pc_factor_mat_solver_type"] = "mumps"
    ksp = PETSc.KSP().create()
    ksp.setOperators(A, A)
    ksp.setType("richardson")
    ksp.setTolerances(max_it=1)
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    ksp.setInitialGuessNonzero(True)
    pc = ksp.getPC()
    pc.setType("gamgmc")
    mg = pymgmc.PCGAMGMCGetInternalPC(pc)
    mg.setOptionsPrefix("gamgmc_")  # same prefix PCSetUp_GAMGMC gives it, set before the levels exist
    mg.setType("mg")
    mg.setMGLevels(2)
    mg.setMGInterpolation(1, Z)
    sm = mg.getMGSmoother(1)
    sm.setType("richardson")
    sm.getPC().setType(fine)
    if fine == "patchgibbs":
        pymgmc.PCPatchGibbsSetPatches(sm.getPC(), offsets, indices)
    ksp.setFromOptions()
    comm.Barrier()
    t0 = time.perf_counter()
    ksp.setUp()
    comm.Barrier()
    make_ksp.t_setup = time.perf_counter() - t0
    assert mg.getType() == "mg" and mg.getMGLevels() == 2 and mg.getMGSmoother(1).getPC().getType() == fine
    return ksp, mg


def energy(x):
    y = A.createVecLeft()
    A.mult(x, y)
    return np.sqrt(x.dot(y))


def gather_owned(values):
    """Gather an owned-DOF array to rank 0 in global order."""
    parts = comm.gather((rstart, np.asarray(values)), root=0)
    if rank:
        return None
    out = np.zeros((ndof,) + np.asarray(values).shape[1:])
    for r0, vals in parts:
        out[r0:r0 + len(vals)] = vals
    return out


tag = (f"N={N} M={M} k={kappa:g} ns={ns} ndof={ndof} ncoarse={nc} fine={fine}"
       + (f" nblocks={nblocks} overlap={overlap}" if fine == "schwarzgibbs" else "") + f" np={size}")


def subdomain_info(mg):
    """PCView-like summary of the PCSCHWARZGIBBS subdomains (sizes, number of colours)."""
    subs, colors = pymgmc.PCSchwarzGibbsGetSubdomains(mg.getMGSmoother(1).getPC())
    colors = np.array(colors, dtype=int)
    sizes = comm.gather([len(s) for s in subs], root=0)
    ncol = comm.allreduce(int(colors.max()) + 1 if len(colors) else 0, op=MPI.MAX)
    if rank:
        return ""
    sizes = [s for part in sizes for s in part]
    return f" | {len(sizes)} subdomains, sizes {min(sizes)}-{max(sizes)}, {ncol} colours"


if mode == "rho":
    ksp, mg = make_ksp(deterministic=True)
    info = subdomain_info(mg) if fine == "schwarzgibbs" else ""
    b, x = A.createVecs()
    b.zeroEntries()
    x.array[:] = coord_noise(keys)
    x.scale(1.0 / energy(x))
    ratios = []
    for it in range(rho_its):
        ksp.solve(b, x)  # one cycle: x <- x + B(b - A x)
        e = energy(x)
        ratios.append(e)
        x.scale(1.0 / e)
    last = np.asarray(ratios[-10:])
    rho = float(np.exp(np.mean(np.log(last))))
    if rank == 0:
        print(f"RESULT rho {tag} | rho {rho:.4f} ({last.min():.4f}-{last.max():.4f}){info}", flush=True)

elif mode == "exact":
    ksp, mg = make_ksp(deterministic=True)
    b, x = A.createVecs()
    b.zeroEntries()
    x.array[:] = coord_noise(keys)
    ksp.solve(b, x)
    allkeys = gather_owned(key_id(keys))
    x1 = gather_owned(x.array_r)
    if fine == "schwarzgibbs":
        subs, colors = pymgmc.PCSchwarzGibbsGetSubdomains(mg.getMGSmoother(1).getPC())
        subs = [np.array(s_, dtype=np.int64) for s_ in subs]
        colors = np.array(colors, dtype=int)
        subs = comm.gather(subs, root=0)
        colors = comm.gather(colors, root=0)
        if rank == 0:
            gkey = allkeys.astype(np.int64)
            subs = [gkey[s] for part in subs for s in part]
            colors = np.concatenate(colors)
            np.savez(outfile, keys=gkey, x1=x1, subdomains=np.array(subs, dtype=object), colors=colors)
            print(f"RESULT exact {tag} | wrote {outfile}: {len(subs)} subdomains, {colors.max() + 1} colours, "
                  f"|x1| = {np.abs(x1).max():.3e}", flush=True)
    else:
        ref = np.load(patchfile, allow_pickle=True) if patchfile else None
        if ref is not None:
            pos = {kid: i for i, kid in enumerate(ref["keys"])}
            refx = np.array([ref["x1"][pos[k]] for k in allkeys.astype(np.int64)])
            err = np.abs(x1 - refx).max() / np.abs(refx).max()
            how = "exported order (colours ignored)" if opts.getBool("ignore_colors", False) else "colour order"
            print(f"RESULT exact {tag} | {len(patches)} patches in {how} | max rel diff to {patchfile}: {err:.2e}", flush=True)

else:
    def iact(q):
        """Integrated autocorrelation time with Sokal's automatic window (c = 5)."""
        q = q - q.mean()
        nq = len(q)
        f = np.fft.rfft(q, 2 * nq)
        acf = np.fft.irfft(f * np.conj(f))[:nq] / (q @ q)
        tau = 1.0
        for w in range(1, nq):
            tau = 1 + 2 * acf[1:w + 1].sum()
            if w >= 5 * tau:
                break
        return tau

    # QoIs: mean over [0.25, 0.5]^2, mean over the whole square, point value at the centre
    x_, y_ = fd.SpatialCoordinate(mesh)
    ind = fd.conditional(fd.And(fd.And(x_ > 0.25, x_ < 0.5), fd.And(y_ > 0.25, y_ < 0.5)), 16.0, 0.0)
    gs = [fd.assemble(ind * v * fd.dx).dat.data_ro[:nown].copy(), fd.assemble(v * fd.dx).dat.data_ro[:nown].copy(),
          (np.all(keys == N, 1)).astype(float)]
    G = np.stack(gs, 1)
    qnames = ["box mean", "domain mean", "centre value"]

    # exact references on rank 0 (gathered matrix)
    ip, ix, dv = A.getValuesCSR()
    rows = comm.gather((rstart, ip, ix, dv), root=0)
    Gall = gather_owned(G)
    if rank == 0 and not opts.getBool("timing_only", False):
        indptr, idx, val = [0], [], []
        for r0, ipr, ixr, dvr in sorted(rows, key=lambda t: t[0]):
            indptr += list(indptr[-1] + ipr[1:])
            idx.append(ixr)
            val.append(dvr)
        Asp = sp.csr_matrix((np.concatenate(val), np.concatenate(idx), np.array(indptr)), shape=(ndof, ndof))
        lu = sp.linalg.splu(Asp.tocsc())
        var_q = np.array([g @ lu.solve(g) for g in Gall.T])
        dvar = np.diag(sl.inv(Asp.toarray())) if ndof <= 6000 else None
    else:
        var_q = dvar = None

    def run_chain(ksp_):
        qs = np.zeros((nsamples, G.shape[1]))
        s1, s2 = np.zeros(nown), np.zeros(nown)
        b, x = A.createVecs()
        b.zeroEntries()
        x.zeroEntries()
        for it in range(nburnin + nsamples):
            if it == nburnin:
                comm.Barrier()
                t0 = time.perf_counter()
            ksp_.solve(b, x)
            if it >= nburnin:
                xa = x.array_r
                qs[it - nburnin] = comm.allreduce(xa @ G, op=MPI.SUM)
                s1 += xa
                s2 += xa * xa
        comm.Barrier()
        run_chain.t_sample = (time.perf_counter() - t0) / nsamples
        mean = s1 / nsamples
        return qs, gather_owned(s2 / nsamples - mean ** 2)

    def report(name, qs, var):
        if rank:
            return
        out = [f"RESULT {name:14s} {tag} | setup {report.t_setup:.2f} s, {1e3 * run_chain.t_sample:.2f} ms/sample (incl. QoI bookkeeping)"]
        for k in range(G.shape[1] if var_q is not None else 0):
            tau = iact(qs[:, k])
            r = qs[:, k].var() / var_q[k]
            err = np.sqrt(2 * tau / nsamples)
            zm = qs[:, k].mean() / np.sqrt(var_q[k] * tau / nsamples)
            out.append(f"  {qnames[k]:12s}: var ratio {r:.3f} +- {err:.3f}  mean z {zm:+.2f}  IACT {tau:.2f}")
        if var_q is not None and dvar is not None:
            rat = var / dvar
            out.append(f"  pointwise var ratio: mean {rat.mean():.4f}  rms(ratio - 1) {np.sqrt(((rat - 1) ** 2).mean()):.4f}"
                       f"  min {rat.min():.3f}  max {rat.max():.3f}")
        print("\n".join(out), flush=True)

    if size == 1:  # exact reference sampler (independent samples) of the same length
        kc = PETSc.KSP().create()
        kc.setOperators(A, A)
        kc.setType("richardson")
        kc.setTolerances(max_it=1)
        kc.setNormType(PETSc.KSP.NormType.NONE)
        kc.setInitialGuessNonzero(True)
        kc.getPC().setType("cholsampler")
        t0 = time.perf_counter()
        kc.setUp()
        report.t_setup = time.perf_counter() - t0
        report("cholesky", *run_chain(kc))

    ksp, mg = make_ksp(deterministic=False, coarse_sample=coarse_noise)
    info = subdomain_info(mg) if fine == "schwarzgibbs" else ""
    if rank == 0:
        print(f"setup{info}", flush=True)
    report.t_setup = make_ksp.t_setup
    report("mgmc" if coarse_noise else "mgmc NO-NOISE", *run_chain(ksp))
