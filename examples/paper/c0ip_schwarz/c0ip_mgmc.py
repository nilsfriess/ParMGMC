"""alpha = 2 on the unit square / cube: two-level MGMC (PCSCHWARZMGMC) for a C0 interior penalty (P2)
discretisation of (kappa^2 - Delta)^2 with natural boundary conditions, and the SPDE sampler.

    a(u,v) = sum_T (D2u, D2v) + 2k^2 (grad u, grad v) + k^4 (u, v)
             - <{d2u/dn2}, [dv/dn]> - <{d2v/dn2}, [du/dn]> + sigma/h <[du/dn], [dv/dn]>   (interior facets)

Subdomains: the M^d boxes of side H = 1/M, each extended by delta (default H/4) in every direction.
Each box is owned by the rank that owns most of its core. Coarse space: Z = {chi_b p}, chi_b a smooth
(C2, quintic smoothstep) partition of unity over the same boxes, p in {1, x, .., x^2, xy, ..} in
coordinates centred and scaled per box, orthonormalised per box. The coarse operator is Z^T A Z.

Modes (-mode):
  rho     deterministic cycle (-pc_schwarzmgmc_deterministic): convergence factor by power iteration in
          the energy norm (-rho_its steps). -one_level: the Schwarz sweep alone, without coarse space.
  sample  -nsamples MGMC samples after -nburnin; timing and integrated autocorrelation times (IACT)
          of three QoIs. Without -timing_only also QoI variances against the exact g^T A^{-1} g and,
          for small problems, pointwise variances against diag(A^{-1}) (exact references on rank 0).
          -cholesky: also an exact Cholesky sampler of the same length (independent samples).
  spde    standard SPDE sampler for comparison: P1, lumped mass M_L, L = K + kappa^2 M,
          x = L^{-1} M_L^{1/2} z with one CG + GAMG solve (relative tolerance -spde_rtol). P1 with 2N
          cells per side has as many unknowns as P2 with N. Without -timing_only also the variance
          checks against the exact covariance L^{-1} M_L L^{-1}.

Options: -dim 2|3, -N cells per side, -M boxes per side, -delta, -kappa (1), -sigma (20), -seed,
         -pc_schwarzmgmc_up_sweeps 0|1 (V(1,0) default), -gamgmc_mg_levels_... (fine sampler).
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpi4py import MPI
from firedrake.petsc import PETSc
import firedrake as fd
import pymgmc

opts = PETSc.Options()
dim = opts.getInt("dim", 2)
N = opts.getInt("N", 32)
M = opts.getInt("M", 4)
delta = opts.getReal("delta", 1.0 / (4 * M))
kappa = opts.getReal("kappa", 1.0)
sigma = opts.getReal("sigma", 20.0)
mode = opts.getString("mode", "rho")
nsamples = opts.getInt("nsamples", 1000)
nburnin = opts.getInt("nburnin", 10)
timing_only = opts.getBool("timing_only", False)
seed = opts.getInt("seed", 1)
pymgmc.seed(seed)
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

mesh = fd.UnitSquareMesh(N, N) if dim == 2 else fd.UnitCubeMesh(N, N, N)
P = 1 if mode == "spde" else 2
V = fd.FunctionSpace(mesh, "CG", P)
u, v = fd.TrialFunction(V), fd.TestFunction(V)
k2 = fd.Constant(kappa ** 2)

if mode == "spde":
    A = fd.assemble((k2 * u * v + fd.inner(fd.grad(u), fd.grad(v))) * fd.dx, mat_type="aij").M.handle  # L
    with fd.assemble(v * fd.dx).dat.vec_ro as mv:
        ml = mv.copy()  # lumped mass (row sums of M)
else:
    n = fd.FacetNormal(mesh)
    h = fd.avg(fd.CellDiameter(mesh))

    def d2n(w):
        H = fd.grad(fd.grad(w))
        return 0.5 * (fd.dot(fd.dot(H("+"), n("+")), n("+")) + fd.dot(fd.dot(H("-"), n("+")), n("+")))

    a = ((fd.inner(fd.grad(fd.grad(u)), fd.grad(fd.grad(v))) + 2 * k2 * fd.inner(fd.grad(u), fd.grad(v)) + k2 ** 2 * u * v) * fd.dx
         - (d2n(u) * fd.jump(fd.grad(v), n) + d2n(v) * fd.jump(fd.grad(u), n)) * fd.dS
         + fd.Constant(sigma) / h * fd.jump(fd.grad(u), n) * fd.jump(fd.grad(v), n) * fd.dS)
    A = fd.assemble(a, mat_type="aij").M.handle
rstart, rend = A.getOwnershipRange()
nown = rend - rstart
ndof = A.getSize()[0]
X = fd.Function(fd.VectorFunctionSpace(mesh, "CG", P)).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro[:nown].copy()
keys = np.round(X * P * N).astype(np.int64)  # nodes lie on the 1/(P N) grid: exact, partition-independent


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


def polys(xr):
    """{1, x, .., x^2, xy, ..} in box coordinates xr (n x d)."""
    ps = [np.ones(len(xr))] + [xr[:, a_] for a_ in range(dim)]
    return ps + [xr[:, a_] * xr[:, b_] for a_ in range(dim) for b_ in range(a_, dim)]


def coarse_space():
    """Z (owned rows x coarse functions, all columns on rank 0): chi_b p for the at most 3^d boxes
    around each DOF, orthonormalised per box (Gram matrices summed over all ranks)."""
    npoly = len(polys(np.zeros((1, dim))))
    strides = M ** np.arange(dim - 1, -1, -1)
    ci = np.minimum((X * M).astype(int), M - 1)
    rows, cols, vals = [], [], []
    for off in np.ndindex(*(3,) * dim):
        idx = ci + np.array(off) - 1
        r = np.where(np.all((idx >= 0) & (idx < M), 1))[0]
        idx = idx[r]
        chi = np.ones(len(r))
        for ax in range(dim):
            k, xk = idx[:, ax], X[r, ax]
            chi *= np.where(k > 0, smoothstep((xk - (k / M - delta)) / (2 * delta)), 1.0)
            chi *= np.where(k < M - 1, smoothstep(((k + 1) / M + delta - xk) / (2 * delta)), 1.0)
        nz = chi != 0
        r, idx, chi = r[nz], idx[nz], chi[nz]
        for q, pq in enumerate(polys((X[r] - (idx + 0.5) / M) * M)):
            rows.append(r)
            cols.append(idx @ strides * npoly + q)
            vals.append(chi * pq)
    nbox = M ** dim
    Zc = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nown, nbox * npoly))
    G = (Zc.T @ Zc).tocoo()
    same = (G.row // npoly) == (G.col // npoly)
    Gb = np.zeros((nbox, npoly, npoly))
    np.add.at(Gb, (G.row[same] // npoly, G.row[same] % npoly, G.col[same] % npoly), G.data[same])
    Gb = comm.allreduce(Gb, op=MPI.SUM)
    blocks = []
    for gb in Gb:
        lam, W = np.linalg.eigh(gb)
        keep = lam > 1e-10 * lam.max()
        blocks.append(sp.csr_matrix(W[:, keep] / np.sqrt(lam[keep])))
    Zl = (Zc @ sp.block_diag(blocks, format="csr")).tocsr()
    nc = Zl.shape[1]
    return PETSc.Mat().createAIJ(size=((nown, ndof), (nc if rank == 0 else 0, nc)),
                                 csr=(Zl.indptr.astype(PETSc.IntType), Zl.indices.astype(PETSc.IntType), Zl.data))


def box_subdomains():
    """The boxes owned by this rank, extended by delta, as arrays of global DOF indices."""
    nbox = M ** dim
    strides = M ** np.arange(dim - 1, -1, -1)
    core = np.minimum((X * M).astype(int), M - 1) @ strides
    owner = np.argmax(np.array(comm.allgather(np.bincount(core, minlength=nbox))), axis=0)
    eps = 1e-12
    lo = np.clip(np.floor((X - delta - eps) * M).astype(int), 0, M - 1)
    hi = np.clip(np.floor((X + delta + eps) * M).astype(int), 0, M - 1)
    pb, pd = [], []
    for off in np.ndindex(*(2,) * dim):  # delta < H/2: at most two boxes per direction
        idx = lo + np.array(off)
        ok = np.all(idx <= hi, 1)
        for ax in range(dim):
            ok &= (X[:, ax] >= idx[:, ax] / M - delta - eps) & (X[:, ax] <= (idx[:, ax] + 1) / M + delta + eps)
        r = np.where(ok)[0]
        pb.append(idx[r] @ strides)
        pd.append(r + rstart)
    pb, pd = np.concatenate(pb), np.concatenate(pd)
    recv = comm.alltoall([(pb[owner[pb] == q], pd[owner[pb] == q]) for q in range(size)])
    b_all = np.concatenate([x_[0] for x_ in recv])
    d_all = np.concatenate([x_[1] for x_ in recv])
    order = np.argsort(b_all, kind="stable")
    b_all, d_all = b_all[order], d_all[order]
    mine = np.where(owner == rank)[0]
    starts, ends = np.searchsorted(b_all, mine), np.searchsorted(b_all, mine, side="right")
    return [np.sort(d_all[s_:e_]).astype(PETSc.IntType) for s_, e_ in zip(starts, ends) if e_ > s_]


def make_ksp(pctype):
    """KSP (richardson, one iteration = one sample) with the given sampler PC."""
    ksp = PETSc.KSP().create()
    ksp.setOperators(A, A)
    ksp.setType("richardson")
    ksp.setTolerances(max_it=1)
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    ksp.setInitialGuessNonzero(True)
    ksp.getPC().setType(pctype)
    return ksp


def subdomain_info(pc):
    subs, colors = pymgmc.PCSchwarzGibbsGetSubdomains(pc)
    sizes = [s_ for part in comm.gather([len(s_) for s_ in subs], root=0) or [] for s_ in part]
    ncol = comm.allreduce(max(colors) + 1 if colors else 0, op=MPI.MAX)
    return f"{len(sizes)} subdomains, sizes {min(sizes)}-{max(sizes)}, {ncol} colours" if rank == 0 else ""


def gather_owned(values):
    parts = comm.gather((rstart, np.asarray(values)), root=0)
    if rank:
        return None
    out = np.zeros((ndof,) + np.asarray(values).shape[1:])
    for r0, vals in parts:
        out[r0:r0 + len(vals)] = vals
    return out


def iact(q):
    """Integrated autocorrelation time, Sokal's automatic window (c = 5), and its standard error."""
    q = q - q.mean()
    nq = len(q)
    f = np.fft.rfft(q, 2 * nq)
    acf = np.fft.irfft(f * np.conj(f))[:nq] / (q @ q)
    tau, w = 1.0, 1
    for w in range(1, nq):
        tau = 1 + 2 * acf[1:w + 1].sum()
        if w >= 5 * tau:
            break
    return tau, tau * np.sqrt(2 * (2 * w + 1) / nq)


tag = f"dim={dim} N={N} " + ("" if mode == "spde" else f"M={M} ") + f"k={kappa:g} ndof={ndof} np={size}"

# --- sampler set-up ---------------------------------------------------------------------------
if mode != "spde":
    comm.Barrier()
    t0 = time.perf_counter()
    Z = coarse_space()
    subs = box_subdomains()
    comm.Barrier()
    t_coarse = time.perf_counter() - t0
    if opts.getBool("one_level", False):  # the Schwarz sweep alone (rho only)
        assert mode == "rho"
        opts["pc_schwarzgibbs_deterministic"] = 1
        ksp = make_ksp("schwarzgibbs")
        pymgmc.PCSchwarzGibbsSetSubdomains(ksp.getPC(), subs)
    else:
        if mode == "rho":
            opts["pc_schwarzmgmc_deterministic"] = 1
        ksp = make_ksp("schwarzmgmc")
        pymgmc.PCSchwarzMGMCSetCoarseSpace(ksp.getPC(), Z)
        pymgmc.PCSchwarzMGMCSetSubdomains(ksp.getPC(), subs)
    ksp.setFromOptions()
    comm.Barrier()
    t0 = time.perf_counter()
    ksp.setUp()
    comm.Barrier()
    t_setup = time.perf_counter() - t0
    fine_pc = ksp.getPC() if opts.getBool("one_level", False) else pymgmc.PCSchwarzMGMCGetMG(ksp.getPC()).getMGSmoother(1).getPC()
    info = subdomain_info(fine_pc) + f", coarse dim {Z.getSize()[1]}"
else:
    sqrt_ml = ml.copy()
    sqrt_ml.sqrtabs()
    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix("spde_")
    ksp.setOperators(A, A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=opts.getReal("spde_rtol", 1e-8), atol=0.0, max_it=5000)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setErrorIfNotConverged(True)
    ksp.setFromOptions()
    comm.Barrier()
    t0 = time.perf_counter()
    ksp.setUp()
    comm.Barrier()
    t_coarse, t_setup = 0.0, time.perf_counter() - t0
    info = f"rtol {opts.getReal('spde_rtol', 1e-8):g}"

# --- rho: power iteration in the energy norm --------------------------------------------------
if mode == "rho":
    def energy(x):
        y = A.createVecLeft()
        A.mult(x, y)
        return np.sqrt(x.dot(y))

    b, x = A.createVecs()
    b.zeroEntries()
    t = np.sin((keys @ ((P * N + 1) ** np.arange(dim - 1, -1, -1))).astype(float) * 12.9898 + 78.233) * 43758.5453
    x.array[:] = t - np.floor(t) - 0.5  # pseudo-random start vector, independent of the partition
    x.scale(1.0 / energy(x))
    ratios = []
    for it in range(opts.getInt("rho_its", 50)):
        ksp.solve(b, x)  # one cycle: x <- x + B(b - A x)
        e = energy(x)
        ratios.append(e)
        x.scale(1.0 / e)
    last = np.asarray(ratios[-10:])
    if rank == 0:
        print(f"RESULT rho{' one-level' if opts.getBool('one_level', False) else ''} {tag} | rho {np.exp(np.mean(np.log(last))):.4f} "
              f"({last.min():.4f}-{last.max():.4f}) | {info}", flush=True)
        if opts.getBool("rho_print", False):
            print("ratios " + " ".join(f"{r_:.3g}" for r_ in ratios), flush=True)
    raise SystemExit(0)

# --- sampling ---------------------------------------------------------------------------------
# QoIs: mean over [0.25, 0.5]^d, mean over the domain, value at the centre
xyz = fd.SpatialCoordinate(mesh)
cond = fd.And(xyz[0] > 0.25, xyz[0] < 0.5)
for a_ in range(1, dim):
    cond = fd.And(cond, fd.And(xyz[a_] > 0.25, xyz[a_] < 0.5))
G = np.stack([fd.assemble(fd.conditional(cond, 4.0 ** dim, 0.0) * v * fd.dx).dat.data_ro[:nown].copy(),
              fd.assemble(v * fd.dx).dat.data_ro[:nown].copy(),
              np.all(keys == P * N // 2, 1).astype(float)], 1)
qnames = ["box mean", "domain mean", "centre value"]

# exact references on rank 0: Var[g^T x] and diag(Cov) for Cov = A^{-1} (MGMC) or L^{-1} M_L L^{-1} (SPDE)
var_q = dvar = None
if not timing_only:
    ip, ix, dv = A.getValuesCSR()
    parts = comm.gather((rstart, ip, ix, dv), root=0)
    Gall = gather_owned(G)
    mlall = gather_owned(ml.array_r) if mode == "spde" else None
    if rank == 0:
        indptr, idx, val = [0], [], []
        for r0, ipr, ixr, dvr in sorted(parts, key=lambda t_: t_[0]):
            indptr += list(indptr[-1] + ipr[1:])
            idx.append(ixr)
            val.append(dvr)
        Asp = sp.csr_matrix((np.concatenate(val), np.concatenate(idx), np.array(indptr)), shape=(ndof, ndof)).tocsc()
        lu = spla.splu(Asp)
        if mode == "spde":
            var_q = np.array([(w_ := lu.solve(g)) @ (mlall * w_) for g in Gall.T])
            if ndof <= 6000:
                Linv = np.linalg.inv(Asp.toarray())
                dvar = np.einsum("ij,j,ij->i", Linv, mlall, Linv)
        else:
            var_q = np.array([g @ lu.solve(g) for g in Gall.T])
            dvar = np.diag(np.linalg.inv(Asp.toarray())) if ndof <= 6000 else None


def sample(ksp_, solve):
    """nburnin + nsamples draws; returns the QoI chain, the pointwise variance (rank 0) and ms/sample."""
    sample.its = 0
    qs = np.zeros((nsamples, G.shape[1]))
    s1, s2 = np.zeros(nown), np.zeros(nown)
    rng = np.random.default_rng(seed + 7919 * rank)
    b, x = A.createVecs()
    b.zeroEntries()
    x.zeroEntries()
    for it in range(nburnin + nsamples):
        if it == nburnin:
            comm.Barrier()
            t0 = time.perf_counter()
        if solve:  # SPDE: independent sample x = L^{-1} M_L^{1/2} z
            b.array[:] = rng.standard_normal(nown)
            b.pointwiseMult(b, sqrt_ml)
        ksp_.solve(b, x)  # MGMC: one cycle from the current state
        sample.its += ksp_.getIterationNumber()
        if it >= nburnin:
            xa = x.array_r
            qs[it - nburnin] = comm.allreduce(xa @ G, op=MPI.SUM)
            s1 += xa
            s2 += xa * xa
    comm.Barrier()
    t_sample = 1e3 * (time.perf_counter() - t0) / nsamples
    var = None if timing_only else gather_owned(s2 / nsamples - (s1 / nsamples) ** 2)
    return qs, var, t_sample


def report(name, qs, var, t_sample, t_coarse_, t_setup_, extra=""):
    if rank:
        return
    out = [f"RESULT {name:9s} {tag} | setup {t_coarse_:.2f} s (coarse space, subdomains) + {t_setup_:.2f} s (PETSc), "
           f"{t_sample:.2f} ms/sample{extra} | {info}"]
    iacts = [iact(qs[:, k]) for k in range(G.shape[1])]
    out.append("  IACT " + ", ".join(f"{qnames[k]} {t_:.2f} +- {e_:.2f}" for k, (t_, e_) in enumerate(iacts)))
    for k in range(G.shape[1] if var_q is not None else 0):
        tau = max(iacts[k][0], 1.0)
        r = qs[:, k].var() / var_q[k]
        zm = qs[:, k].mean() / np.sqrt(var_q[k] * tau / nsamples)
        out.append(f"  {qnames[k]:12s}: var ratio {r:.3f} +- {np.sqrt(2 * tau / nsamples):.3f}  mean z {zm:+.2f}")
    if dvar is not None:
        rat = var / dvar
        out.append(f"  pointwise var ratio: mean {rat.mean():.4f}  rms(ratio - 1) {np.sqrt(((rat - 1) ** 2).mean()):.4f}"
                   f"  min {rat.min():.3f}  max {rat.max():.3f}")
    print("\n".join(out), flush=True)


if mode == "spde":
    qs, var, t_s = sample(ksp, solve=True)
    report("spde", qs, var, t_s, t_coarse, t_setup, f", {sample.its / (nburnin + nsamples):.1f} CG its/sample")
else:
    if opts.getBool("cholesky", False):  # exact reference sampler (independent samples) of the same length
        kc = make_ksp("cholsampler")
        comm.Barrier()
        t0 = time.perf_counter()
        kc.setUp()
        report("cholesky", *sample(kc, solve=False), 0.0, time.perf_counter() - t0)
    report("mgmc", *sample(ksp, solve=False), t_coarse, t_setup)
