"""Two-level MGMC for a C0 interior penalty (P2) discretisation of (kappa^2 - Delta)^2 on the unit
square with natural BCs (alpha = 2), serial proof of concept.

Fine level: PCPATCHGIBBS, i.e. multiplicative overlapping block Gibbs over M x M boxes (each box
extended by delta on every side, visited in lexicographic order, exact dense Cholesky samples).
Coarse level: Galerkin projection onto Z = {chi_j p}, where chi_j is a smooth (C2, quintic
smoothstep) partition of unity over the boxes and p runs over {1, x, y, x^2, xy, y^2} (centred and
scaled per box); sampled exactly with PCCHOLSAMPLER.  Both run inside PCGAMGMC with a
user-configured PCMG (2 levels, interpolation Z).

a(u,v) = sum_T (D2u, D2v) + 2k^2 (grad u, grad v) + k^4 (u, v)
         - <{d2u/dn2}, [dv/dn]> - <{d2v/dn2}, [du/dn]> + sigma/h <[du/dn], [dv/dn]>   (interior edges)

Options
  -N cells per side, -M boxes per side, -delta overlap (default H/4), -kappa, -sigma (20)
  -ns affine|quad   near-kernel per box (default quad)
  -mode rho         deterministic smoother and coarse solve: convergence factor of the two-level
                    cycle (power iteration in the energy norm, -rho_its steps)
  -mode sample      -nsamples samples after -nburnin; QoI and pointwise variances vs. exact, plus an
                    exact Cholesky sampler of the same length as reference for the statistical noise
  -coarse_noise 0   negative control: deterministic coarse step (the chain is then biased)
  -seed s
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
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
mode = opts.getString("mode", "rho")
nsamples = opts.getInt("nsamples", 20000)
nburnin = opts.getInt("nburnin", 100)
rho_its = opts.getInt("rho_its", 50)
coarse_noise = opts.getBool("coarse_noise", True)
seed = opts.getInt("seed", 1)
pymgmc.seed(seed)

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
ip, ix, dv = A.getValuesCSR()
Asp = sp.csr_matrix((dv, ix, ip), shape=A.getSize())
X = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro.copy()
ndof = X.shape[0]

# --- boxes (patches) and coarse space ----------------------------------------------------------


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
        polys = [np.ones(ndof), xr, yr] + ([xr * xr, xr * yr, yr * yr] if ns == "quad" else [])
        cols += [sp.csr_matrix(chi * p).T for p in polys]
Zs = sp.hstack(cols).tocsr()
Zs.eliminate_zeros()
assert np.abs(sum(c.toarray().ravel() for c in cols[:: len(polys)]) - 1).max() < 1e-12, "not a partition of unity"
cover = np.zeros(ndof)
for b in boxes:
    cover[b] += 1
assert cover.min() >= 1, "every DOF must lie in a patch"
offsets = np.concatenate([[0], np.cumsum([len(b) for b in boxes])]).astype(PETSc.IntType)
indices = np.concatenate(boxes).astype(PETSc.IntType)
Z = PETSc.Mat().createAIJ(size=Zs.shape, csr=(Zs.indptr.astype(PETSc.IntType), Zs.indices.astype(PETSc.IntType), Zs.data))


# --- two-level MGMC inside PCGAMGMC --------------------------------------------------------------
def make_ksp(deterministic, coarse_sample=True):
    for k in [k for k in opts.getAll() if k.startswith("gamgmc_") or k.startswith("pc_gamgmc")]:
        opts.delValue(k)
    opts["pc_gamgmc_mg_type"] = "mg"
    opts["gamgmc_mg_levels_pc_type"] = "patchgibbs"  # otherwise PCSetUp_GAMGMC injects sorgibbs
    if deterministic:
        opts["gamgmc_mg_levels_pc_patchgibbs_deterministic"] = "1"
    if deterministic or not coarse_sample:
        opts["gamgmc_mg_coarse_ksp_type"] = "preonly"
        opts["gamgmc_mg_coarse_pc_type"] = "cholesky"
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
    sm.getPC().setType("patchgibbs")
    pymgmc.PCPatchGibbsSetPatches(sm.getPC(), offsets, indices)
    ksp.setFromOptions()
    ksp.setUp()
    return ksp, mg


def check_setup(mg):
    """The configuration must survive PCSetUp_GAMGMC."""
    assert mg.getType() == "mg" and mg.getMGLevels() == 2, (mg.getType(), mg.getMGLevels())
    assert mg.getMGSmoother(1).getPC().getType() == "patchgibbs", mg.getMGSmoother(1).getPC().getType()
    P = mg.getMGInterpolation(1)
    assert P.getSize() == Z.getSize()
    Ac = mg.getMGCoarseSolve().getOperators()[0]
    ia, ja, va = Ac.getValuesCSR()
    Ac_ref = (Zs.T @ Asp @ Zs).toarray()
    Ac_got = sp.csr_matrix((va, ja, ia), shape=Ac.getSize()).toarray()
    return np.abs(Ac_got - Ac_ref).max() / np.abs(Ac_ref).max()


def energy(x):
    y = A.createVecLeft()
    A.mult(x, y)
    return np.sqrt(x.dot(y))


tag = f"N={N} M={M} H/h={N // M} delta={delta:g} k={kappa:g} ns={ns} ndof={ndof} ncoarse={Zs.shape[1]}"

if mode == "rho":
    ksp, mg = make_ksp(deterministic=True)
    gal = check_setup(mg)
    b, x = A.createVecs()
    b.zeroEntries()
    x.array[:] = np.random.default_rng(1).standard_normal(ndof)
    x.scale(1.0 / energy(x))
    ratios = []
    for it in range(rho_its):
        ksp.solve(b, x)  # one cycle: x <- x + B(b - A x)
        e = energy(x)
        ratios.append(e)
        x.scale(1.0 / e)
    last = np.asarray(ratios[-10:])
    rho = float(np.exp(np.mean(np.log(last))))
    print(f"RESULT rho {tag} | rho {rho:.4f} ({last.min():.4f}-{last.max():.4f}) | Galerkin check {gal:.1e}", flush=True)

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
    gs = [fd.assemble(ind * v * fd.dx).dat.data_ro.copy(), fd.assemble(v * fd.dx).dat.data_ro.copy()]
    e = np.zeros(ndof)
    e[np.argmin(np.linalg.norm(X - 0.5, axis=1))] = 1.0
    gs.append(e)
    G = np.stack(gs, 1)
    qnames = ["box mean", "domain mean", "centre value"]

    # exact references
    lu = sp.linalg.splu(Asp.tocsc())
    var_q = np.array([g @ lu.solve(g) for g in gs])
    pointwise = ndof <= 6000
    if pointwise:
        dvar = np.diag(sl.inv(Asp.toarray()))

    def run_chain(ksp_):
        qs = np.zeros((nsamples, len(gs)))
        s1, s2 = np.zeros(ndof), np.zeros(ndof)
        b, x = A.createVecs()
        b.zeroEntries()
        x.zeroEntries()
        for it in range(nburnin + nsamples):
            ksp_.solve(b, x)
            if it >= nburnin:
                xa = x.array_r
                qs[it - nburnin] = xa @ G
                s1 += xa
                s2 += xa * xa
        mean = s1 / nsamples
        return qs, mean, s2 / nsamples - mean ** 2

    def report(name, qs, mean, var):
        taus = [iact(qs[:, k]) for k in range(len(gs))]
        out = [f"RESULT {name:14s} {tag}"]
        for k in range(len(gs)):
            r = qs[:, k].var() / var_q[k]
            err = np.sqrt(2 * taus[k] / nsamples)
            zm = qs[:, k].mean() / np.sqrt(var_q[k] * taus[k] / nsamples)
            out.append(f"  {qnames[k]:12s}: var ratio {r:.3f} +- {err:.3f}  mean z {zm:+.2f}  IACT {taus[k]:.2f}")
        if pointwise:
            rat = var / dvar
            out.append(f"  pointwise var ratio: mean {rat.mean():.4f}  rms(ratio - 1) {np.sqrt(((rat - 1) ** 2).mean()):.4f}"
                       f"  min {rat.min():.3f}  max {rat.max():.3f}")
        print("\n".join(out), flush=True)

    # exact reference sampler (independent samples) of the same length
    kc = PETSc.KSP().create()
    kc.setOperators(A, A)
    kc.setType("richardson")
    kc.setTolerances(max_it=1)
    kc.setNormType(PETSc.KSP.NormType.NONE)
    kc.setInitialGuessNonzero(True)
    kc.getPC().setType("cholsampler")
    kc.setUp()
    report("cholesky", *run_chain(kc))

    ksp, mg = make_ksp(deterministic=False, coarse_sample=coarse_noise)
    gal = check_setup(mg)
    report("mgmc" if coarse_noise else "mgmc NO-NOISE", *run_chain(ksp))
    print(f"Galerkin check {gal:.1e}", flush=True)
