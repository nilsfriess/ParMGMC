"""Per-sample cost breakdown of MGMC (two-level Schwarz-Gibbs) and SPDE (CG+GAMG) runs from -log_view.

Times are PETSc's max over ranks, divided by the number of applications (nburnin + nsamples = 1010).
Usage: python breakdown.py <run.log_view.txt> [...]   (the matching .out must sit next to it)
"""
import re
import sys

NAPP = 1010


def events(path):
    """name -> [count, max time, flops (max rank)] summed over all stages."""
    ev = {}
    in_table = False
    for line in open(path):
        if line.startswith("--- Event Stage"):
            in_table = True
            continue
        if in_table and line.startswith("----"):
            in_table = False
        if not in_table:
            continue
        m = re.match(r"^(\S.{0,17}?)\s+(\d+)\s+[\d.]+\s+([\d.e+-]+)\s+[\d.]+\s+([\d.e+-]+)", line)
        if not m:
            continue
        name, cnt, t, fl = m.group(1).strip(), int(m.group(2)), float(m.group(3)), float(m.group(4))
        e = ev.setdefault(name, [0, 0.0, 0.0])
        e[0] += cnt
        e[1] += t
        e[2] += fl
    return ev


def result_line(path):
    out = path.replace(".log_view.txt", ".out")
    for line in open(out):
        if line.startswith("RESULT"):
            return line.strip()
    return ""


def per(ev, name, k=1):
    e = ev.get(name)
    return (e[k] / NAPP) if e else 0.0


def report(path):
    ev = events(path)
    res = result_line(path)
    ms = float(re.search(r"([\d.]+) ms/sample", res).group(1)) if "ms/sample" in res else float("nan")
    head = res.split("|")[0].replace("RESULT", "").strip()
    if "spde" in res.split()[1]:
        its = float(re.search(r"([\d.]+) CG its", res).group(1))
        mm = per(ev, "MatMult", 0)
        print(f"{head}\n  total {ms:.2f} ms/sample, {its:.1f} CG its; "
              f"V-cycles (PCApply) {1e3 * per(ev, 'PCApply'):.2f} ms; "
              f"MatMult (all levels) {mm:.0f}/sample {1e3 * per(ev, 'MatMult'):.2f} ms; "
              f"dot/norm reductions {per(ev, 'VecTDot', 0) + per(ev, 'VecNorm', 0):.0f}/sample "
              f"{1e3 * (per(ev, 'VecTDot') + per(ev, 'VecNorm')):.2f} ms; "
              f"coarse level {1e3 * per(ev, 'MGSmooth Level 0'):.2f} ms; "
              f"VecScatterEnd {1e3 * per(ev, 'VecScatterEnd'):.2f} ms")
    else:
        sweep = per(ev, "SchwarzGibbsSweep")
        print(f"{head}\n  total {ms:.2f} ms/sample; fine sweep {1e3 * sweep:.2f} ms "
              f"[MatMult {per(ev, 'MatMult', 0):.0f}/sample {1e3 * per(ev, 'MatMult'):.2f} ms, "
              f"subdomain solves (fwd+bwd, max rank) {1e3 * (per(ev, 'MatForwardSolve') + per(ev, 'MatBackwardSolve')):.2f} ms, "
              f"VecScatterEnd {1e3 * per(ev, 'VecScatterEnd'):.2f} ms]; "
              f"coarse sample (rank 0) {1e3 * per(ev, 'MGSmooth Level 0'):.2f} ms; "
              f"interp+restrict (incl. waiting for rank 0) {1e3 * per(ev, 'MGInterp Level 1'):.2f} ms")


for p in sys.argv[1:]:
    report(p)
