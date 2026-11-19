"""Table and figure for the alpha = 2 example from the output of c0ip_mgmc.py (RESULT lines).

Series: growing number of box subdomains of fixed size, with 64 (2D) or 32 (3D) subdomains per rank.
For every MGMC point (dim, N, M) the SPDE runs on the same mesh (N) and with the same number of
unknowns (2N) on the same number of ranks are matched. rho comes from the -mode rho runs (any rank
count); the coarse-sample time per cycle from the log_view file next to the .out (MGSmooth Level 0).

    python collect.py results/**/*.out --table alpha2_table.tex --figure alpha2_scaling.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

PER_RANK = {2: 64, 3: 32}


def num(pattern, line, cast=float):
    m = re.search(pattern, line)
    return cast(m.group(1)) if m else None


def parse(path):
    """All runs in one .out file as dicts."""
    runs = []
    lines = Path(path).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("RESULT"):
            continue
        kind = line.split()[1]
        r = dict(kind=kind, file=str(path), N=num(r"\bN=(\d+)", line, int), M=num(r"\bM=(\d+)", line, int),
                 ndof=num(r"ndof=(\d+)", line, int), np=num(r"np=(\d+)", line, int))
        dim = num(r"dim=(\d)", line, int)
        if dim is None:  # older output: infer from the number of P2 unknowns
            dim = 3 if r["ndof"] == (2 * r["N"] + 1) ** 3 else 2
        r["dim"] = dim
        r["ms"] = num(r"([\d.]+) ms/sample", line)
        setup = re.search(r"setup ([\d.]+) s(?: \([^)]*\) \+ ([\d.]+) s)?", line)
        if setup:
            r["setup"] = sum(float(g) for g in setup.groups() if g)
        r["rho"] = num(r"\| rho ([\d.]+)", line)
        r["nsub"] = num(r"(\d+) subdomains", line, int)
        r["colours"] = num(r"(\d+) colours", line, int)
        r["ncoarse"] = num(r"coarse dim (\d+)", line, int)
        r["cg"] = num(r"([\d.]+) CG its", line)
        if i + 1 < len(lines) and "IACT" in lines[i + 1]:
            r["iact"] = [float(x) for x in re.findall(r"(?:mean|value) ([\d.]+)", lines[i + 1])]
        log = Path(str(path).replace(".out", ".log_view.txt"))
        if kind == "mgmc" and log.exists():
            for lline in log.read_text(errors="replace").splitlines():
                if lline.startswith("MGSmooth Level 0"):
                    f = lline.split()
                    r["coarse_ms"] = 1e3 * float(f[5]) / int(f[3])
        runs.append(r)
    return runs


def series(runs):
    """One row per (dim, N, M) of the MGMC series, with matched SPDE and rho."""
    rows = []
    mg = [r for r in runs if r["kind"] == "mgmc"]
    rho = {(r["dim"], r["N"], r["M"]): r for r in runs if r["kind"] == "rho" and r["rho"] is not None}
    spde = {(r["dim"], r["N"], r["np"]): r for r in runs if r["kind"] == "spde"}
    for key in sorted({(r["dim"], r["N"], r["M"]) for r in mg} | set(rho)):
        dim, N, M = key
        nsub = M ** dim
        np_ = max(1, nsub // PER_RANK[dim])
        cand = [r for r in mg if (r["dim"], r["N"], r["M"]) == key and r["np"] == np_]
        row = dict(dim=dim, N=N, M=M, nsub=nsub, np=np_, ndof=(2 * N + 1) ** dim, mgmc=cand[-1] if cand else None,
                   rho=rho.get(key), same_mesh=spde.get((dim, N, np_)), same_dofs=spde.get((dim, 2 * N, np_)))
        rows.append(row)
    return rows


def fmt(x, f="{:.1f}"):
    return "--" if x is None else f.format(x)


def table(rows):
    out = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
           r" & & & & & \multicolumn{2}{c}{MGMC} & \multicolumn{2}{c}{SPDE [ms]} \\ \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
           r"subdomains & unknowns & colours & $\rho$ & IACT & set-up [s] & [ms] & same mesh & same unknowns \\", r"\midrule"]
    for dim in (2, 3):
        out.append(rf"\multicolumn{{9}}{{l}}{{\emph{{unit {'square' if dim == 2 else 'cube'}}}}} \\")
        for r in (x for x in rows if x["dim"] == dim):
            m, rh = r["mgmc"], r["rho"]
            iact = max(m["iact"]) if m and m.get("iact") else None
            colours = (m or {}).get("colours") or (rh or {}).get("colours")
            out.append(f"{r['nsub']} & {r['ndof']:,} & {fmt(colours, '{}')} & {fmt(rh and rh['rho'], '{:.3f}')} & "
                       f"{fmt(iact, '{:.2f}')} & {fmt(m and m.get('setup'), '{:.2f}')} & {fmt(m and m['ms'])} & "
                       f"{fmt(r['same_mesh'] and r['same_mesh']['ms'])} & {fmt(r['same_dofs'] and r['same_dofs']['ms'])} \\\\".replace(",", r"\,"))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def figure(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, dim in zip(axes, (2, 3)):
        rs = [r for r in rows if r["dim"] == dim and r["mgmc"]]
        x = [r["nsub"] for r in rs]
        lines = [("MGMC", [r["mgmc"]["ms"] for r in rs], dict(ls="-", marker="o", color="#0072B2")),
                 ("MGMC: coarse sample (rank 0)", [r["mgmc"].get("coarse_ms") for r in rs], dict(ls=":", marker="o", mfc="none", color="#0072B2")),
                 ("SPDE, same mesh", [r["same_mesh"] and r["same_mesh"]["ms"] for r in rs], dict(ls="--", marker="s", color="#D55E00")),
                 ("SPDE, same unknowns", [r["same_dofs"] and r["same_dofs"]["ms"] for r in rs], dict(ls="-.", marker="s", mfc="none", color="#D55E00"))]
        for label, y, kw in lines:
            pts = [(a, b) for a, b in zip(x, y) if b is not None]
            if pts:
                ax.plot(*zip(*pts), label=label, **kw)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel("number of subdomains")
        ax.set_title(f"unit {'square' if dim == 2 else 'cube'} ({PER_RANK[dim]} subdomains per rank)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("time per sample [ms]")
    axes[1].legend(fontsize=8)
    fig.savefig(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--table")
    ap.add_argument("--figure")
    a = ap.parse_args()
    rows = series([r for f in sorted(a.files) for r in parse(f)])
    for r in rows:
        m = r["mgmc"] or {}
        print(f"dim {r['dim']} N {r['N']} M {r['M']}: {r['nsub']} subdomains on {r['np']} ranks, rho {fmt(r['rho'] and r['rho']['rho'], '{:.3f}')}, "
              f"MGMC {fmt(m.get('ms'))} ms (coarse {fmt(m.get('coarse_ms'))}), SPDE {fmt(r['same_mesh'] and r['same_mesh']['ms'])} / "
              f"{fmt(r['same_dofs'] and r['same_dofs']['ms'])} ms  [{m.get('file', '-')}]")
    if a.table:
        Path(a.table).write_text(table(rows) + "\n")
    if a.figure:
        figure(rows, a.figure)
