"""Geometric vs algebraic MGMC on the unit square: figure and LaTeX table from the wrench_refinement.py runs
of runs_unitsquare_k10*.txt (JSON records and QoI chains, one or more results directories).

Left: time per independent sample (time per iteration x IACT of the QoI) against the number of unknowns,
for both QoIs, with the Cholesky sampler (independent samples) for reference. Right: the IACTs with
standard errors. IACT: Sokal's adaptive window (c = 5), standard error tau * sqrt(2 (2W + 1) / N).

    python plot_geo_alg.py <results-dir>... -o geo_alg.pdf --table geo_alg_table.tex
"""

import argparse
import glob
import json
import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
                     "lines.markersize": 4, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
USETEX = shutil.which("latex") is not None and shutil.which("pdflatex") is not None
if USETEX:
    mpl.rcParams.update({"text.usetex": True, "pgf.texsystem": "pdflatex", "pgf.rcfonts": False})
# Samplers by their -wr_label (variants of amgmc get their own label); others are printed but not plotted
COL = {"gmgmc": "#0072B2", "amgmc": "#D55E00", "amgmc_tuned": "#CC79A7", "amgmc_ns2": "#E69F00",
       "cholesky": "#009E73"}
NAME = {"gmgmc": "geometric MGMC", "amgmc": "algebraic MGMC", "amgmc_tuned": "algebraic MGMC (tuned)",
        "amgmc_ns2": "algebraic MGMC (2 prol. smoothing steps)", "cholesky": "Cholesky"}


def sokal(x, c=5):
    """IACT with Sokal's adaptive window and its standard error."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau, w = 1.0, 1
    for w in range(1, n):
        tau = 1 + 2 * acf[1:w + 1].sum()
        if w >= c * tau:
            break
    return max(tau, 1.0), tau * np.sqrt(2 * (2 * w + 1) / n)


def load(dirs):
    rows = []
    for path in sorted(p for d in dirs for p in glob.glob(os.path.join(d, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("driver") != "wrench_refinement.py" or r["problem"]["mesh"] != "unitsquare":
            continue
        tau, se = sokal(np.load(path[:-5] + ".npy"))
        sampler = r["sampler"].get("label") or r["sampler"]["name"]
        rows.append({"sampler": sampler, "qoi": "smooth" if r["problem"]["qoi"]["radius"] >= 0.5 else "local",
                     "ndofs": r["problem"]["ndofs"], "t_iter": r["timings"]["per_iteration"],
                     "t_setup": r["timings"]["setup"], "nsamples": r["run"]["nsamples"],
                     "tau": 1.0 if sampler == "cholesky" else tau, "tau_se": 0.0 if sampler == "cholesky" else se,
                     "var_ratio": (r["reference"] or {}).get("var_ratio"),
                     "var_ratio_se": (r["reference"] or {}).get("var_ratio_se"), "file": os.path.basename(path)})
    return rows


def series(rows, sampler, qoi=None):
    rs = [r for r in rows if r["sampler"] == sampler and (qoi is None or r["qoi"] == qoi)]
    return sorted(rs, key=lambda r: r["ndofs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("-o", "--output", default="geo_alg.pdf")
    ap.add_argument("--table")
    a = ap.parse_args()
    rows = load(a.results)

    for r in sorted(rows, key=lambda r: (r["sampler"], r["qoi"], r["ndofs"])):
        vr = f"{r['var_ratio']:.3f} +- {r['var_ratio_se']:.3f}" if r["var_ratio"] is not None else "-"
        print(f"{r['sampler']:12s} {r['qoi']:6s} {r['ndofs']:9d}  t/iter {1e3 * r['t_iter']:9.3f} ms  "
              f"IACT {r['tau']:5.2f} +- {r['tau_se']:4.2f}  N {r['nsamples']:6d}  setup {r['t_setup']:7.2f} s  var {vr}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.3), constrained_layout=True)
    for s in [s for s in COL if s != "cholesky"]:
        for q, ls, mfc in (("smooth", "-", COL[s]), ("local", "--", "white")):
            rs = series(rows, s, q)
            if not rs:
                continue
            n = [r["ndofs"] for r in rs]
            lab = f"{NAME[s]}, {q} QoI"
            ax1.plot(n, [r["t_iter"] * r["tau"] for r in rs], ls=ls, marker="o", color=COL[s], mfc=mfc, label=lab)
            ax2.errorbar(n, [r["tau"] for r in rs], yerr=[r["tau_se"] for r in rs], ls=ls, marker="o",
                         color=COL[s], mfc=mfc, capsize=2, label=lab)
    rs = series(rows, "cholesky")
    if rs:
        ax1.plot([r["ndofs"] for r in rs], [r["t_iter"] for r in rs], ls=":", marker="s", color=COL["cholesky"],
                 label=NAME["cholesky"])
    if rows:
        n = np.array([min(r["ndofs"] for r in rows), max(r["ndofs"] for r in rows)], float)
        t0 = min(r["t_iter"] for r in series(rows, "gmgmc")) if series(rows, "gmgmc") else 1e-4
        ax1.plot(n, t0 * n / n[0], color="0.6", lw=0.8, label=r"$\propto n$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of unknowns $n$")
    ax1.set_ylabel("time per indep. sample [s]")
    ax2.set_xscale("log")
    ax2.set_xlabel("number of unknowns $n$")
    ax2.set_ylabel("IACT")
    ax2.set_ylim(bottom=0)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.savefig(a.output, backend="pgf" if USETEX and a.output.endswith(".pdf") else None)

    if a.table:
        sizes = sorted({r["ndofs"] for r in rows})
        get = {(r["sampler"], r["qoi"], r["ndofs"]): r for r in rows}
        lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
                 r" & \multicolumn{3}{c}{geometric MGMC} & \multicolumn{3}{c}{algebraic MGMC} & Cholesky \\"
                 r" \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8}",
                 r"unknowns & [ms] & IACT smooth & IACT local & [ms] & IACT smooth & IACT local & [ms] \\",
                 r"\midrule"]
        for n in sizes:
            cells = [f"{n:,}".replace(",", r"\,")]
            for s in ("gmgmc", "amgmc"):
                rs, rl = get.get((s, "smooth", n)), get.get((s, "local", n))
                cells.append(f"{1e3 * rs['t_iter']:.3g}" if rs else "--")
                cells += [f"{r['tau']:.2f}" if r else "--" for r in (rs, rl)]
            rc = get.get(("cholesky", "smooth", n)) or get.get(("cholesky", "local", n))
            cells.append(f"{1e3 * rc['t_iter']:.3g}" if rc else "--")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(a.table, "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
