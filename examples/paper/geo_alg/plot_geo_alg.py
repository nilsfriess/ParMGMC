"""Geometric vs algebraic MGMC on the unit square: figure and LaTeX table from the wrench_refinement.py runs
of runs_unitsquare_k10*.txt (JSON records and QoI chains, one or more results directories).

Plotted: geometric MGMC and algebraic MGMC with the tuned GAMG parameters (label amgmc_tuned), both QoIs.
Left: time per independent sample (time per iteration x IACT of the QoI) divided by the number of unknowns n
(flat = optimal O(n) cost). Right: the IACTs. IACT: Sokal's adaptive window (c = 5); its standard error
tau * sqrt(2 (2W + 1) / N) is printed (not plotted). All runs (also the other variants) are printed to stdout.

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
# Plotted samplers, by their -wr_label
PLOT = {"gmgmc": ("geometric MGMC", "#0072B2"), "amgmc_tuned": ("algebraic MGMC", "#D55E00")}


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


def series(rows, sampler, qoi):
    return sorted((r for r in rows if r["sampler"] == sampler and r["qoi"] == qoi), key=lambda r: r["ndofs"])


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.2), constrained_layout=True)
    for s, (name, col) in PLOT.items():
        for q, ls, mfc in (("smooth", "-", col), ("local", "--", "white")):
            rs = series(rows, s, q)
            if not rs:
                continue
            n = np.array([r["ndofs"] for r in rs], float)
            lab = f"{name}, {q} QoI"
            ax1.plot(n, [1e6 * r["t_iter"] * r["tau"] / r["ndofs"] for r in rs], ls=ls, marker="o", color=col,
                     mfc=mfc, label=lab)
            ax2.plot(n, [r["tau"] for r in rs], ls=ls, marker="o", color=col, mfc=mfc, label=lab)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("number of unknowns $n$")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel(r"time per indep.\ sample $/\,n$ [$\mu$s]" if USETEX else "time per indep. sample / n [µs]")
    ax2.set_ylabel("IACT")
    ax2.set_ylim(top=max(1.5, ax2.get_ylim()[1]))
    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.savefig(a.output, backend="pgf" if USETEX and a.output.endswith(".pdf") else None)

    if a.table:
        sizes = sorted({r["ndofs"] for r in rows if r["sampler"] in PLOT})
        get = {(r["sampler"], r["qoi"], r["ndofs"]): r for r in rows}
        head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in PLOT.values())
        lines = [r"\begin{tabular}{r" + "rrr" * len(PLOT) + "}", r"\toprule",
                 rf" & {head} \\ " + " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(PLOT))),
                 r"unknowns" + r" & [ms] & IACT smooth & IACT local" * len(PLOT) + r" \\", r"\midrule"]
        for n in sizes:
            cells = [f"{n:,}".replace(",", r"\,")]
            for s in PLOT:
                rs, rl = get.get((s, "smooth", n)), get.get((s, "local", n))
                cells.append(f"{1e3 * rs['t_iter']:.3g}" if rs else "--")
                cells += [f"{r['tau']:.2f}" if r else "--" for r in (rs, rl)]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(a.table, "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
