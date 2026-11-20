"""Weak-scaling figure of the twisted toroid (alpha = 1, AMGMC) from the CSV of collect.py: time per
iteration (left; exact parallel Gibbs smoother and the Hogwild baseline) and per independent sample, i.e. times
the QoI IACT written next to each point (right; Gibbs only: Hogwild does not sample the target distribution), against
the number of ranks. The weak-scaling efficiency
t(1 rank) / t(P ranks) is written next to each point of the left panel.

    python collect.py results/toroid/weak3 --csv weak3.csv
    python plot_weak_scaling.py weak3.csv -o toroid_weak_scaling.pdf
"""

import argparse
import csv
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt

# same style as ../graph/graph_prior_scaling_plot.py
mpl.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
                     "lines.markersize": 4, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "pdf.fonttype": 42})
USETEX = shutil.which("latex") is not None and shutil.which("pdflatex") is not None
if USETEX:
    mpl.rcParams.update({"text.usetex": True, "pgf.texsystem": "pdflatex", "pgf.rcfonts": False})

STYLE = {"gibbs": dict(label="Gibbs (exact)", color="#0072B2", marker="o", ls="-"),
         "hogwild": dict(label="Hogwild", color="#D55E00", marker="s", ls="--", mfc="none")}

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("-o", "--output", default="toroid_weak_scaling.pdf")
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.csv)) if r["gamg"] == "default"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.1), constrained_layout=True)
for sm, st in STYLE.items():
    rs = sorted((r for r in rows if r["smoother"] == sm), key=lambda r: int(r["ranks"]))
    if not rs:
        continue
    p = [int(r["ranks"]) for r in rs]
    ax1.plot(p, [1e3 * float(r["t_iter"]) for r in rs], **st)
    for r in rs[1:]:
        ax1.annotate(f"{float(r['efficiency']):.2f}", (int(r["ranks"]), 1e3 * float(r["t_iter"])), fontsize=7,
                     textcoords="offset points", xytext=(4, -9 if sm == "hogwild" else 4), color=st["color"])
    if sm == "gibbs":  # Hogwild does not sample the target distribution: no time per independent sample
        ax2.plot(p, [1e3 * float(r["t_indep"]) for r in rs], **st)
        for r in rs:
            ax2.annotate(f"{float(r['iact']):.2f}", (int(r["ranks"]), 1e3 * float(r["t_indep"])), fontsize=7,
                         textcoords="offset points", xytext=(4, -9), color=st["color"])
for ax, lab in ((ax1, "time per iteration [ms]"), (ax2, "time per indep. sample [ms]")):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 64, 512], ["1", "8", "64", "512"])
    ax.minorticks_off()
    ax.set_xlabel("number of ranks")
    ax.set_ylabel(lab)
    ax.set_ylim(0, 1.15 * ax.get_ylim()[1])
    ax.grid(True, alpha=0.3)
ax1.legend(frameon=False)
fig.savefig(a.output, backend="pgf" if USETEX and a.output.endswith(".pdf") else None)
