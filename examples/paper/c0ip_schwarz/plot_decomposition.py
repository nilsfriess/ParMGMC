"""Sketch of the box decomposition of the alpha = 2 example: boxes of width H = 1/M on a mesh with
h = 1/N, each extended by delta = H/4; the smooth partition of unity chi_b; and why boxes two apart
conflict in 3D (H/h = 4) but not in 2D (H/h = 8): the gap between them is H - 2 delta, the C0IP
stencil couples unknowns up to 2h apart.

    python plot_decomposition.py -o decomposition.pdf
"""

import argparse
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
USETEX = shutil.which("latex") is not None and shutil.which("pdflatex") is not None
if USETEX:
    mpl.rcParams.update({"text.usetex": True, "pgf.texsystem": "pdflatex", "pgf.rcfonts": False})
COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


def chi1d(x, k, M, delta):
    f = np.ones_like(x)
    if k > 0:
        f *= smoothstep((x - (k / M - delta)) / (2 * delta))
    if k < M - 1:
        f *= smoothstep(((k + 1) / M + delta - x) / (2 * delta))
    return f


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="decomposition.pdf")
a = ap.parse_args()

fig = plt.figure(figsize=(7.0, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.75])
ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# (a) 1D cut through the unit square: M = 4 boxes, N = 32 cells (H = 8h), delta = H/4 = 2h
M, N = 4, 32
H, h, delta = 1 / M, 1 / N, 1 / (4 * M)
x = np.linspace(0, 1, 2001)
for k in range(M):
    ax1.plot(x, chi1d(x, k, M, delta), color=COL[k], label=rf"$\chi_{k}$")
    lo, hi = max(k * H - delta, 0), min((k + 1) * H + delta, 1)
    ax1.plot([lo, hi], [1.12 + 0.07 * (k % 2)] * 2, color=COL[k], lw=3, solid_capstyle="butt")
for k in range(1, M):
    ax1.axvspan(k * H - delta, k * H + delta, color="0.85", zorder=0)
ax1.set_xticks(np.arange(0, 1 + 1e-9, h), minor=True)
ax1.set_xticks(np.arange(0, 1 + 1e-9, H))
ax1.tick_params(axis="x", which="minor", length=3)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.05, 1.3)
ax1.set_yticks([0, 1])
ax1.set_xlabel(r"$x$ (minor ticks: elements, $h=1/32$)")
ax1.set_title(r"(a) 1D cut: $H=8h$, $\delta=H/4=2h$", fontsize=9)
ax1.annotate("", xy=(H + delta, -0.02), xytext=(H - delta, -0.02), arrowprops=dict(arrowstyle="<->", lw=0.8))
ax1.text(H, 0.04, r"$2\delta=4h$", ha="center", fontsize=8)
ax1.text(0.5, 1.24, "extended boxes", ha="center", fontsize=8)
ax1.legend(loc="center right", fontsize=7, frameon=False, bbox_to_anchor=(1.0, 0.45))

# (b) 2D: chi of box (1, 1) on the 32 x 32 mesh, box grid and its extended box
X, Y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
C = chi1d(X, 1, M, delta) * chi1d(Y, 1, M, delta)
im = ax2.imshow(C, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
for t in np.arange(0, 1 + 1e-9, h):
    ax2.axhline(t, color="0.8", lw=0.3)
    ax2.axvline(t, color="0.8", lw=0.3)
for t in np.arange(0, 1 + 1e-9, H):
    ax2.axhline(t, color="k", lw=0.9)
    ax2.axvline(t, color="k", lw=0.9)
ax2.add_patch(plt.Rectangle((H - delta, H - delta), H + 2 * delta, H + 2 * delta, fill=False, ec=COL[1], lw=1.4, ls="--"))
ax2.add_patch(plt.Rectangle((H + delta, H + delta), H - 2 * delta, H - 2 * delta, fill=False, ec=COL[2], lw=1.2))
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title(r"(b) $\chi$ of one box (black: boxes, dashed: extended box, green: $\chi=1$)", fontsize=9)
fig.colorbar(im, ax=ax2, shrink=0.8)

# (c) conflict of boxes i and i+2: gap H - 2 delta vs. C0IP coupling reach 2h
for row, (Hh, label) in enumerate([(8, r"2D: $H=8h$, $\delta=2h$: gap $4h>2h$, no conflict"),
                                   (4, r"3D: $H=4h$, $\delta=h$: gap $2h$, coupled, conflict")]):
    y = 1.0 - row
    d = Hh / 4
    for i, c in zip((0, 2), (COL[0], COL[3])):
        ax3.add_patch(plt.Rectangle((i * Hh - d, y - 0.15), Hh + 2 * d, 0.3, color=c, alpha=0.35))
        ax3.add_patch(plt.Rectangle((i * Hh, y - 0.15), Hh, 0.3, fill=False, ec=c, lw=1.0))
    ax3.add_patch(plt.Rectangle((Hh - d, y - 0.3), Hh + 2 * d, 0.6, fill=False, ec=COL[1], ls="--", lw=1.0))
    gap0, gap1 = Hh + d, 2 * Hh - d
    ax3.annotate("", xy=(gap1, y + 0.27), xytext=(gap0, y + 0.27), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax3.text((gap0 + gap1) / 2, y + 0.33, f"gap ${gap1 - gap0:g}h$", ha="center", fontsize=8)
    ax3.annotate("", xy=(gap0 + 2, y - 0.22), xytext=(gap0, y - 0.22), arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.3"))
    ax3.text(gap0 + 1, y - 0.42, "C$^0$IP reach $2h$", ha="center", fontsize=7, color="0.3")
    ax3.text(-3.2, y, label, ha="right", va="center", fontsize=8)
    for t in range(int(-d), int(3 * Hh + d) + 1):
        ax3.plot([t, t], [y - 0.05, y + 0.05], color="0.5", lw=0.5)
ax3.set_xlim(-26, 26)
ax3.set_ylim(-0.6, 1.5)
ax3.axis("off")
ax3.set_title(r"(c) boxes $i$ and $i+2$ (blue, pink; box $i+1$ dashed): ticks are elements", fontsize=9)

fig.savefig(a.output, backend="pgf" if USETEX and a.output.endswith(".pdf") else None)
