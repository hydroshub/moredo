"""
Figure 9 without the R_in map (that map is already Figure 1): two panels only.
  (a) R_in (y) vs mean F_add (x) per catchment, coloured by Koppen-Geiger major class, with the
      per-class OLS regression lines of F_add on R_in drawn in these swapped axes.
  (b) Per-class regression slope beta = dF_add/dR_in (dot plot), each dot labelled with the class
      Pearson r, and a vertical line at the pooled global slope.

Layout: panel a spans the left column; the right column holds the Koppen-Geiger legend on top and
panel b below. The quantity is a regression slope, not a "sensitivity", so the axis reads
"Slope beta". Reads the cached non-closure panel (fig10_nonclosure_v2_cache.pkl); no model
inference.

Usage: python scripts/analysis/fig9_relation_2panel.py
"""
import os
import sys
import pickle

import numpy as np
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE
from basin_attributes import KG_MAJOR

from project_paths import RESULTS_DIR as RD
CACHE = os.path.join(RD, "figures", "fig10_nonclosure_v2_cache.pkl")
OUT = os.path.join(RD, "figures_manuscript_final", "fig_9_nonclosure_relation_2panel.png")
RIN_LABEL = r"$R_{\mathrm{in}}^{(s)}$"


def main():
    p = pickle.load(open(CACHE, "rb"))
    v = p["v"].astype(bool)
    r_in = -np.asarray(p["eps_wb"], float)
    fadd = np.asarray(p["fadd"], float)
    cat = np.asarray(p["kg_major_id"])
    kg_ids = list(KG_MAJOR.keys())

    rlo, rhi = -3.0, 5.0   # R_in range (now the y axis of panel a)
    mall = v & np.isfinite(r_in) & np.isfinite(fadd)
    global_beta = linregress(r_in[mall], fadd[mall]).slope

    fig = uplt.figure(figwidth="200mm", figheight="100mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=2, ncols=2, wspace=5, hspace=1.5, wratios=(1.25, 1.0),
                       hratios=(0.28, 1.0))

    def _style(ax):
        ax.format(grid=False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2)

    # --- a: scatter with per-class OLS regression lines ---
    ax_a = fig.add_subplot(gs[:, 0], abc=True, abcloc="l")
    beta, rval = {}, {}
    for kg_id in kg_ids:
        m = v & (cat == kg_id) & np.isfinite(r_in) & np.isfinite(fadd)
        if not np.any(m):
            continue
        ax_a.scatter(fadd[m], r_in[m], color=KG_MAJOR[kg_id]["color"], label=KG_MAJOR[kg_id]["code"],
                     s=22, alpha=0.8, edgecolor="k", lw=0.3, zorder=2)
        if np.sum(m) > 2:
            res = linregress(r_in[m], fadd[m])          # OLS of F_add on R_in, as in the text
            beta[kg_id], rval[kg_id] = res.slope, res.rvalue
            rg = np.linspace(rlo, rhi, 200)
            ax_a.plot(res.slope * rg + res.intercept, rg, color=KG_MAJOR[kg_id]["color"],
                      lw=1.5, zorder=3)
    ax_a.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax_a.format(xlim=(0, 3), ylim=(rlo, rhi),
                xlabel="Mean $F_{add}$ (mm day$^{-1}$)", ylabel=f"{RIN_LABEL} (mm day$^{{-1}}$)",
                xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
                xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE)
    _style(ax_a)

    # --- b first (so it gets the letter b), then the legend in its own cell above it ---
    ax_b = fig.add_subplot(gs[1, 1], abc=True, abcloc="l")
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", ncols=4, frame=False, prop={"size": LEGENDSIZE - 1},
                  title="Köppen–Geiger class", titlefontsize=LEGENDSIZE - 1,
                  columnspacing=0.9, handletextpad=0.3, borderpad=0.2, labelspacing=0.3)

    # --- b: per-class regression slope beta (dot plot) ---
    order = sorted(beta, key=beta.get)
    ypos = np.arange(len(order))
    for y, kg_id in zip(ypos, order):
        ax_b.scatter(beta[kg_id], y, color=KG_MAJOR[kg_id]["color"], s=60, edgecolor="k",
                     lw=0.4, zorder=3)
        ax_b.text(beta[kg_id] + 0.020, y, f"r={rval[kg_id]:.2f}", va="center", ha="left",
                  fontsize=TICKLABELSIZE - 3, color="0.3")
    ax_b.axvline(global_beta, color="gray", ls="-", lw=1.2, zorder=1)
    ax_b.text(global_beta + 0.006, 1.0, "global $\\beta$", transform=ax_b.get_xaxis_transform(),
              rotation=90, ha="left", va="top", fontsize=TICKLABELSIZE - 2, color="gray")
    ax_b.format(yticks=ypos, yticklabels=[KG_MAJOR[k]["code"] for k in order],
                xlim=(0, 0.42), ylim=(-0.6, len(order) - 0.4),
                xlabel=r"Slope $\beta$ = $\Delta F_{add}$/$\Delta R_{in}$", ylabel="",
                xlabelsize=LABELSIZE - 1, xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE)
    _style(ax_b)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"global beta = {global_beta:.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
