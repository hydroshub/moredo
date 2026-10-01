"""
Paired per-basin NSE scatter: baseline (x) vs recurrent LSTM(A,F) (y), one panel per target,
colored by major Koppen-Geiger class. Points below the 1:1 line are catchments the recurrent
subsidy degrades; the diagnostic question is whether they cluster at low baseline NSE (inherently
hard-to-simulate catchments) and in which climate zones.

Produces: fig_paired_nse_scatter.png
"""
import os
import sys
import pickle

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE
from basin_attributes import build_basin_tables, KG_MAJOR_NAMES, KG_MAJOR_COLORS
from metrics import basin_nse_table, basin_nse_twsa

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")

VARS = ["et", "q", "swe", "twsa"]
VAR_TITLE = {"q": "Q", "et": "ET", "swe": "SWE", "twsa": "TWSA"}
LO = -1.0  # display floor for NSE (values below are clipped to the axis edge)


def nse_all(result):
    """Return the per-catchment held-out NSE table (Q, ET, SWE) for one experiment result."""
    t = basin_nse_table(result, target_channels=("q", "et", "swe"))
    t["twsa"] = basin_nse_twsa(result)
    return t


def main(base_exp="baseline", rec_exp="lstm_af", ylabel="Recurrent NSE",
         out_name="fig_paired_nse_scatter.png", row_layout=False):
    """Render Figure 4: per-catchment recurrent-vs-baseline NSE scatter per target, coloured by KG class."""
    OUT = os.path.join(RD, "figures", out_name)
    base = pickle.load(open(os.path.join(RD, base_exp, "averaged_full_output.pickle"), "rb"))
    rec = pickle.load(open(os.path.join(RD, rec_exp, "averaged_full_output.pickle"), "rb"))
    dfc, _ = build_basin_tables(list(rec["valid_ids"]))
    kg = dfc["kg_major_id"].values

    nb, nr = nse_all(base), nse_all(rec)
    present = [g for g in sorted(KG_MAJOR_NAMES) if np.any(kg == g)]

    if row_layout:  # 1x4
        fig, axs = uplt.subplots(nrows=1, ncols=4, figwidth="330mm", share=False, wspace=6,
                                 abc=True, abcloc="l")
    else:           # 2x2
        fig, axs = uplt.subplots(nrows=2, ncols=2, figwidth="185mm", share=False,
                                 hspace=4.5, wspace=4.5, abc=True, abcloc="l")

    for i, (ax, v) in enumerate(zip(axs, VARS)):
        # 1x4: single row so every panel gets the x-label, only the first gets the y-label.
        # 2x2: only the bottom row gets the x-label, only the left column the y-label.
        row, col = (0, i) if row_layout else (i // 2, i % 2)
        keep_x = True if row_layout else (row == 1)
        keep_y = (i == 0) if row_layout else (col == 0)
        xr, yr = np.asarray(nb[v], float), np.asarray(nr[v], float)   # raw NSE (for counting)
        x, y = np.clip(xr, LO, 1), np.clip(yr, LO, 1)                 # clipped (for plotting)
        for g in present:
            m = (kg == g) & np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[m], y[m], s=24, color=KG_MAJOR_COLORS[g], edgecolor="k",
                       linewidth=0.2, alpha=0.85, zorder=3)
        ax.plot([LO, 1], [LO, 1], color="k", ls="--", lw=1.0, zorder=2)  # 1:1
        mm = np.isfinite(xr) & np.isfinite(yr)
        below = int(np.sum(mm & (yr < xr)))
        tot = int(np.sum(mm))
        ax.format(
            title=VAR_TITLE[v], titlesize=TITLESIZE,
            xlabel="Baseline NSE" if keep_x else "",
            ylabel=ylabel if keep_y else "",
            xlim=(LO, 1), ylim=(LO, 1),
            xlabelsize=LABELSIZE, ylabelsize=LABELSIZE,
            xticklabelsize=TICKLABELSIZE - 1, yticklabelsize=TICKLABELSIZE, grid=False,
        )
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2)

    handles = [Line2D([], [], marker="o", linestyle="None", markersize=9,
                      markerfacecolor=KG_MAJOR_COLORS[g], markeredgecolor="k", markeredgewidth=0.2,
                      label=KG_MAJOR_NAMES[g]) for g in present]
    fig.legend(handles, [h.get_label() for h in handles], loc="b",
               ncols=len(present), frame=False, prop={"size": LEGENDSIZE + 3})

    fig.save(OUT, dpi=200)
    print(f"Saved -> {OUT}")
    for v in VARS:
        x = np.asarray(nb[v], float); y = np.asarray(nr[v], float)
        m = np.isfinite(x) & np.isfinite(y)
        print(f"  {v:5s} median dNSE(rec-base)={np.median((y-x)[m]):+.3f}  below 1:1: {np.sum((y<x)[m])}/{m.sum()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rec", default="lstm_af", help="recurrent experiment (y-axis)")
    p.add_argument("--base", default="baseline", help="baseline experiment (x-axis)")
    p.add_argument("--ylabel", default="Recurrent NSE")
    p.add_argument("--out", default="fig_paired_nse_scatter.png")
    p.add_argument("--row", action="store_true", help="1x4 layout instead of 2x2")
    a = p.parse_args()
    main(base_exp=a.base, rec_exp=a.rec, ylabel=a.ylabel, out_name=a.out, row_layout=a.row)
