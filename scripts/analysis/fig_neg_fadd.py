"""
Signed-subsidy figure (lstm_af_neg vs the default non-negative lstm_af), in the same inset layout as Figure 1
(fig1_nonclosure_inset.py): the world map fills the figure as panel (a), a thin vertical colorbar
sits as an inset in the upper-left, and panel (b) is a small inset in the lower-left.
  (a) map of the signed catchment-mean F_add (diverging, 0-centred): where the flux is a net sink.
  (b) held-out per-catchment NSE (ET, Q, SWE, TWSA) default vs signed, as an inset panel.
Reads neg_fadd_compare.npz. Usage: python scripts/analysis/fig_neg_fadd.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, sns, LABELSIZE, TICKLABELSIZE, LEGENDSIZE, format_map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from project_paths import RESULTS_DIR as RD
NPZ = os.path.join(RD, "figures", "neg_fadd_compare.npz")
OUT = os.path.join(RD, "figures", "fig_neg_fadd.png")

VARS = ["ET", "Q", "SWE", "TWSA"]
RUNS = [("base", "Default ($F_{add}\\geq0$)", "#9aa0a6"),
        ("neg", "Signed ($F_{add}$ free)", "#4C72B0")]


def main():
    c = dict(np.load(NPZ, allow_pickle=True))
    mean_n = c["mean_neg"].astype(float)
    lon, lat = c["lon"].astype(float), c["lat"].astype(float)
    mp = np.isfinite(mean_n) & np.isfinite(lon) & np.isfinite(lat)
    vlim = float(np.nanpercentile(np.abs(mean_n[mp]), 98))
    cmap = "RdBu_r"

    # --- fixed layout (matches fig1_nonclosure_inset / fig5_spatial_fadd_inset) ---
    fig = uplt.figure(figwidth="250mm", figheight="112mm", tight=False,
                      spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())

    # --- a: world map of signed mean F_add ---
    ax.add_feature(cfeature.LAND, facecolor="#ebebeb", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    order = np.argsort(np.abs(mean_n[mp]))          # strong values on top
    sc = ax.scatter(lon[mp][order], lat[mp][order], c=mean_n[mp][order], s=14,
                    cmap=cmap, vmin=-vlim, vmax=vlim, edgecolor="0.35", linewidth=0.2,
                    transform=ccrs.PlateCarree(), zorder=2)
    format_map(ax, title="")
    ax.set_position([0.12, 0.11, 0.88, 0.88])
    fig.text(0.012, 0.975, "a", fontsize=LABELSIZE, fontweight="bold", va="top", ha="left", zorder=6)

    # --- colorbar: thin vertical, top of the left column (signed F_add, diverging) ---
    cax = fig.add_axes([0.10, 0.62, 0.013, 0.30])
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(r"Mean $F_{add}$ (mm day$^{-1}$)", size=LABELSIZE - 4)
    cb.ax.tick_params(labelsize=TICKLABELSIZE - 4)
    cb.outline.set_linewidth(0.6)
    # end labels: red (top) = net gain, blue (bottom) = net loss
    fig.text(0.1065, 0.925, "net\ngain", fontsize=TICKLABELSIZE - 4, color="#b2182b",
             va="bottom", ha="center", linespacing=0.9)
    fig.text(0.1065, 0.608, "net\nloss", fontsize=TICKLABELSIZE - 4, color="#2166ac",
             va="top", ha="center", linespacing=0.9)

    # --- b: held-out skill default vs signed, as an inset (lower-left, over the South Pacific) ---
    bx = fig.add_axes([0.075, 0.16, 0.24, 0.36])
    recs = []
    for var in VARS:
        for exp, lab, _ in RUNS:
            a = c[f"nse_{var}_{exp}"].astype(float)
            a = a[np.isfinite(a)]
            recs += [{"var": var, "run": lab, "nse": x} for x in a]
    df = pd.DataFrame(recs)
    pal = {lab: col for _, lab, col in RUNS}
    order_runs = [lab for _, lab, _ in RUNS]
    sns.boxplot(data=df, x="var", y="nse", hue="run", order=VARS, hue_order=order_runs,
                palette=pal, legend=False, width=0.7, linewidth=0.8, showfliers=False, ax=bx)
    bx.axhline(0, color="0.4", ls="--", lw=0.9, alpha=0.7, zorder=0)
    bx.set_ylim(-4.8, 1.2)
    bx.set_yticks([-4, -3, -2, -1, 0, 1])
    bx.set_xlabel(""); bx.set_ylabel("Per-catchment NSE (held-out)", size=LABELSIZE - 4)
    bx.tick_params(axis="both", labelsize=TICKLABELSIZE - 4, length=2)
    for sp in bx.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    bx.patch.set_facecolor("white"); bx.patch.set_alpha(0.9)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=col, edgecolor="0.3", label=lab) for _, lab, col in RUNS]
    bx.legend(handles=handles, loc="lower right", ncols=1, frameon=True, framealpha=0.9,
              edgecolor="none", prop={"size": LEGENDSIZE - 5}, handlelength=1.1,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.3)
    fig.text(0.012, 0.50, "b", fontsize=LABELSIZE, fontweight="bold", va="bottom", ha="left", zorder=6)

    fig.savefig(OUT, dpi=200)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
