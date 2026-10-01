"""
Figure 8: effective input-history dependence of the recurrent subsidy. (a) Normalized truncation
error E_s(k) against the retained history length k (catchments in grey, median in blue, the
epsilon = 0.10 threshold dashed). (b) Map of the effective history length L_s, catchments that never
cross the threshold in grey, with an inset bar plot of the mean L_s per major Koppen-Geiger class.
Reuses the drawing helpers of memory_truncation_plot_v2 and reads memory_truncation.npz.

Usage:
  python scripts/analysis/fig8_memory_inset.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE
import cartopy.crs as ccrs
from basin_attributes import KG_MAJOR_NAMES, KG_MAJOR_COLORS
import memory_truncation_plot_v2 as M

OUT_DIR = M.OUT_DIR


def _inset_bar_kg(ax_b, L, kg):
    """Draw an inset bar plot of the mean effective memory L_s per major Koppen-Geiger class.

    Styled to match the study-domain histogram inset in fig_si_04_catchment_climate_map.png: compact,
    bottom-left over open ocean, open frame (bottom spine only), no y-axis, and the per-bar value
    printed vertically on top. Bars are ordered by increasing mean L_s so the climate gradient reads
    left to right, coloured by the same KG palette as the rest of the manuscript. Mean is used rather
    than median because L_s is quantised to the truncation grid, which collapses most class medians
    onto a single value and hides the between-climate gradient."""
    order = [g for g in sorted(KG_MAJOR_NAMES) if np.any((kg == g) & np.isfinite(L))]
    mu = {g: float(np.nanmean(L[(kg == g) & np.isfinite(L)])) for g in order}
    order = sorted(order, key=lambda g: mu[g], reverse=True)         # descending: large left, small right

    ins = ax_b.inset_axes([0.0, 0.06, 0.23, 0.36], transform=ax_b.transAxes, proj="cartesian")
    pos = np.arange(len(order))
    for p, g in zip(pos, order):
        ins.bar(p, mu[g], width=1.0, color=KG_MAJOR_COLORS[g], edgecolor="k", linewidth=0.4, zorder=3)
    ymax = max(mu[g] for g in order)
    off = ymax * 0.05
    for p, g in zip(pos, order):
        ins.text(p, mu[g] + off, f"{mu[g]:.0f}", ha="center", va="bottom",
                 fontsize=12.0, rotation=90, zorder=4)
    ins.set_xticks(list(pos))
    ins.set_xticklabels([KG_MAJOR_NAMES[g] for g in order], fontsize=11.5)
    ins.set_xlim(-0.5, len(order) - 0.5)
    ins.set_ylim(0, ymax * 1.48)
    ins.set_yticks([])
    ins.set_ylabel("mean $L_s$ (d)", fontsize=11, labelpad=3)
    ins.tick_params(axis="x", length=0, pad=1)
    ins.grid(False)
    for sp in ins.spines.values():                                  # full frame
        sp.set_visible(True); sp.set_linewidth(0.9)
    ins.patch.set_facecolor("white"); ins.patch.set_alpha(0.85)


def build(K, E_sk, L, eps, lon, lat, kg, out_name="fig8_memory_k365_curve_Ls_inset.png"):
    """Build the two-panel Fig. 8 with the KG median-L_s inset on the map panel."""
    kmax = int(np.nanmax(K))
    fig = uplt.figure(figwidth="300mm", tight=True, spanx=False, spany=False,
                      sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=1, ncols=2, wspace=1.5, wratios=(1, 2.4))

    ax_a = fig.add_subplot(gs[0, 0])
    M._curve(ax_a, K, E_sk, eps); M._panel_letter(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1], proj=ccrs.PlateCarree())
    crossed = np.isfinite(L); never = ~crossed
    ax_b.scatter(lon[never], lat[never], s=24, color="0.6", edgecolor="k", linewidth=0.3,
                 transform=ccrs.PlateCarree(), zorder=1,
                 label=f"$>${kmax} d\n(no crossing, n={int(never.sum())})")
    sc = ax_b.scatter(lon[crossed], lat[crossed], c=L[crossed], s=24, cmap="viridis",
                      vmin=0, vmax=kmax, edgecolor="grey", linewidth=0.3,
                      transform=ccrs.PlateCarree(), zorder=2)
    M.format_map(ax_b, title="")
    ax_b.colorbar(sc, loc="b", length=0.48, width=0.14, label="$L_s$ (days)",
                  labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE, ticks=[0, 90, 180, 270, 365])
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.0, -0.06), ncols=1, frame=False,
                borderaxespad=0.0, prop={"size": LEGENDSIZE - 1})
    _inset_bar_kg(ax_b, L, kg)
    M._panel_letter(ax_b, "b")

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=200, bbox_inches="tight"); print(f"Saved -> {out}")


def main():
    """Load the cached truncation arrays and render the KG-inset Fig. 8 variant."""
    K, E_sk, L, r, eps, lon, lat = M._load()
    kg = M._load_kg()
    build(K, E_sk, L, eps, lon, lat, kg)


if __name__ == "__main__":
    main()
