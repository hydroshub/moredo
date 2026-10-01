"""
Drawing helpers for the history-truncation results (memory_truncation.npz): the data loaders
and the E_s(k) curve panel used by Fig. 8 (fig8_memory_inset.py), and the map of the antecedent
correlation of Fig. S12 (build_map_only).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE,
                        format_map, right_colorbar, EXP_COLORS, CATEGORY_NAMES, CATEGORY_COLORS)
import cartopy.crs as ccrs
from basin_attributes import build_basin_tables, KG_MAJOR_NAMES, KG_MAJOR_COLORS

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")
R_LABEL = r"corr($\Sigma_{L_s}$ (P$-$PET)', $F_{add}$')"


def _load():
    """Load the truncation-experiment arrays (history lengths K, error curves E_sk, effective length L, correlation r, threshold eps, and catchment coordinates) from memory_truncation.npz."""
    d = np.load(os.path.join(OUT_DIR, "memory_truncation.npz"), allow_pickle=True)
    gauge_id = list(d["gauge_id"])
    df_coord, _ = build_basin_tables(gauge_id)
    return (d["k_values"], d["E_sk"], d["L"], d["r"], float(d["eps"]),
            df_coord["basin_lon"].values, df_coord["basin_lat"].values)


def _load_kg():
    """Load the per-catchment Koppen-Geiger major-class ids cached alongside the truncation arrays."""
    d = np.load(os.path.join(OUT_DIR, "memory_truncation.npz"), allow_pickle=True)
    df_coord, _ = build_basin_tables(list(d["gauge_id"]))
    return df_coord["kg_major_id"].values


def _panel_letter(ax, letter):
    # placed just above the top-left corner (outside the axes), matching the abcloc="l" style
    # used by the ultraplot-based manuscript figures.
    """Place a panel letter just above the top-left corner of ax."""
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=LABELSIZE + 1,
            fontweight="bold", va="bottom", ha="left", zorder=10)


def _curve(ax, K, E_sk, eps):
    """Draw the per-catchment truncation-error curves with the ensemble median and the eps threshold."""
    finite = np.isfinite(E_sk).all(axis=1)
    sub = E_sk[finite]
    step = max(1, sub.shape[0] // 400)
    for row in sub[::step]:
        ax.plot(K, row, color="gray", lw=0.3, alpha=0.25)
    ax.plot(K, np.nanmedian(E_sk, axis=0), color=EXP_COLORS["lstm_af"], lw=2.5, marker="o",
            markersize=5, label="median")
    ax.axhline(eps, color="k", ls="--", lw=1.2, label=f"$\\epsilon$ = {eps:.2f}")
    ax.format(xlabel="History length $k$ (days)", ylabel="$E_s(k)$  (RMSE / SD)",
              ylim=(0, min(2.0, np.nanpercentile(E_sk, 98))), xlim=(0, K.max()),
              xlabelsize=LABELSIZE, ylabelsize=LABELSIZE,
              xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    ax.legend(loc="ur", ncols=1, frame=False, prop={"size": LEGENDSIZE - 1})


def _map_r(fig, ax, lon, lat, r, s=20, lonlim=(-180, 180), cbar="right"):
    """Map the per-catchment F_add vs antecedent P-PET correlation r."""
    vr = np.isfinite(r)
    rmax = float(np.nanpercentile(np.abs(r[vr]), 95))
    sc = ax.scatter(lon[vr], lat[vr], c=r[vr], s=s, cmap="RdBu_r", vmin=-rmax, vmax=rmax,
                    edgecolor="grey", linewidth=0.4, transform=ccrs.PlateCarree())
    format_map(ax, title="", lonlim=lonlim)
    if cbar == "bottom":
        cb = ax.colorbar(sc, loc="b", length=0.6, width=0.14, label=R_LABEL,
                         labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE)
    else:
        right_colorbar(fig, ax, sc, R_LABEL)


def build_map_only(K, E_sk, L, r, eps, lon, lat, out_name="fig_si_memory_ac_map.png", cbar="bottom",
                   figwidth="240mm"):
    """Standalone SI figure: only the antecedent-correlation map (panel b of build_ac), no
    companion curve panel and no panel letter."""
    fig = uplt.figure(figwidth=figwidth, tight=True, spanx=False, spany=False,
                      sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())
    _map_r(fig, ax, lon, lat, r, s=24, cbar=cbar)
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=200, bbox_inches="tight"); print(f"Saved -> {out}")

