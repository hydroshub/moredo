"""
Shared figure code: the Fig. 5 map of the 2001-2020 catchment-mean F_add with the
per-Koppen-Geiger-class boxplot inset (fig5_spatial_fadd_inset), and the 14 base predictors of the
random-forest surrogates with their labels (_prepare_shap_features).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (uplt, sns, CATEGORY_NAMES, CATEGORY_COLORS, EXP_LABELS, EXP_COLORS,
                        LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE, format_map, right_colorbar)
import cartopy.crs as ccrs

from residual_utils import ACTIVE_START_YEAR, _daily_series, _reconstruct_precip, compute_residual_vs_fadd
from metrics import basin_nse_table, basin_nse_twsa
from run_analysis import load_results, EXPERIMENTS
from basin_attributes import build_basin_tables
from project_paths import RESULTS_DIR

from project_paths import RESULTS_DIR
OUT_DIR_DEFAULT = os.path.join(RESULTS_DIR, "figures")


# =============================================================================
# Figure 4: improvement by hydroclimatic regime (Delta-NSE, LSTM(A,F) - MLP(A), by aridity class)
# =============================================================================
# =============================================================================
# Figure 5: spatial pattern of mean Fadd -- map, distribution by aridity class, dry-season map
# (single representative experiment, LSTM(A,F), per the earlier decision that a 3-map panel
# has no place/novelty in the paper version of this figure)
# =============================================================================
def fig5_spatial_fadd_inset(results, df_coord, out_path, exp="lstm_af", fadd_mean=None):
    """Compact variant of Fig. 5: the ACTIVE_START_YEAR-2020 mean-Fadd world map fills the
    figure, with the Fadd colorbar as a thin vertical inset in the upper-left and the per-KG-class
    boxplot as a small inset in the lower-left (over the South Pacific). Koppen-Geiger background
    legend runs across the bottom. fadd_mean, if given, is a precomputed catchment-mean F_add
    aligned with df_coord (the manuscript passes the mean of the 25 model instances); otherwise it
    is computed from results[exp]."""
    from basin_attributes import KG_MAJOR_NAMES, KG_MAJOR_COLORS, KG_MAJOR, load_kg_major_grid
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    if fadd_mean is None:
        idx_daily = pd.DatetimeIndex(results[exp]["idx_daily"])
        active = idx_daily.year >= ACTIVE_START_YEAR
        fadd_mean = np.squeeze(results[exp]["fluxes"]["f_lat"])[:, active].mean(axis=1)

    cat = df_coord["kg_major_id"].values
    ordered_ids = [c for c in sorted(KG_MAJOR_NAMES.keys()) if np.any(cat == c)]
    order_labels = [KG_MAJOR_NAMES[c] for c in ordered_ids]
    palette = {KG_MAJOR_NAMES[c]: KG_MAJOR_COLORS[c] for c in ordered_ids}

    # Fixed layout (no tight engine): a dedicated left column holds the colorbar and boxplot, the
    # map fills the remaining right ~82% at ~PlateCarree aspect, legend runs along the bottom.
    fig = uplt.figure(figwidth="250mm", figheight="112mm", tight=False,
                      spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())

    # --- map: Koppen-Geiger background + mean-Fadd scatter ---
    kg_lon, kg_lat, kg_grid = load_kg_major_grid()
    kg_ids = list(KG_MAJOR.keys())
    kg_cmap = ListedColormap([KG_MAJOR[k]['color'] for k in kg_ids])
    kg_norm = BoundaryNorm(boundaries=np.arange(0.5, len(kg_ids) + 1.5, 1), ncolors=len(kg_ids))
    ax.pcolormesh(kg_lon, kg_lat, kg_grid, cmap=kg_cmap, norm=kg_norm, alpha=0.35,
                  transform=ccrs.PlateCarree(), zorder=0)
    sc = ax.scatter(df_coord["basin_lon"], df_coord["basin_lat"], c=fadd_mean, s=14, cmap="Blues",
                    vmin=0, vmax=1, zorder=2, edgecolor="grey", linewidth=0.2,
                    transform=ccrs.PlateCarree())
    format_map(ax, title="")
    ax.set_position([0.12, 0.11, 0.88, 0.88])  # small left column; map fills the rest
    fig.text(0.012, 0.975, "a", fontsize=LABELSIZE, fontweight="bold",
             va="top", ha="left", zorder=6)

    # --- colorbar: thin vertical, top of the left column (numbers only) ---
    cax = fig.add_axes([0.10, 0.65, 0.013, 0.28])
    cb = fig.colorbar(sc, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("Mean $F_{add}$ (mm day$^{-1}$)", size=LABELSIZE - 4)
    cb.ax.tick_params(labelsize=TICKLABELSIZE - 4)
    cb.outline.set_linewidth(0.6)

    # --- boxplot: wide, spanning the left column and into the map's lower-left ocean ---
    bx = fig.add_axes([0.075, 0.16, 0.24, 0.36])
    rows = [{"Cat": KG_MAJOR_NAMES[c], "fadd": v} for c in ordered_ids
            for v in fadd_mean[cat == c] if np.isfinite(v)]
    df_box = pd.DataFrame(rows)
    sns.boxplot(data=df_box, x="Cat", y="fadd", order=order_labels, hue="Cat", palette=palette,
                hue_order=order_labels, legend=False, width=0.62, linewidth=0.7, fliersize=1.0,
                flierprops=dict(marker="o", markersize=1.0, markerfacecolor="0.4",
                                markeredgecolor="none", alpha=0.4), ax=bx)
    bx.set_xlabel("")
    bx.set_ylabel("Mean $F_{add}$ (mm day$^{-1}$)", size=LABELSIZE - 4)
    bx.tick_params(axis="both", labelsize=TICKLABELSIZE - 4)
    bx.tick_params(axis="x", rotation=0, length=2)
    for sp in bx.spines.values():
        sp.set_linewidth(0.8)
    bx.patch.set_facecolor("white")
    bx.patch.set_alpha(0.9)
    fig.text(0.012, 0.585, "b", fontsize=LABELSIZE, fontweight="bold", va="bottom", ha="left",
             zorder=6)

    # --- Koppen-Geiger background legend across the bottom ---
    patches = [Patch(facecolor=KG_MAJOR[k]['color'], edgecolor='none', alpha=0.6,
                     label=KG_MAJOR[k]['code']) for k in kg_ids]
    from matplotlib.figure import Figure as _MplFig
    leg = _MplFig.legend(fig, patches, [p.get_label() for p in patches], loc="lower center",
                         bbox_to_anchor=(0.5, 0.005), ncol=len(kg_ids), frameon=False,
                         fontsize=LEGENDSIZE - 2, title="Köppen–Geiger major classes")
    leg.get_title().set_fontsize(LEGENDSIZE - 2)

    fig.savefig(out_path, dpi=200)
    print(f"Saved -> {out_path}")


# =============================================================================
# Figure 6: SHAP ranking + dependence plots for mean Fadd (adds mean LAI to the static
# predictor set used elsewhere in the pipeline, per the manuscript's example driver list)
# =============================================================================
def _prepare_shap_features(results, df_features, exp):
    """Shared feature prep for the mean-Fadd SHAP figures (Fig 6 and its seasonal variant):
    slope/aridity-index unit corrections (see run_analysis.run_shap_analysis), PCI, and mean
    LAI. Returns (df_features, selected_features, legend, legend_short)."""
    from run_analysis import _compute_pci

    df_features = df_features.copy()
    df_features["slp_dg_sav"] = df_features["slp_dg_sav"] / 10.0  # HydroATLAS stores slope *10
    df_features["pci"] = _compute_pci(results[exp])
    df_features["lai_mean"] = np.nanmean(_daily_series(results[exp], "LAI_normed", group="diagnos"), axis=1)

    selected_features = [
        "cover_forest", "cover_shrub", "cover_grass", "cover_crop", "cover_others",
        "slp_dg_sav", "new_cly_pc", "new_snd_pc", "snw_pc_syr",
        "ari_ix_conv", "gwt_cm_sav", "pci", "ele_mt_sav", "lai_mean",
    ]
    legend = {
        "cover_forest": "Forest Cover (%)", "cover_shrub": "Shrub Cover (%)",
        "cover_grass": "Grass Cover (%)", "cover_crop": "Crop Cover (%)",
        "cover_others": "Other Cover (%)", "slp_dg_sav": "Slope (deg)",
        "new_cly_pc": "Clay Content (%)", "new_snd_pc": "Sand Content (%)",
        "snw_pc_syr": "Snow Fraction (%)", "ari_ix_conv": "Aridity Index (PET/P, -)",
        "gwt_cm_sav": "GWT (cm)", "pci": "Precip. Concentr. Index (-)", "ele_mt_sav": "Elevation (m)",
        "lai_mean": "Mean LAI (-)",
    }
    legend_short = {
        "cover_forest": "Forest Cover", "cover_shrub": "Shrub Cover",
        "cover_grass": "Grass Cover", "cover_crop": "Crop Cover",
        "cover_others": "Other Cover", "slp_dg_sav": "Slope",
        "new_cly_pc": "Clay Content", "new_snd_pc": "Sand Content",
        "snw_pc_syr": "Snow Fraction", "ari_ix_conv": "Aridity Index",
        "gwt_cm_sav": "GWT", "pci": "PCI", "ele_mt_sav": "Elevation", "lai_mean": "Mean LAI",
    }
    return df_features, selected_features, legend, legend_short


# Physically related predictor groups (Jiang-style aggregation), matching fig_si_grouped_shap.py.
SHAP_GROUPS = {
    "Vegetation & LAI": ["cover_forest", "cover_shrub", "cover_grass", "cover_crop",
                          "cover_others", "lai_mean"],
    "Climate & aridity": ["ari_ix_conv", "pci", "snw_pc_syr"],
    "Soil texture": ["new_cly_pc", "new_snd_pc"],
    "Terrain": ["slp_dg_sav", "ele_mt_sav"],
    "Groundwater": ["gwt_cm_sav"],
}
SHAP_GROUP_COLORS = {
    "Climate & aridity": "#4C72B0", "Terrain": "#DD8452", "Soil texture": "#C44E52",
    "Vegetation & LAI": "#55A868", "Groundwater": "#8172B3",
}


# =============================================================================
# Figure 6b: same SHAP analysis, split into summer (JJA) vs winter (DJF), Northern
# Hemisphere basins only (per Georgios' comment: is the ranking similar or not by season?)
# =============================================================================
# =============================================================================
# Figure 8: monthly time series of Fadd(t), P(t), ET(t) in representative basins
# (one basin per aridity class shown -- the basin with the highest mean Fadd within that
# class, as the most illustrative example of the signal)
# =============================================================================
# =============================================================================
# Figure 9: lag correlation between Fadd(t) and antecedent (P-PET) deficit
# =============================================================================
# =============================================================================
# Figure 10: Fadd vs long-term water-balance non-closure eps_wb
# =============================================================================
# =============================================================================
# Figure 11: eta_dry = sum(Fadd)/sum(ET) over each basin's climatological dry season
# =============================================================================