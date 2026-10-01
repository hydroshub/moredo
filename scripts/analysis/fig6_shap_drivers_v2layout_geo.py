"""
Render the Fig. 6 v2 layout (ranking bar chart panel a + top-5 dependence scatters b-f, one shared
ET colorbar) over the EXTENDED predictor set (base 14 + karst, area, perimeter, lithology), and
save it as fig_6_shap_drivers_v4.png.

This is a pure re-render: it reuses the cached 100x5-fold CV SHAP arrays computed for the extended
surrogate (fig6_shap_ext_cache_geo.npz), so no RF re-fitting is done. The ranking panel therefore
shows all 18 predictors, group-coloured, with two extra groups (Geology, Geometry); the dependence
panels remain the top-5 drivers (unchanged, since the new predictors rank low). The aridity axis is
relabelled PET/P.

Usage:
  python scripts/analysis/fig6_shap_drivers_v2layout_geo.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import LABELSIZE
from manuscript_figures import SHAP_GROUP_COLORS
from run_analysis import plot_shap_dependence_grid

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
FIG = os.path.join(RD, "figures")
OUT = os.path.join(RD, "figures_manuscript_final", "fig_6_shap_drivers_v4.png")
CACHE = os.path.join(FIG, "fig6_shap_ext_cache_geo.npz")

# Extended physical groups (base five + geology + geometry). Every predictor maps to one group.
GROUPS = {
    "Vegetation & LAI": ["cover_forest", "cover_shrub", "cover_grass", "cover_crop",
                         "cover_others", "lai_mean"],
    "Climate & aridity": ["ari_ix_conv", "pci", "snw_pc_syr"],
    "Soil texture": ["new_cly_pc", "new_snd_pc"],
    "Terrain": ["slp_dg_sav", "ele_mt_sav"],
    "Groundwater": ["gwt_cm_sav"],
    "Geology": ["kar_pc_sse", "lit_cl_smj"],
    "Geometry": ["area_shp", "perimeter_geod_km"],
}
GROUP_COLORS = {**SHAP_GROUP_COLORS, "Geology": "#937860", "Geometry": "#CCB974"}


def main():
    cz = np.load(CACHE, allow_pickle=True)
    cols = [str(c) for c in cz["cols"]]
    shap_med, q1, q3 = cz["shap_median"], cz["shap_q1"], cz["shap_q3"]
    rank_stats = (np.asarray(cz["rank_med"], float), np.asarray(cz["rank_q1"], float),
                  np.asarray(cz["rank_q3"], float))
    r2_mean, r2_std = float(cz["r2_mean"]), float(cz["r2_std"])
    X_proc = pd.DataFrame(cz["Xvals"], columns=cols)
    et_color = np.asarray(cz["et_color"], float)

    # long labels from the cache (ari_ix_conv is PET/P, not P/PET); short labels drop the unit suffix
    legend = {str(k): str(v).replace("P/PET", "PET/P")
              for k, v in zip(cz["legend_keys"], cz["legend_vals"])}
    legend_short = {k: re.sub(r"\s*\(.*\)\s*$", "", v).strip() for k, v in legend.items()}
    # abbreviations, explained in the caption (PCI = precipitation concentration index,
    # GWT = groundwater table depth)
    legend["pci"] = "PCI (-)"
    legend_short["pci"] = "PCI"
    legend_short["gwt_cm_sav"] = "GWT"

    feat2grp = {f: g for g, members in GROUPS.items() for f in members}
    bar_group_colors = {f: GROUP_COLORS[feat2grp[f]] for f in cols}
    rank_med = rank_stats[0]
    gtot = {g: sum(rank_med[cols.index(f)] for f in members if f in cols)
            for g, members in GROUPS.items()}
    group_legend = [(g, GROUP_COLORS[g]) for g in sorted(gtot, key=gtot.get, reverse=True)]

    plot_shap_dependence_grid("lstm_af", X_proc, shap_med, r2_mean, legend, legend_short,
                              et_color, "Mean ET (mm day$^{-1}$)", OUT,
                              n_features=5, cbar_vmax=2.4, shap_iqr=(q1, q3),
                              rank_stats=rank_stats, r2_std=None, title="Recurrent",
                              bar_group_colors=bar_group_colors, group_legend=group_legend,
                              title_size=LABELSIZE - 3)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
