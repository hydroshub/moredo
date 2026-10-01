"""
Fig. S11: seasonal SHAP predictor importance of the random-forest surrogate for the
Northern-Hemisphere catchment-mean F_add over the 18 predictors of Fig. 6, as a vertical grouped
bar chart: predictors on the x-axis (ordered by JJA+DJF combined importance), relative SHAP
contribution (%) on the y-axis, JJA and DJF side by side. Reads the CV cache
fig_si_s12_seasonal_geo_cv_cache.npz written by shap_seasonal_cache.py.

Usage:
  python scripts/analysis/fig_si_s12_seasonal_shap_vertical.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")
CACHE = os.path.join(OUT_DIR, "fig_si_s12_seasonal_geo_cv_cache.npz")

LEGEND_SHORT = {
    "cover_forest": "Forest Cover", "cover_shrub": "Shrub Cover", "cover_grass": "Grass Cover",
    "cover_crop": "Crop Cover", "cover_others": "Other Cover", "slp_dg_sav": "Slope",
    "new_cly_pc": "Clay Content", "new_snd_pc": "Sand Content", "snw_pc_syr": "Snow Fraction",
    "ari_ix_conv": "Aridity Index", "gwt_cm_sav": "GWT", "pci": "PCI",
    "ele_mt_sav": "Elevation", "lai_mean": "Mean LAI",
    "kar_pc_sse": "Karst extent", "area_shp": "Catchment area",
    "perimeter_geod_km": "Perimeter", "lit_cl_smj": "Lithology class",
}
SEASON_COLOR = {"JJA": "#d7301f", "DJF": "#2b8cbe"}


def main():
    if not os.path.exists(CACHE):
        raise FileNotFoundError(f"{CACHE} not found -- run shap_seasonal_cache.py first")
    cz = np.load(CACHE, allow_pickle=True)
    feats = list(cz["feats"])
    med, q1, q3, r2m, r2s = {}, {}, {}, {}, {}
    for s in ("JJA", "DJF"):
        med[s] = pd.Series(cz[f"{s}_med"], index=feats)
        q1[s] = pd.Series(cz[f"{s}_q1"], index=feats)
        q3[s] = pd.Series(cz[f"{s}_q3"], index=feats)
        r2m[s], r2s[s] = float(cz[f"{s}_r2m"]), float(cz[f"{s}_r2s"])
    print(f"[S12-vert] Loaded CV cache from {CACHE}")

    order = (med["JJA"] + med["DJF"]).sort_values(ascending=False).index
    xp = np.arange(len(order))
    fig = uplt.figure(figwidth="165mm", figheight="105mm", tight=True, spanx=False, spany=False,
                      sharex=False, sharey=False)
    ax = fig.add_subplot()
    for season, off in [("JJA", -0.2), ("DJF", +0.2)]:
        m_ = med[season][order].values
        err = np.vstack([m_ - q1[season][order].values, q3[season][order].values - m_])
        ax.bar(xp + off, height=m_, width=0.4, color=SEASON_COLOR[season],
              edgecolor="k", lw=0.4, label=f"{season} (R$^2$={r2m[season]:.2f})")
        ax.errorbar(xp + off, m_, yerr=err, fmt="none", ecolor="k", elinewidth=0.8, capsize=2, zorder=5)
    ax.format(xticks=xp, xticklabels=[LEGEND_SHORT[f] for f in order], xrotation=90,
              ylabel="Relative SHAP contribution (%)", xlabel="",
              title="", titlesize=TITLESIZE - 1, xlabelsize=LABELSIZE, yticklabelsize=TICKLABELSIZE,
              xticklabelsize=TICKLABELSIZE - 1, grid=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    ax.legend(loc="ur", ncols=1, frame=False, prop={"size": LEGENDSIZE})
    out = os.path.join(OUT_DIR, "fig_si_s12_seasonal_shap_vertical.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
