"""
Fig. S8: grouped-SHAP predictor importance (Jiang et al., 2024 aggregation) of the random-forest
surrogate for the 2001-2020 catchment-mean F_add (mean of the 25 model instances, the same target as
Fig. 6), over the 18 predictors of Fig. 6 in seven physically related groups.

This backs the Methods sentence: "Because several terrain and subsurface predictors are
correlated, SHAP values are also aggregated into physically related groups. For each catchment,
the signed SHAP values of predictors belonging to the same group are first summed, after which
the mean absolute grouped contribution is calculated and normalized across groups."

Same RF surrogate configuration and validation as Fig. 6: 200-tree random forest evaluated with
100 repetitions of 5-fold cross-validation (every catchment held out once per repetition). Per
repetition the out-of-fold signed SHAP values are summed within each physically related group and
the normalized mean-absolute grouped contribution is formed; the reported importance is the median
across repetitions (IQR whiskers), and R^2 is the mean +/- 1 std across all fold/repetition test
sets. Results are cached to fig_si_grouped_shap_cv_cache.npz.

Usage:
  python scripts/analysis/fig_si_grouped_shap.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE, EXP_COLORS
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap

from basin_attributes import build_basin_tables
from manuscript_figures import _prepare_shap_features, SHAP_GROUP_COLORS
from shap_fig6_cache import fadd_25_instance_mean

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")
CACHE = os.path.join(OUT_DIR, "fig_si_grouped_shap_cv_cache.npz")
GEOMETRY_NPZ = os.path.join(OUT_DIR, "catchment_geometry.npz")   # geodesic perimeter (catchment_geometry.py)

# The four predictors added to the 14 of _prepare_shap_features, as in Fig. 6.
EXTRA = ["kar_pc_sse", "area_shp", "perimeter_geod_km", "lit_cl_smj"]

# Physically related predictor groups (Jiang-style aggregation), as in Fig. 6. Every one of the
# 18 predictors is assigned to exactly one group.
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
# axis labels where they differ from the group key (the key is part of the CV cache signature)
GROUP_LABELS = {"Geometry": "Catchment geometry"}


def _grouped_importance_cv(dff2, feats, target, mask, n_iter=100, n_folds=5):
    """100x5-fold repeated CV (as in Fig. 6): per repetition build the out-of-fold signed-SHAP
    matrix, sum within each group per catchment, take mean|.| across catchments and normalize
    across groups -> one grouped importance vector per repetition. Returns the per-group median
    (%) with 25/75 percentiles across repetitions, and the mean +/- std held-out R^2."""
    feats = list(feats)
    X = dff2.set_index("gauge_id")[feats].copy()
    y = pd.Series(np.where(mask, target, np.nan), index=dff2["gauge_id"].values, name="t")
    data = pd.concat([X, y], axis=1).dropna()
    Xv, yv = data[feats].values, data["t"].values
    n, p = Xv.shape
    gnames = list(GROUPS.keys())
    gcols = {g: [feats.index(m) for m in GROUPS[g] if m in feats] for g in gnames}

    if os.path.exists(CACHE):
        cz = np.load(CACHE, allow_pickle=True)
        if int(cz["n_iter"]) == n_iter and int(cz["n"]) == n and list(cz["gnames"]) == gnames:
            print(f"[grouped] loaded CV cache {CACHE}")
            return (pd.Series(cz["med"], index=gnames), pd.Series(cz["q1"], index=gnames),
                    pd.Series(cz["q3"], index=gnames), float(cz["r2_mean"]), float(cz["r2_std"]), n)

    rel = np.full((n_iter, len(gnames)), np.nan)
    r2_list = []
    for it in range(n_iter):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=it)
        shap_store = np.full((n, p), np.nan)   # out-of-fold signed SHAP, all catchments
        for tr, te in kf.split(Xv):
            rf = RandomForestRegressor(n_estimators=200, random_state=42,
                                       n_jobs=int(os.environ.get("RF_NJOBS", "4"))).fit(Xv[tr], yv[tr])
            r2_list.append(r2_score(yv[te], rf.predict(Xv[te])))
            shap_store[te, :] = shap.TreeExplainer(rf).shap_values(Xv[te])
        grouped = np.array([np.abs(shap_store[:, gcols[g]].sum(1)).mean() for g in gnames])
        rel[it] = grouped / grouped.sum() * 100.0
        if (it + 1) % 20 == 0:
            print(f"  grouped CV iteration {it + 1}/{n_iter}", flush=True)

    med = pd.Series(np.median(rel, axis=0), index=gnames)
    q1 = pd.Series(np.percentile(rel, 25, axis=0), index=gnames)
    q3 = pd.Series(np.percentile(rel, 75, axis=0), index=gnames)
    r2m, r2s = float(np.mean(r2_list)), float(np.std(r2_list))
    np.savez(CACHE, gnames=np.array(gnames), med=med.values, q1=q1.values, q3=q3.values,
             r2_mean=r2m, r2_std=r2s, n_iter=n_iter, n=n)
    print(f"[grouped] cached CV -> {CACHE}")
    return med, q1, q3, r2m, r2s, n


def main():
    """Render Figure S8: SHAP importance aggregated into physically related predictor groups."""
    import pickle
    lstm = pickle.load(open(os.path.join(REPO, "results_lateral_experiments/lstm_af/averaged_full_output.pickle"), "rb"))
    vids = list(lstm["valid_ids"])
    dfc, dff = build_basin_tables(vids)
    dff2, feats, _, _ = _prepare_shap_features({"lstm_af": lstm}, dff, "lstm_af")
    # extend to the 18 predictors of Fig. 6: karst, area and lithology are already in dff2, the
    # geodesic perimeter is joined from catchment_geometry.npz
    geo = dict(np.load(GEOMETRY_NPZ, allow_pickle=True))
    perim_by_id = {str(g): p for g, p in zip(geo["gauge_id"], geo["perimeter_geod_km"].astype(float))}
    dff2 = dff2.copy()
    dff2["perimeter_geod_km"] = [perim_by_id.get(str(g), np.nan) for g in dff2["gauge_id"].values]
    feats = list(feats) + EXTRA

    fadd = fadd_25_instance_mean(vids)   # same target as Fig. 6

    med, q1, q3, r2m, r2s, n_f = _grouped_importance_cv(dff2, feats, fadd, np.ones(len(fadd), bool))

    print("=== Grouped SHAP (normalized %, 100x5-fold CV median) ===")
    order = med.sort_values().index
    for g in order[::-1]:
        print(f"  {g:20s}  F_add={med[g]:5.1f}%  (IQR {q1[g]:.1f}-{q3[g]:.1f})")
    print(f"  surrogate held-out R^2:  F_add={r2m:.2f} +/- {r2s:.2f} (n={n_f})")

    yp = np.arange(len(order))
    fig = uplt.figure(figwidth="150mm", figheight="85mm", tight=True, spanx=False, spany=False,
                      sharex=False, sharey=False)
    ax = fig.add_subplot()
    # One bar per group, coloured by physically related predictor group (as in Fig. 6), with
    # IQR whiskers across the 100 CV repetitions.
    for j, g in enumerate(order):
        ax.barh(yp[j], height=med[g], width=0.8, color=GROUP_COLORS[g], edgecolor="k", lw=0.4)
    err = np.vstack([med[order].values - q1[order].values, q3[order].values - med[order].values])
    ax.errorbar(med[order].values, yp, xerr=err, fmt="none", ecolor="k", elinewidth=0.8,
                capsize=2, zorder=5)
    ax.format(yticks=yp, yticklabels=[GROUP_LABELS.get(g, g) for g in order],
              xlabel="Grouped SHAP contribution (%)", ylabel="",
              title="",
              titlesize=TITLESIZE - 1, xlabelsize=LABELSIZE, yticklabelsize=TICKLABELSIZE,
              xticklabelsize=TICKLABELSIZE, grid=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    out = os.path.join(OUT_DIR, "fig_si_grouped_shap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
