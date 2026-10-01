"""
Fig. S11 input: seasonal SHAP predictor importance of the random-forest surrogate for the
Northern-Hemisphere catchment-mean F_add in summer (JJA) and winter (DJF), over the same 18
predictors as Fig. 6. 100 repeats of 5-fold CV per season. Caches the median and IQR relative
importance and the held-out R^2 per season to fig_si_s12_seasonal_geo_cv_cache.npz (rendered by
fig_si_s12_seasonal_shap_vertical.py) and prints the three leading predictors per season.

Heavy, run on SLURM:
  sbatch --job-name=moredo-s11 slurm/submit_analysis_cpu.sh shap_seasonal_cache.py
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap

from basin_attributes import build_basin_tables
from residual_utils import ACTIVE_START_YEAR, _daily_series
from manuscript_figures import _prepare_shap_features

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")
CACHE = os.path.join(OUT_DIR, "fig_si_s12_seasonal_geo_cv_cache.npz")
MORPH_NPZ = os.path.join(OUT_DIR, "catchment_geometry.npz")   # geodesic perimeter (catchment_geometry.py)

EXTRA = ["kar_pc_sse", "area_shp", "perimeter_geod_km", "lit_cl_smj"]
LEGEND_SHORT = {
    "cover_forest": "Forest Cover", "cover_shrub": "Shrub Cover", "cover_grass": "Grass Cover",
    "cover_crop": "Crop Cover", "cover_others": "Other Cover", "slp_dg_sav": "Slope",
    "new_cly_pc": "Clay Content", "new_snd_pc": "Sand Content", "snw_pc_syr": "Snow Fraction",
    "ari_ix_conv": "Aridity Index", "gwt_cm_sav": "GWT", "pci": "PCI",
    "ele_mt_sav": "Elevation", "lai_mean": "Mean LAI",
    "kar_pc_sse": "Karst extent", "area_shp": "Catchment area",
    "perimeter_geod_km": "Perimeter", "lit_cl_smj": "Lithology class",
}
RF_NJOBS = int(os.environ.get("RF_NJOBS", "4"))


def _season_importance(results, dff, feats, exp, months, nh_mask, n_iter=100, n_folds=5):
    idx = pd.DatetimeIndex(results[exp]["idx_daily"]); active = idx.year >= ACTIVE_START_YEAR
    fadd = _daily_series(results[exp], "f_lat")[:, active]
    m = np.isin(idx[active].month, months)
    season_mean = np.nanmean(fadd[:, m], axis=1)
    df_nh = dff.loc[nh_mask].copy()
    X = df_nh.set_index("gauge_id")[feats].copy()
    y = pd.Series(season_mean[nh_mask], index=df_nh["gauge_id"].values, name="t")
    data = pd.concat([X, y], axis=1).dropna()
    Xv, yv = data[feats].values, data["t"].values
    n, p = Xv.shape

    rel_per_iter = np.full((n_iter, p), np.nan)
    r2_list = []
    for it in range(n_iter):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=it)
        shap_store = np.full((n, p), np.nan)
        for tr, te in kf.split(Xv):
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=RF_NJOBS).fit(Xv[tr], yv[tr])
            r2_list.append(r2_score(yv[te], rf.predict(Xv[te])))
            shap_store[te, :] = shap.TreeExplainer(rf).shap_values(Xv[te])
        mabs = np.abs(shap_store).mean(axis=0)
        rel_per_iter[it] = mabs / mabs.sum() * 100
        if (it + 1) % 20 == 0:
            print(f"    CV iteration {it + 1}/{n_iter}", flush=True)

    med = pd.Series(np.median(rel_per_iter, axis=0), index=feats)
    q1 = pd.Series(np.percentile(rel_per_iter, 25, axis=0), index=feats)
    q3 = pd.Series(np.percentile(rel_per_iter, 75, axis=0), index=feats)
    return med, q1, q3, float(np.mean(r2_list)), float(np.std(r2_list))


def main(n_iter=100, n_folds=5):
    lstm = pickle.load(open(os.path.join(REPO, "results_lateral_experiments/lstm_af/averaged_full_output.pickle"), "rb"))
    vids = list(lstm["valid_ids"])
    dfc, dff = build_basin_tables(vids)
    results = {"lstm_af": lstm}
    dff2, feats, _, _ = _prepare_shap_features(results, dff, "lstm_af")
    # join geodesic perimeter; karst/area/lithology already present
    mm = dict(np.load(MORPH_NPZ, allow_pickle=True))
    perim_by_id = {str(g): p for g, p in zip(mm["gauge_id"], mm["perimeter_geod_km"].astype(float))}
    dff2 = dff2.copy()
    dff2["perimeter_geod_km"] = [perim_by_id.get(str(g), np.nan) for g in dff2["gauge_id"].values]
    feats = list(feats) + EXTRA
    nh = dfc["basin_lat"].values > 0

    med = {}; q1 = {}; q3 = {}; r2m = {}; r2s = {}
    if os.path.exists(CACHE):
        cz = np.load(CACHE, allow_pickle=True)
        if int(cz["n_iter"]) == n_iter and list(cz["feats"]) == list(feats):
            print(f"[S11] Loaded CV cache from {CACHE}")
            for s in ("JJA", "DJF"):
                med[s] = pd.Series(cz[f"{s}_med"], index=feats)
                q1[s] = pd.Series(cz[f"{s}_q1"], index=feats)
                q3[s] = pd.Series(cz[f"{s}_q3"], index=feats)
                r2m[s], r2s[s] = float(cz[f"{s}_r2m"]), float(cz[f"{s}_r2s"])
    if not med:
        for season, mlist in [("JJA", [6, 7, 8]), ("DJF", [12, 1, 2])]:
            print(f"[S11] {season} (NH) CV surrogate ...", flush=True)
            med[season], q1[season], q3[season], r2m[season], r2s[season] = \
                _season_importance(results, dff2, feats, "lstm_af", mlist, nh, n_iter, n_folds)
        np.savez(CACHE, feats=np.array(list(feats)), n_iter=n_iter,
                 **{f"{s}_med": med[s].values for s in med},
                 **{f"{s}_q1": q1[s].values for s in q1},
                 **{f"{s}_q3": q3[s].values for s in q3},
                 **{f"{s}_r2m": r2m[s] for s in r2m}, **{f"{s}_r2s": r2s[s] for s in r2s})
        print(f"[S11] Cached CV arrays -> {CACHE}")

    for season in ("JJA", "DJF"):
        top = med[season].sort_values(ascending=False)
        print(f"[S11] {season} (NH) R2={r2m[season]:.2f}+/-{r2s[season]:.2f}: " +
              ", ".join(f"{LEGEND_SHORT[k]} {v:.0f}%" for k, v in top.head(3).items()))


if __name__ == "__main__":
    main()
