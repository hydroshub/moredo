"""
R_in surrogate skill (Sect. 3.4): the repeated cross-validated random forest used for F_add
(100 repeats of 5-fold CV, 200 trees) fitted to the observation-based apparent missing input
R_in = -eps_wb (from fig10_nonclosure_v2_cache.pkl, nonclosure_relation_cache.py) over the same
18 predictors as the F_add surrogate of Fig. 6 (the 14 of manuscript_figures._prepare_shap_features
plus karst extent, catchment area, catchment perimeter and lithology class). Caches the median
relative SHAP importance and the held-out R^2 (mean and std over all test folds) to
rin_surrogate_cv_cache.npz and prints them.

Heavy, run on SLURM:
  sbatch --job-name=moredo-rin slurm/submit_analysis_cpu.sh rin_surrogate_cv.py
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
from manuscript_figures import _prepare_shap_features

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
FIGS = os.path.join(RD, "figures")
GEOMETRY_NPZ = os.path.join(FIGS, "catchment_geometry.npz")   # geodesic perimeter (catchment_geometry.py)
EXTRA = ["kar_pc_sse", "area_shp", "perimeter_geod_km", "lit_cl_smj"]


def cv_shap_importance(X, y, n_iter=100, n_folds=5):
    """100x5-fold CV: per-iteration relative |SHAP| (%) -> median across iterations; and the
    mean/std R^2 across all fold/iteration test sets. Returns (imp_series, r2_mean, r2_std)."""
    Xv, yv = X.values, y.values
    n, p = X.shape
    rel = np.full((n_iter, p), np.nan)
    r2s = []
    for it in range(n_iter):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=it)
        shp = np.full((n, p), np.nan)
        for tr, te in kf.split(Xv):
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xv[tr], yv[tr])
            r2s.append(r2_score(yv[te], rf.predict(Xv[te])))
            shp[te, :] = shap.TreeExplainer(rf).shap_values(Xv[te])
        mabs = np.abs(shp).mean(axis=0)
        rel[it] = mabs / mabs.sum() * 100
        if (it + 1) % 20 == 0:
            print(f"  R_in CV iteration {it + 1}/{n_iter}", flush=True)
    imp = pd.Series(np.median(rel, axis=0), index=X.columns)
    return imp, float(np.mean(r2s)), float(np.std(r2s))


def main():
    """Fit the repeated-CV surrogate to R_in, cache it, and print its held-out R^2 and ranking."""
    panel = pickle.load(open(os.path.join(FIGS, "fig10_nonclosure_v2_cache.pkl"), "rb"))

    # feature matrix (the 18 predictors of Fig. 6) + target aligned by gauge_id
    lstm = pickle.load(open(os.path.join(RD, "lstm_af", "averaged_full_output.pickle"), "rb"))
    _, dff = build_basin_tables(list(lstm["valid_ids"]))
    dff2, feats, _, _ = _prepare_shap_features({"lstm_af": lstm}, dff, "lstm_af")
    geo = dict(np.load(GEOMETRY_NPZ, allow_pickle=True))
    perim_by_id = {str(g): p for g, p in zip(geo["gauge_id"], geo["perimeter_geod_km"].astype(float))}
    dff2 = dff2.copy()
    dff2["perimeter_geod_km"] = [perim_by_id.get(str(g), np.nan) for g in dff2["gauge_id"].values]
    feats = list(feats) + EXTRA
    gid = dff2["gauge_id"].values
    X = dff2.set_index("gauge_id")[feats].copy()

    r_in = -panel["eps_wb"]
    v = panel["v"]
    y_rin = pd.Series(np.where(v, r_in, np.nan), index=gid)

    # ---- R_in CV SHAP (cache) ----
    rin_cache = os.path.join(FIGS, "rin_surrogate_cv_cache.npz")
    if os.path.exists(rin_cache):
        cz = np.load(rin_cache, allow_pickle=True)
        imp_rin = pd.Series(cz["imp"], index=list(cz["feats"]))
        r2_rin, r2_rin_std = float(cz["r2_mean"]), float(cz["r2_std"])
        print(f"Loaded R_in CV cache: R2={r2_rin:.2f}+/-{r2_rin_std:.2f}")
    else:
        data = pd.concat([X, y_rin.rename("t")], axis=1).dropna()
        print(f"R_in CV SHAP on n={len(data)} catchments ...", flush=True)
        imp_rin, r2_rin, r2_rin_std = cv_shap_importance(data[feats], data["t"])
        np.savez(rin_cache, imp=imp_rin.values, feats=np.array(feats, dtype=object),
                 r2_mean=r2_rin, r2_std=r2_rin_std)
        print(f"Cached R_in CV -> {rin_cache}  (R2={r2_rin:.2f}+/-{r2_rin_std:.2f})")

    print("\nR_in CV median relative SHAP contribution (%):")
    for f, val in imp_rin.sort_values(ascending=False).items():
        print(f"  {f:14s} {val:5.1f}")


if __name__ == "__main__":
    main()
