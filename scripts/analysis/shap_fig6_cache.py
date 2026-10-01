"""
Fig. 6 input: repeated cross-validated random-forest surrogate of the 2001-2020 catchment-mean F_add
(recurrent representation, mean of the 25 model instances as in Fig. 5, see fadd_25_instance_mean;
the strictly held-out counterpart is Fig. S9) over the 18 catchment predictors, i.e. the 14 of
manuscript_figures._prepare_shap_features plus karst extent, catchment area, catchment perimeter
and lithology class. 100 repeats of 5-fold CV with a 200-tree forest; each catchment is held out
once per repeat and its SHAP values come from that held-out fold. Caches the per-catchment median
and IQR SHAP values, the relative-importance ranking and the held-out R^2 (mean and std over all
test folds) to fig6_shap_ext_cache_geo.npz, rendered as Fig. 6 by fig6_shap_drivers_v2layout_geo.py.

Heavy (500 forest fits plus TreeSHAP), run on SLURM:
  sbatch --job-name=moredo-fig6 slurm/submit_analysis_cpu.sh shap_fig6_cache.py
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basin_attributes import build_basin_tables
from manuscript_figures import _prepare_shap_features
from residual_utils import ACTIVE_START_YEAR

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
FIG = os.path.join(RD, "figures")
CACHE = os.path.join(FIG, "fig6_shap_ext_cache_geo.npz")
MORPH_NPZ = os.path.join(FIG, "catchment_geometry.npz")   # geodesic perimeter (catchment_geometry.py)
STABILITY_NPZ = os.path.join(FIG, "fig_si_stability.npz")  # per-instance F_add maps (fig_si_stability.py)
RF_NJOBS = int(os.environ.get("RF_NJOBS", "-1"))

# The four predictors added to the base 14: karst and area are already in df_features, the geodesic
# perimeter is joined from catchment_geometry.npz, lithology is the HydroATLAS majority class.
EXTRA = ["kar_pc_sse", "area_shp", "perimeter_geod_km", "lit_cl_smj"]
EXTRA_LEGEND = {"kar_pc_sse": "Karst extent (%)", "area_shp": "Catchment area (km$^2$)",
                "perimeter_geod_km": "Perimeter (km)", "lit_cl_smj": "Lithology class (-)"}
EXTRA_SHORT = {"kar_pc_sse": "Karst", "area_shp": "Area",
               "perimeter_geod_km": "Perimeter", "lit_cl_smj": "Lithology"}


def fadd_25_instance_mean(ids):
    """2001-2020 catchment-mean F_add of the recurrent representation averaged over its 25 model
    instances (every catchment in every instance), as mapped in Fig. 5, aligned to `ids`."""
    st = np.load(STABILITY_NPZ, allow_pickle=True)
    by_id = dict(zip([str(g) for g in st["gauge_id"]], np.nanmean(st["inst_maps"], axis=0)))
    return np.array([by_id.get(str(g), np.nan) for g in ids], dtype=float)


def _prep(exp="lstm_af"):
    """Build the extended dependence-plot inputs (feature matrix, mean-ET colour, target)."""
    lstm = pickle.load(open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb"))
    df_coord, df_features = build_basin_tables(list(lstm["valid_ids"]))
    df_features, base_sel, legend, legend_short = _prepare_shap_features({exp: lstm}, df_features, exp)
    # join geodesic catchment perimeter (not in df_features) by gauge_id
    m = dict(np.load(MORPH_NPZ, allow_pickle=True))
    perim_by_id = {str(g): p for g, p in zip(m["gauge_id"], m["perimeter_geod_km"].astype(float))}
    df_features["perimeter_geod_km"] = [perim_by_id.get(str(g), np.nan)
                                        for g in df_features["gauge_id"].values]
    selected = list(base_sel) + list(EXTRA)
    legend = {**legend, **EXTRA_LEGEND}
    legend_short = {**legend_short, **EXTRA_SHORT}

    # target: 25-instance mean F_add (Fig. 5). Colour: 2001-2020 mean simulated ET (held-out run)
    f_lat_mean = fadd_25_instance_mean(df_features["gauge_id"].values)
    active = np.asarray(pd.DatetimeIndex(lstm["idx_daily"]).year >= ACTIVE_START_YEAR)
    et_mean = np.squeeze(lstm["fluxes"]["et"])[:, active].mean(axis=1)
    y = pd.Series(f_lat_mean, index=df_features["gauge_id"].values, name="mean_f_lat")
    et_series = pd.Series(et_mean, index=df_features["gauge_id"].values, name="mean_et")
    X = df_features.set_index("gauge_id")[selected].copy()
    data = pd.concat([X, y], axis=1).dropna()
    X_proc = data[selected]
    et_color = et_series.loc[X_proc.index].values
    yv = data["mean_f_lat"].values
    return selected, legend, legend_short, X_proc, et_color, yv


def _compute(n_iter=100, n_folds=5):
    """100x5-fold CV SHAP for the extended set; cache per-catchment median/IQR SHAP + ranking."""
    import shap
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    selected, legend, legend_short, X_proc, et_color, yv = _prep()
    Xv = X_proc.values
    n, p = Xv.shape
    store = np.full((n_iter, n, p), np.nan)
    rel = np.full((n_iter, p), np.nan)
    from sklearn.metrics import r2_score
    r2_list = []
    for it in range(n_iter):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=it)
        for tr, te in kf.split(Xv):
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=RF_NJOBS).fit(Xv[tr], yv[tr])
            r2_list.append(r2_score(yv[te], rf.predict(Xv[te])))
            store[it, te, :] = shap.TreeExplainer(rf).shap_values(Xv[te])
        mabs = np.abs(store[it]).mean(axis=0)
        rel[it] = mabs / mabs.sum() * 100.0
        if (it + 1) % 20 == 0:
            print(f"  ext SHAP iter {it + 1}/{n_iter}", flush=True)
    shap_median = np.nanmedian(store, axis=0)
    shap_q1 = np.nanpercentile(store, 25, axis=0)
    shap_q3 = np.nanpercentile(store, 75, axis=0)
    rank_med = np.median(rel, axis=0)
    rank_q1 = np.percentile(rel, 25, axis=0)
    rank_q3 = np.percentile(rel, 75, axis=0)
    np.savez(CACHE, cols=np.array(selected, dtype=object),
             shap_median=shap_median, shap_q1=shap_q1, shap_q3=shap_q3,
             rank_med=rank_med, rank_q1=rank_q1, rank_q3=rank_q3,
             Xvals=X_proc.values, et_color=et_color,
             legend_keys=np.array(list(legend.keys()), dtype=object),
             legend_vals=np.array(list(legend.values()), dtype=object),
             r2_mean=float(np.mean(r2_list)), r2_std=float(np.std(r2_list)))
    print(f"[fig6-v4] R^2 = {np.mean(r2_list):.3f} +/- {np.std(r2_list):.3f}")
    print(f"[fig6-v4] cached -> {CACHE}")


def main(n_iter=100):
    """Compute and cache the Fig. 6 surrogate arrays (see the module docstring)."""
    _compute(n_iter=n_iter)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
