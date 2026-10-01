"""
Fig. S9: SHAP predictor ranking of the random-forest surrogate fitted to the strictly held-out
F_add estimate (fig_si_repr_heldout.npz), over the 18 predictors of Fig. 6 (base 14 plus karst
extent, catchment area, catchment perimeter and lithology class), as a vertical bar chart with
predictors ordered by decreasing importance.

Same surrogate configuration and validation as Fig. 6 (shap_fig6_cache.py): a 200-tree random
forest evaluated with 100 repetitions of 5-fold cross-validation, every catchment held out once
per repetition with its SHAP values taken from that held-out fold. Per repetition the mean
absolute SHAP value is normalized across predictors. Bars show the median across repetitions,
whiskers the interquartile range, and R^2 is the mean +/- 1 std over all test folds. Results are
cached to fig_si_heldout_shap_cv_cache.npz.

Heavy on first run (500 forest fits plus TreeSHAP), run on SLURM:
  sbatch --job-name=moredo-s9 slurm/submit_analysis_cpu.sh fig_si_s10_heldout_shap_vertical.py
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, EXP_COLORS
from basin_attributes import build_basin_tables
from manuscript_figures import _prepare_shap_features

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")
MORPH_NPZ = os.path.join(OUT_DIR, "catchment_geometry.npz")
CACHE = os.path.join(OUT_DIR, "fig_si_heldout_shap_cv_cache.npz")
RF_NJOBS = int(os.environ.get("RF_NJOBS", "-1"))

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


def _prep():
    """Feature matrix (18 predictors of Fig. 6) and the strictly held-out F_add target."""
    d = np.load(os.path.join(OUT_DIR, "fig_si_repr_heldout.npz"), allow_pickle=True)
    fadd_heldout = d["fadd_heldout"]; vids = list(d["gauge_id"])
    _, dff = build_basin_tables(vids)
    lstm = pickle.load(open(os.path.join(REPO, "results_lateral_experiments/lstm_af/averaged_full_output.pickle"), "rb"))
    dff2, feats, _, _ = _prepare_shap_features({"lstm_af": lstm}, dff, "lstm_af")
    m = dict(np.load(MORPH_NPZ, allow_pickle=True))
    perim_by_id = {str(g): p for g, p in zip(m["gauge_id"], m["perimeter_geod_km"].astype(float))}
    dff2 = dff2.copy()
    dff2["perimeter_geod_km"] = [perim_by_id.get(str(g), np.nan) for g in dff2["gauge_id"].values]
    feats = list(feats) + EXTRA
    X = dff2.set_index("gauge_id")[feats].copy()
    y = pd.Series(fadd_heldout, index=dff2["gauge_id"].values, name="t")
    data = pd.concat([X, y], axis=1).dropna()
    return feats, data[feats].values, data["t"].values


def _ranking_cv(n_iter=100, n_folds=5):
    """100x5-fold CV SHAP ranking (as in Fig. 6), cached. Returns feats, median/q1/q3 relative
    importance (%) across repetitions, and the mean/std test-fold R^2."""
    if os.path.exists(CACHE):
        cz = np.load(CACHE, allow_pickle=True)
        if int(cz["n_iter"]) == n_iter:
            print(f"[S9] loaded CV cache {CACHE}")
            return ([str(f) for f in cz["feats"]], cz["med"], cz["q1"], cz["q3"],
                    float(cz["r2_mean"]), float(cz["r2_std"]), int(cz["n"]))
    import shap
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    feats, Xv, yv = _prep()
    n, p = Xv.shape
    rel = np.full((n_iter, p), np.nan)
    r2_list = []
    for it in range(n_iter):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=it)
        store = np.full((n, p), np.nan)    # out-of-fold signed SHAP, all catchments
        for tr, te in kf.split(Xv):
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=RF_NJOBS).fit(Xv[tr], yv[tr])
            r2_list.append(r2_score(yv[te], rf.predict(Xv[te])))
            store[te, :] = shap.TreeExplainer(rf).shap_values(Xv[te])
        mabs = np.abs(store).mean(axis=0)
        rel[it] = mabs / mabs.sum() * 100.0
        if (it + 1) % 20 == 0:
            print(f"  [S9] CV iteration {it + 1}/{n_iter}", flush=True)
    med, q1, q3 = np.median(rel, 0), np.percentile(rel, 25, 0), np.percentile(rel, 75, 0)
    r2m, r2s = float(np.mean(r2_list)), float(np.std(r2_list))
    np.savez(CACHE, feats=np.array(feats, dtype=object), med=med, q1=q1, q3=q3,
             r2_mean=r2m, r2_std=r2s, n_iter=n_iter, n=n)
    print(f"[S9] cached CV -> {CACHE}")
    return feats, med, q1, q3, r2m, r2s, n


def main():
    """Render Fig. S9 (computing the cross-validated ranking on first use)."""
    feats, med, q1, q3, r2m, r2s, n = _ranking_cv()
    order = np.argsort(med)[::-1]             # descending, left to right
    print(f"[S9] held-out F_add surrogate R^2 = {r2m:.3f} +/- {r2s:.3f}  (n={n}, p={len(feats)})")
    for i in order:
        print(f"    {LEGEND_SHORT[feats[i]]:16s} {med[i]:5.1f}%  (IQR {q1[i]:.1f}-{q3[i]:.1f})")

    fig = uplt.figure(figwidth="165mm", figheight="105mm", tight=True, spanx=False, spany=False,
                      sharex=False, sharey=False)
    ax = fig.add_subplot()
    xp = np.arange(len(order))
    ax.bar(xp, height=med[order], width=0.7, color=EXP_COLORS["lstm_af"], edgecolor="k", lw=0.4)
    err = np.vstack([med[order] - q1[order], q3[order] - med[order]])
    ax.errorbar(xp, med[order], yerr=err, fmt="none", ecolor="k", elinewidth=0.8, capsize=2, zorder=5)
    ax.format(xticks=xp, xticklabels=[LEGEND_SHORT[feats[i]] for i in order], xrotation=90,
              ylabel="Relative SHAP contribution (%)", xlabel="",
              title="", titlesize=TITLESIZE,
              xlabelsize=LABELSIZE, yticklabelsize=TICKLABELSIZE, xticklabelsize=TICKLABELSIZE - 1,
              grid=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    out = os.path.join(OUT_DIR, "fig_si_s10_heldout_shap_vertical.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
