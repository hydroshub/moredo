"""
Signed-subsidy analysis of the recurrent model. Compares the negative-allowed run (lstm_af_neg,
lat_alpha in [-0.01, 0.01]) against the baseline (lstm_af, lat_alpha in [0, 0.01]). Both share
architecture, basins, folds and seeds and differ only by the sign freedom of lat_alpha.

Over the 2001-2020 active period, per catchment:
  1. Does the model actually use the negative freedom? signed catchment-mean F_add, the fraction of
     active days with F_add < 0, and the most negative daily F_add.
  2. F_add spatial pattern vs baseline (Pearson/Spearman on catchment-mean, on the |>0| part).
  3. Held-out per-basin NSE for ET, Q, SWE, TWSA under both runs (does signed F_add change skill?).
  4. Where the model learns net loss: correlation of the negative-day fraction and of signed mean
     F_add with aridity (PET/P), mean P, snow fraction and terrain slope.

Obs identical across runs, read from lstm_af. Loads two ~3.3 GB pickles, so run on SLURM.

Usage:
  sbatch --job-name=moredo-negcmp slurm/submit_analysis_cpu.sh neg_fadd_compare.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_basin_nse
from residual_utils import ACTIVE_START_YEAR
from run_io import _open, _twsa_sim
from basin_attributes import build_basin_tables

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT_NPZ = os.path.join(RD, "figures", "neg_fadd_compare.npz")
from project_paths import GRDC_ATTR_DIR
HYDROATLAS = os.path.join(GRDC_ATTR_DIR, "attributes_hydroatlas_grdc.csv")


def _daily(d):
    ad = np.asarray([t.year >= ACTIVE_START_YEAR for t in d["idx_daily"]])
    am = np.asarray([t.year >= ACTIVE_START_YEAR for t in d["idx_monthly"]])
    fadd = np.squeeze(d["fluxes"]["f_lat"])[:, ad]           # signed daily F_add [mm/day]
    et = np.squeeze(d["fluxes"]["et"])[:, ad]
    q = np.squeeze(d["fluxes"]["q_total"])[:, ad]
    swe = np.squeeze(d["states"]["S_SNOW"])[:, ad] if "S_SNOW" in d.get("states", {}) else None
    tw = _twsa_sim(d)[:, am]
    return fadd, et, q, swe, tw, ad, am


def main():
    db = _open("lstm_af")
    dn = _open("lstm_af_neg")
    ids = list(db["valid_ids"])
    assert ids == list(dn["valid_ids"]), "catchment ordering differs"

    fadd_b, et_b, q_b, swe_b, tw_b, ad, am = _daily(db)
    fadd_n, et_n, q_n, swe_n, tw_n, _, _ = _daily(dn)

    y = np.squeeze(db["y_all"])
    q_o = y[:, :, 0][:, ad]; et_o = y[:, :, 1][:, ad]
    swe_o = y[:, :, 2][:, ad] if y.shape[-1] > 2 else None
    z = np.asarray(db["z_all"], float)
    z = z[:, :, 0] if z.ndim == 3 else np.squeeze(z)
    tw_o = z[:, am]

    # --- signed usage ---
    mean_b = np.nanmean(fadd_b, axis=1)
    mean_n = np.nanmean(fadd_n, axis=1)
    neg_frac = np.nanmean(fadd_n < 0, axis=1)          # fraction of active days with net loss
    min_n = np.nanmin(fadd_n, axis=1)                  # most negative daily value
    n_neg_mean = int(np.nansum(mean_n < 0))

    print("\n=== Does the model use the negative freedom? (lstm_af_neg) ===")
    tot = np.isfinite(fadd_n).sum()
    print(f"  fraction of ALL (catchment x active-day) values < 0 : {np.nansum(fadd_n < 0) / tot:.4f}")
    print(f"  catchments with negative mean F_add                 : {n_neg_mean} / {len(ids)}")
    print(f"  median signed mean F_add   baseline {np.nanmedian(mean_b):+.4f}  neg {np.nanmedian(mean_n):+.4f} mm/day")
    print(f"  median negative-day fraction per catchment          : {np.nanmedian(neg_frac):.4f}")
    print(f"  90th pct negative-day fraction                      : {np.nanpercentile(neg_frac, 90):.4f}")
    print(f"  median of most-negative daily F_add                 : {np.nanmedian(min_n):+.4f} mm/day")

    # --- pattern vs baseline ---
    v = np.isfinite(mean_b) & np.isfinite(mean_n)
    r = stats.pearsonr(mean_b[v], mean_n[v])[0]; rho = stats.spearmanr(mean_b[v], mean_n[v])[0]
    print("\n=== F_add spatial pattern: baseline vs signed ===")
    print(f"  catchment-mean  Pearson r = {r:+.3f}  Spearman rho = {rho:+.3f}  n = {int(v.sum())}")

    # --- held-out skill ---
    print("\n=== Held-out skill (median per-basin NSE) ===")
    rows = [("ET", et_b, et_n, et_o), ("Q", q_b, q_n, q_o), ("SWE", swe_b, swe_n, swe_o),
            ("TWSA", tw_b, tw_n, tw_o)]
    nse_out = {}
    print(f"  {'var':5s}{'baseline':>12s}{'signed':>12s}{'delta':>12s}")
    for lab, sb, sn, o in rows:
        if o is None or sb is None or sn is None:
            continue
        nb = compute_basin_nse(o, sb); nn = compute_basin_nse(o, sn)
        nse_out[f"nse_{lab}_base"] = np.asarray(nb, float)
        nse_out[f"nse_{lab}_neg"] = np.asarray(nn, float)
        print(f"  {lab:5s}{np.nanmedian(nb):>+12.3f}{np.nanmedian(nn):>+12.3f}{np.nanmedian(nn - nb):>+12.3f}")

    # --- where does net loss appear? ---
    ha = pd.read_csv(HYDROATLAS).set_index("gauge_id").reindex([str(i) for i in ids])
    arid = (ha["pet_mm_syr"] / ha["pre_mm_syr"]).values
    pmean = ha["pre_mm_syr"].values
    snow = ha["snw_pc_syr"].values
    slope = (ha["slp_dg_sav"] / 10.0).values
    print("\n=== Where net loss is learned (drivers of neg-day fraction & signed mean) ===")
    for name, x in [("aridity PET/P", arid), ("mean P [mm/yr]", pmean),
                    ("snow fraction [%]", snow), ("slope [deg]", slope)]:
        vv = np.isfinite(x) & np.isfinite(neg_frac)
        rf = stats.spearmanr(x[vv], neg_frac[vv])[0]
        rm = stats.spearmanr(x[vv], mean_n[vv])[0]
        print(f"  {name:18s}: rho(neg-frac) = {rf:+.3f}   rho(signed mean) = {rm:+.3f}")

    dc, _ = build_basin_tables([str(i) for i in ids])
    dc = dc.set_index("gauge_id").reindex([str(i) for i in ids])
    save = dict(gauge_id=np.array(ids, dtype=object),
                mean_base=mean_b, mean_neg=mean_n, neg_frac=neg_frac, min_neg=min_n,
                aridity=arid, lon=dc["basin_lon"].values, lat=dc["basin_lat"].values)
    save.update(nse_out)
    np.savez(OUT_NPZ, **save)
    print(f"\nSaved -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
