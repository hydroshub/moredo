"""
Fig. 1 input: observational long-term water-balance non-closure per catchment over 2001-2020.

Definitions (catchment long-term means over the active period 2001-2020, mm/day unless noted):
  R_in       = ET_obs + Q_obs + dS/dt - P            (= -residual_corrected from residual_utils)
  R_in / P   = R_in / P_mean                          (dimensionless)
  ET / P     = ET_obs_mean / P_mean                   (Budyko y-axis, evaporative index)
  PET / P    = pet_mm_syr / (P_mean * 365.25)         (Budyko x-axis, dryness index, PET from HydroATLAS)

Writes fig_budyko_nonclosure_cache.npz, rendered as Fig. 1 by fig1_nonclosure_inset.py.
Loads the ~3.4 GB averaged pickle, run on SLURM:
  sbatch --job-name=moredo-fig1 slurm/submit_analysis_cpu.sh nonclosure_cache.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_utils import (ACTIVE_START_YEAR, compute_residual_vs_fadd,
                            _reconstruct_precip, _daily_series, _to_monthly_mean)

from project_paths import RESULTS_DIR
from project_paths import GRDC_ATTR_DIR
HYDROATLAS = os.path.join(GRDC_ATTR_DIR, "attributes_hydroatlas_grdc.csv")


def _active_basin_mean(daily, idx_daily):
    """Long-term basin mean rate over the active period, via monthly means (matches residual_utils)."""
    m, midx = _to_monthly_mean(daily, idx_daily)
    m = m[:, midx.year >= ACTIVE_START_YEAR]
    with np.errstate(invalid="ignore"):
        return np.nanmean(m, axis=1)


def compute_and_cache(cache_path):
    from run_analysis import load_results
    from basin_attributes import build_basin_tables

    print("Loading averaged full-output pickle (lstm_af only)...")
    results = load_results(RESULTS_DIR, experiments=["lstm_af"])
    res = results["lstm_af"]
    valid_ids = list(res["valid_ids"])
    idx_daily = res["idx_daily"]

    print("Building basin coordinate/attribute tables...")
    df_coord, _ = build_basin_tables(valid_ids)

    # residual machinery: gives twsa_trend (dS/dt, mm/day), residual_corrected, mean_fadd, valid
    r = compute_residual_vs_fadd(res, df_coord)
    residual_corrected = r["residual_corrected"]       # P - Q - ET - dS  (mm/day)
    twsa_trend = r["twsa_trend"]                        # dS/dt (mm/day)
    fadd = r["mean_fadd"]
    valid = r["valid"]

    # basin long-term mean rates (mm/day), same active-period monthly-mean pipeline
    p_mean = _active_basin_mean(_reconstruct_precip(res), idx_daily)
    et_mean = _active_basin_mean(res["y_all"][:, :, 1], idx_daily)
    q_mean = _active_basin_mean(res["y_all"][:, :, 0], idx_daily)

    r_in = -residual_corrected                          # apparent missing input, mm/day
    # consistency check: R_in should equal ET + Q + dS - P
    chk = et_mean + q_mean + twsa_trend - p_mean
    ok = np.isfinite(r_in) & np.isfinite(chk)
    print(f"R_in vs (ET+Q+dS-P) max abs diff = {np.nanmax(np.abs(r_in[ok]-chk[ok])):.4g} mm/day")

    # PET (mm/yr) from HydroATLAS, aligned to valid_ids
    ha = pd.read_csv(HYDROATLAS).set_index("gauge_id")
    pet_mm_yr = ha["pet_mm_syr"].reindex(df_coord["gauge_id"].values).values.astype(float)
    print(f"PET join: {np.isfinite(pet_mm_yr).sum()}/{len(pet_mm_yr)} basins have pet_mm_syr")

    with np.errstate(invalid="ignore", divide="ignore"):
        rin_over_p = r_in / p_mean
        et_over_p = et_mean / p_mean
        pet_over_p = pet_mm_yr / (p_mean * 365.25)

    np.savez(cache_path,
             gauge_id=np.array(df_coord["gauge_id"].values, dtype=object),
             basin_lon=df_coord["basin_lon"].values, basin_lat=df_coord["basin_lat"].values,
             valid=valid, r_in=r_in, p_mean=p_mean, et_mean=et_mean, q_mean=q_mean,
             twsa_trend=twsa_trend, fadd=fadd, pet_mm_yr=pet_mm_yr,
             rin_over_p=rin_over_p, et_over_p=et_over_p, pet_over_p=pet_over_p)
    print(f"Cached -> {cache_path}")
    return dict(np.load(cache_path, allow_pickle=True))


def main():
    """Compute the non-closure diagnostics and write the Fig. 1 cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(RESULTS_DIR, "figures", "fig_budyko_nonclosure_cache.npz"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    compute_and_cache(args.cache)


if __name__ == "__main__":
    main()
