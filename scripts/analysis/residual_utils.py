"""
Shared helpers for the averaged ensemble outputs: daily and monthly series, the reconstructed
precipitation, and the long-term observational water-balance residual (compute_residual_vs_fadd)
used by the non-closure caches and Fig. S14.
"""
import numpy as np
import pandas as pd
from scipy import stats

ACTIVE_START_YEAR = 2001


def _daily_series(result, key, group="fluxes"):
    """Return one variable from a result pickle as a squeezed [basins, days] array."""
    return np.squeeze(result[group][key])  # [basins, days]


def _reconstruct_precip(result):
    """P = rain + snow exactly reproduces the raw driver (rainsnow_partition conserves mass)."""
    return _daily_series(result, "rain") + _daily_series(result, "snow")


def _to_monthly_mean(daily_data, idx_daily):
    """[basins, days] daily mean-rate series -> [basins, months] monthly mean rate."""
    df = pd.DataFrame(daily_data.T, index=idx_daily)
    monthly = df.resample("MS").mean()
    return monthly.values.T, monthly.index  # [basins, months], month index


def compute_residual_vs_fadd(result, df_coord, end_year=2020):
    """Compare the learned F_add with the observational P-Q-ET residual per basin, returning aligned
    monthly-mean series and their spatial catchment-mean correspondence up to end_year."""
    idx_daily = result["idx_daily"]
    f_lat = _daily_series(result, "f_lat")
    precip = _reconstruct_precip(result)
    y_all = result["y_all"]  # [basins, days, 3] = q_obs, et_obs, swe_obs
    q_obs = y_all[:, :, 0]
    et_obs = y_all[:, :, 1]

    residual_daily = precip - q_obs - et_obs  # mm/day, observational

    f_lat_m, month_idx_f = _to_monthly_mean(f_lat, idx_daily)
    residual_m, month_idx_r = _to_monthly_mean(residual_daily, idx_daily)
    assert (month_idx_f == month_idx_r).all()

    active_mask = month_idx_f.year >= ACTIVE_START_YEAR
    f_lat_m = f_lat_m[:, active_mask]
    residual_m = residual_m[:, active_mask]

    # Observed TWSA (z_all) monthly storage trend per basin (mean month-to-month rate, mm/day-equivalent)
    z = np.squeeze(result["z_all"])  # [basins, months] — assumed aligned to idx_monthly
    idx_monthly = result["idx_monthly"]
    z_df = pd.DataFrame(z.T, index=idx_monthly)
    z_active = z_df.loc[z_df.index.year >= ACTIVE_START_YEAR]

    twsa_trend = np.full(z.shape[0], np.nan)  # mm/day equivalent
    for i, col in enumerate(z_active.columns):
        series = z_active[col].dropna()
        if len(series) < 6:
            continue
        t = (series.index - series.index[0]).days.values.astype(float)
        twsa_trend[i] = stats.linregress(t, series.values).slope

    # --- basin-level long-term means for spatial correlation ---
    with np.errstate(invalid="ignore"):
        mean_residual = np.nanmean(residual_m, axis=1)
        mean_fadd = np.nanmean(f_lat_m, axis=1)
    residual_corrected = mean_residual - twsa_trend  # subtract storage trend

    valid = np.isfinite(mean_residual) & np.isfinite(mean_fadd) & np.isfinite(twsa_trend)
    r_raw, p_raw = stats.pearsonr(mean_residual[valid], mean_fadd[valid])
    r_corr, p_corr = stats.pearsonr(residual_corrected[valid], mean_fadd[valid])

    # --- per-basin temporal correlation (does f_add track the residual month-to-month?) ---
    temporal_r = np.full(f_lat_m.shape[0], np.nan)
    for i in range(f_lat_m.shape[0]):
        f_i, r_i = f_lat_m[i], residual_m[i]
        m = np.isfinite(f_i) & np.isfinite(r_i)
        if m.sum() < 24:  # need at least 2 years of monthly data
            continue
        if np.std(f_i[m]) == 0 or np.std(r_i[m]) == 0:
            continue
        temporal_r[i], _ = stats.pearsonr(f_i[m], r_i[m])

    return {
        "mean_residual": mean_residual, "mean_fadd": mean_fadd,
        "residual_corrected": residual_corrected, "twsa_trend": twsa_trend,
        "r_spatial_raw": r_raw, "p_spatial_raw": p_raw,
        "r_spatial_corrected": r_corr, "p_spatial_corrected": p_corr,
        "temporal_r": temporal_r,
        "valid": valid,
    }
