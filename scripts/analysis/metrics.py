"""Shared metric helpers for the lateral-flow experiment analysis."""
import numpy as np


def compute_basin_nse(targets, preds):
    """Per-basin NSE. targets/preds: [basins, time] or [basins, time, 1]."""
    targets = np.squeeze(np.asarray(targets))
    preds = np.squeeze(np.asarray(preds))
    num_basins = targets.shape[0]
    nse = np.full(num_basins, np.nan)
    for b in range(num_basins):
        t_b, p_b = targets[b, :], preds[b, :]
        valid = np.isfinite(t_b) & np.isfinite(p_b)
        if valid.sum() < 10:
            continue
        t_v, p_v = t_b[valid], p_b[valid]
        denom = np.sum((t_v - np.mean(t_v)) ** 2)
        if denom == 0:
            continue
        nse[b] = 1 - np.sum((t_v - p_v) ** 2) / denom
    return nse


def basin_nse_table(result, target_channels=('q', 'et', 'swe')):
    """NSE per basin for daily targets (q, et, swe) using the averaged full-output pickle.

    `result['y_all']` channel order matches TARGET_DAILY_COLS from config_global_spatial.py:
    0=streamflow(q), 1=fluxcom_E(et), 2=snow_depth_water_equivalent_mean(swe).
    Predictions: q -> fluxes['q_total'], et -> fluxes['et'], swe -> states['S_SNOW'].
    """
    y = result['y_all']  # [basins, days, 3]
    pred_map = {
        'q':   result['fluxes']['q_total'],
        'et':  result['fluxes']['et'],
        'swe': result['states']['S_SNOW'],
    }
    chan_map = {'q': 0, 'et': 1, 'swe': 2}

    out = {}
    for var in target_channels:
        tgt = y[:, :, chan_map[var]]
        pred = np.squeeze(pred_map[var])
        out[var] = compute_basin_nse(tgt, pred)
    return out


def basin_nse_twsa(result):
    """Per-basin NSE for TWSA (monthly), centering both pred/target per basin over
    months where both are finite (TWSA is an anomaly - absolute level is arbitrary)."""
    z = np.squeeze(result['z_all'])  # [basins, months]
    pred = np.squeeze(result['diagnos']['TWSA_monthly'])  # [basins, months]

    n_months_min = min(z.shape[1], pred.shape[1])
    z = z[:, -n_months_min:]
    pred = pred[:, -n_months_min:]

    both = np.isfinite(z) & np.isfinite(pred)
    z_c, pred_c = z.copy(), pred.copy()
    for b in range(z.shape[0]):
        m = both[b]
        if m.sum() == 0:
            z_c[b] = np.nan
            pred_c[b] = np.nan
            continue
        z_c[b] = z[b] - np.nanmean(z[b, m])
        pred_c[b] = pred[b] - np.nanmean(pred[b, m])

    return compute_basin_nse(z_c, pred_c)
