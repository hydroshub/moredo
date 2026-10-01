"""
Bound-insensitivity check for the capacity-normalized subsidy coefficient. Compares the learned
F_add of the recurrent representation trained with the default coefficient bound [0, 0.01]
(lstm_af) against a re-training with the bound doubled to [0, 0.02] (lstm_af_cap002). If F_add is
essentially unchanged, the [0, 0.01] range is not shaping the result.

Reports the Pearson correlation of the pooled F_add samples between the two bounds, both as
catchment means (one value per basin) and as all pooled catchment-days over the active period,
together with the mean magnitudes so any growth in F_add is visible.

Usage:
  python scripts/analysis/fadd_bound_sensitivity.py
"""
import os
import pickle
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_utils import ACTIVE_START_YEAR

from project_paths import RESULTS_DIR as RD
EXP_A, LAB_A = "lstm_af", "cap=0.01"
EXP_B, LAB_B = "lstm_af_cap002", "cap=0.02"


def load_fadd(exp):
    """Load the 2001-2020 active-period F_add [basins, days] and valid ids for one experiment."""
    with open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb") as f:
        r = pickle.load(f)
    f_lat = np.squeeze(r["fluxes"]["f_lat"])          # [basins, days]
    idx = r["idx_daily"]
    active = np.asarray([d.year >= ACTIVE_START_YEAR for d in idx])
    return f_lat[:, active], list(r["valid_ids"])


def main():
    """Report the F_add correlation between the default and doubled coefficient bounds (catchment-mean and pooled catchment-days)."""
    fa, ids_a = load_fadd(EXP_A)
    fb, ids_b = load_fadd(EXP_B)

    # Align basins by valid_ids (defensive; the two runs share the basin list and order).
    if ids_a != ids_b:
        common = [i for i in ids_a if i in set(ids_b)]
        ia = [ids_a.index(i) for i in common]
        ib = [ids_b.index(i) for i in common]
        fa, fb = fa[ia], fb[ib]
        print(f"aligned to {len(common)} common basins")
    else:
        print(f"{len(ids_a)} basins, identical order")

    # --- catchment-mean F_add (spatial pattern) ---
    ma, mb = np.nanmean(fa, axis=1), np.nanmean(fb, axis=1)
    v = np.isfinite(ma) & np.isfinite(mb)
    r_mean, p_mean = stats.pearsonr(ma[v], mb[v])

    # --- all pooled catchment-days ---
    pa, pb = fa.ravel(), fb.ravel()
    vp = np.isfinite(pa) & np.isfinite(pb)
    r_pool, p_pool = stats.pearsonr(pa[vp], pb[vp])

    print(f"\nF_add bound sensitivity ({LAB_A} vs {LAB_B})")
    print(f"  catchment-mean:   r = {r_mean:.3f}  (p = {p_mean:.1e}, n = {v.sum()})")
    print(f"  pooled cat-days:  r = {r_pool:.3f}  (p = {p_pool:.1e}, n = {vp.sum()})")
    print(f"  mean F_add:       {LAB_A} = {np.nanmean(ma):.4f} mm/day   "
          f"{LAB_B} = {np.nanmean(mb):.4f} mm/day")
    print(f"  median cat-mean:  {LAB_A} = {np.nanmedian(ma):.4f}         "
          f"{LAB_B} = {np.nanmedian(mb):.4f}")
    print(f"  95th pct cat-mean:{LAB_A} = {np.nanpercentile(ma,95):.4f}         "
          f"{LAB_B} = {np.nanpercentile(mb,95):.4f}")


if __name__ == "__main__":
    main()
