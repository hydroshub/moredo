"""
Loss-weight sensitivity comparison of the recurrent model: main (lstm_af, all weights 1) vs
TWSA-weight-0 (lstm_af_twsa0) vs Q-weight-0 (lstm_af_q0). Shows how much the GRACE-TWSA and the
streamflow constraints each shape the learned F_add and the held-out skill.

Over the active period (2001-2020, 1186 catchments):
  1. catchment-mean F_add per run + pairwise Pearson/Spearman correspondence with the main model;
  2. held-out per-basin NSE for ET, Q, SWE, TWSA under each run (medians printed).

Saves per-basin arrays to loss_weight_compare.npz for the figure scripts. Obs are identical across
runs and read from lstm_af. Loads three ~3.3 GB pickles, so run on SLURM.

Usage:
  sbatch --job-name=moredo-wcmp slurm/submit_analysis_cpu.sh loss_weight_compare.py
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_basin_nse
from run_io import _open, _series

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT_NPZ = os.path.join(RD, "figures", "loss_weight_compare.npz")

EXPS = ["lstm_af", "lstm_af_twsa0", "lstm_af_q0"]
LABELS = {"lstm_af": "main (all=1)", "lstm_af_twsa0": "TWSA weight 0", "lstm_af_q0": "Q weight 0"}


def main():
    data = {e: _open(e) for e in EXPS}
    ref = data["lstm_af"]
    ids = list(ref["valid_ids"])
    for e in EXPS:
        assert list(data[e]["valid_ids"]) == ids, f"catchment ordering differs for {e}"

    fadd, et, q, swe, tw = {}, {}, {}, {}, {}
    for e in EXPS:
        fadd[e], et[e], q[e], swe[e], tw[e], ad, am = _series(data[e])

    y = np.squeeze(ref["y_all"])
    q_o = y[:, :, 0][:, ad]; et_o = y[:, :, 1][:, ad]
    swe_o = y[:, :, 2][:, ad] if y.shape[-1] > 2 else None
    z = np.asarray(ref["z_all"], float); z = z[:, :, 0] if z.ndim == 3 else np.squeeze(z)
    tw_o = z[:, am]

    print("\n=== F_add spatial pattern vs main model ===")
    def corr(a, b):
        v = np.isfinite(a) & np.isfinite(b)
        return stats.pearsonr(a[v], b[v])[0], stats.spearmanr(a[v], b[v])[0], int(v.sum())
    for e in EXPS[1:]:
        r, rho, n = corr(fadd["lstm_af"], fadd[e])
        print(f"  main vs {LABELS[e]:14s}: Pearson r={r:+.3f}  Spearman rho={rho:+.3f}  n={n}")
    for e in EXPS:
        print(f"  median mean-F_add {LABELS[e]:14s} = {np.nanmedian(fadd[e]):.3f} mm/day")

    print("\n=== Held-out skill (median per-basin NSE) ===")
    var_obs = [("ET", et, et_o), ("Q", q, q_o), ("SWE", swe, swe_o), ("TWSA", tw, tw_o)]
    nse_store = {}
    print("  var  " + "".join(f"{LABELS[e]:>16s}" for e in EXPS))
    for vname, sim, o in var_obs:
        if o is None:
            continue
        line = f"  {vname:4s}"
        for e in EXPS:
            nse = compute_basin_nse(o, sim[e])
            nse_store[f"nse_{vname}_{e}"] = np.asarray(nse, float)
            line += f"{np.nanmedian(nse):>16.3f}"
        print(line)

    save = {"gauge_id": np.array(ids, dtype=object)}
    for e in EXPS:
        save[f"fadd_{e}"] = np.asarray(fadd[e], float)
    save.update(nse_store)
    np.savez(OUT_NPZ, **save)
    print(f"\nSaved -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
