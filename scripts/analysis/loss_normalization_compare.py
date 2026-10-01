"""
Cache for Fig. S3: the recurrent representation trained with the cross-catchment pooled loss
(lstm_af, Eq. 9) against the catchment-normalized loss (lstm_af_nsestar, config
config_lateral_lstm_af_nsestar.py, LOSS_TYPE = "nse_star"). Both are strictly held-out estimates
(averaged pickles, build_averaged_output.py).

Writes figures/loss_normalization_compare.npz with, per catchment: gauge_id, lon, lat, the
2001-2020 catchment-mean F_add of both models and the held-out per-catchment NSE of ET, Q, SWE and
TWSA of both models.

Usage (loads two ~3.4 GB pickles, run on a CPU node):
  sbatch --job-name=moredo-lossnorm slurm/submit_analysis_cpu.sh loss_normalization_compare.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import basin_nse_table, basin_nse_twsa
from basin_attributes import build_basin_tables
from residual_utils import ACTIVE_START_YEAR

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT = os.path.join(RD, "figures", "loss_normalization_compare.npz")
EXPS = {"pooled": "lstm_af", "normalized": "lstm_af_nsestar"}
VARS = ["et", "q", "swe", "twsa"]


def _load(exp):
    """Catchment-mean F_add (2001-2020) and held-out per-catchment NSE of one experiment."""
    d = pickle.load(open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb"))
    idx = pd.DatetimeIndex(d["idx_daily"])
    act = np.asarray(idx.year >= ACTIVE_START_YEAR)
    t = basin_nse_table(d, target_channels=("q", "et", "swe"))
    t["twsa"] = basin_nse_twsa(d)
    df = pd.DataFrame({"fadd": np.squeeze(d["fluxes"]["f_lat"])[:, act].mean(axis=1),
                       **{f"nse_{v}": np.asarray(t[v], float) for v in VARS}},
                      index=[str(g) for g in d["valid_ids"]])
    print(f"{exp}: {len(df)} catchments", flush=True)
    return df


def main():
    res = {k: _load(e) for k, e in EXPS.items()}
    ids = [g for g in res["pooled"].index if g in res["normalized"].index]
    dc, _ = build_basin_tables(ids)
    dc = dc.set_index("gauge_id").reindex(ids)
    out = dict(gauge_id=np.array(ids, dtype=object), lon=dc["basin_lon"].values,
               lat=dc["basin_lat"].values)
    for k in EXPS:
        out[f"fadd_{k}"] = res[k].loc[ids, "fadd"].values
        for v in VARS:
            out[f"nse_{v}_{k}"] = res[k].loc[ids, f"nse_{v}"].values
    np.savez(OUT, **out)
    fp, fn = out["fadd_pooled"], out["fadd_normalized"]
    print(f"n = {len(ids)}  r = {np.corrcoef(fp, fn)[0, 1]:.3f}  median F_add pooled {np.median(fp):.3f}  "
          f"normalized {np.median(fn):.3f} mm/day")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
