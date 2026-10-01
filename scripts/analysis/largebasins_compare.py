"""
Cache for the large-catchment SI figure (fig_si_largebasins.py): the recurrent representation
retrained on the 362 catchments larger than 1000 km^2 only (lstm_af_largebasins, same config as
lstm_af, trained with slurm/submit_train_global.sh and basin_list_largebasins_ge1000km2.txt)
against the main recurrent model (lstm_af, all 1186 catchments), on the shared 362 catchments.
Both are strictly held-out estimates (averaged pickles, build_averaged_output.py).

Writes figures/largebasins_compare.npz with, per catchment: gauge_id, lon, lat, Koppen-Geiger
major class, drainage area, 2001-2020 catchment-mean F_add of both models and held-out
per-catchment NSE of ET, Q, SWE and TWSA of both models.

Usage (loads two ~3.4 GB pickles, run on a CPU node):
  sbatch --job-name=moredo-largeb slurm/submit_analysis_cpu.sh largebasins_compare.py
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
LARGE_LIST = os.path.join(REPO, "basin_list_largebasins_ge1000km2.txt")
GEOM = os.path.join(RD, "figures", "catchment_geometry.npz")
OUT = os.path.join(RD, "figures", "largebasins_compare.npz")
EXPS = {"main": "lstm_af", "large": "lstm_af_largebasins"}
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
    large = [x.strip() for x in open(LARGE_LIST) if x.strip()]
    res = {k: _load(e) for k, e in EXPS.items()}
    ids = [g for g in large if g in res["main"].index and g in res["large"].index]
    dc, _ = build_basin_tables(ids)
    dc = dc.set_index("gauge_id").reindex(ids)
    g = np.load(GEOM, allow_pickle=True)
    area = dict(zip([str(x) for x in g["gauge_id"]], g["area_geod_km2"].astype(float)))
    out = dict(gauge_id=np.array(ids, dtype=object),
               lon=dc["basin_lon"].values, lat=dc["basin_lat"].values,
               kg=dc["kg_major_id"].values,
               area_km2=np.array([area.get(i, np.nan) for i in ids]))
    for k in EXPS:
        out[f"fadd_{k}"] = res[k].loc[ids, "fadd"].values
        for v in VARS:
            out[f"nse_{v}_{k}"] = res[k].loc[ids, f"nse_{v}"].values
    np.savez(OUT, **out)
    fa, fb = out["fadd_main"], out["fadd_large"]
    print(f"n = {len(ids)} shared catchments, geodesic area min {np.nanmin(out['area_km2']):.0f} km2")
    print(f"r = {np.corrcoef(fa, fb)[0, 1]:.3f}  median F_add main {np.median(fa):.3f}  large-only {np.median(fb):.3f} mm/day")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
