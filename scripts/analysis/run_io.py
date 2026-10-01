"""
Readers for the averaged ensemble output of one experiment
(results_lateral_experiments/<exp>/averaged_full_output.pickle): load it, pick the simulated TWSA
series, and cut the active-period (2001-2020) series used by the post-hoc comparisons.

Used by catchment_geometry.py, neg_fadd_compare.py (Fig. S15) and loss_weight_compare.py (Fig. S16).
"""
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_utils import ACTIVE_START_YEAR

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")


def _twsa_sim(d):
    for grp in ("diagnos", "states", "fluxes"):
        for key in ("TWSA_monthly", "twsa", "TWSA"):
            if key in d.get(grp, {}):
                return np.squeeze(d[grp][key])
    raise KeyError("no TWSA prediction found")


def _open(exp):
    with open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb") as f:
        return pickle.load(f)


def _series(d):
    ad = np.asarray([t.year >= ACTIVE_START_YEAR for t in d["idx_daily"]])
    am = np.asarray([t.year >= ACTIVE_START_YEAR for t in d["idx_monthly"]])
    fadd = np.squeeze(d["fluxes"]["f_lat"])[:, ad].mean(axis=1)
    et = np.squeeze(d["fluxes"]["et"])[:, ad]
    q = np.squeeze(d["fluxes"]["q_total"])[:, ad]
    swe = np.squeeze(d["states"]["S_SNOW"])[:, ad] if "S_SNOW" in d.get("states", {}) else None
    tw = _twsa_sim(d)[:, am]
    return fadd, et, q, swe, tw, ad, am
