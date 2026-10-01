"""
Catchment geometry for the random-forest surrogates and Fig. S10: geodesic catchment perimeter and
area from the GRDC-Caravan morphology attributes (attributes_morphology_grdc.csv), plus the
2001-2020 catchment-mean F_add of the recurrent representation, in the model's catchment order.

Writes catchment_geometry.npz (gauge_id, perimeter_geod_km, area_geod_km2, fadd_lstm_af), read by
shap_fig6_cache.py (Fig. 6), fig_si_s10_heldout_shap_vertical.py
(Fig. S9), fig_morph_geometry.py (Fig. S10), shap_seasonal_cache.py
(Fig. S11) and fig_wcmp_combined.py (Fig. S16).

Loads the ~3.4 GB averaged pickle of the recurrent model (CPU only).
Usage:
  python scripts/analysis/catchment_geometry.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_io import _open, _series

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT_NPZ = os.path.join(RD, "figures", "catchment_geometry.npz")
from project_paths import GRDC_ATTR_DIR
MORPH_PATH = os.path.join(GRDC_ATTR_DIR, "attributes_morphology_grdc.csv")


def main(out_npz=OUT_NPZ):
    """Join perimeter and area to the recurrent model's catchments and store them with mean F_add."""
    d = _open("lstm_af")
    ids = list(d["valid_ids"])
    fadd = _series(d)[0]                      # 2001-2020 catchment-mean F_add (mm/day)
    m = pd.read_csv(MORPH_PATH).set_index("gauge_id").reindex(ids)
    np.savez(out_npz,
             gauge_id=np.array(ids, dtype=object),
             perimeter_geod_km=m["perimeter_geod_km"].values.astype(float),
             area_geod_km2=m["area_geod_km2"].values.astype(float),
             fadd_lstm_af=np.asarray(fadd, float))
    print(f"Saved -> {out_npz}")


if __name__ == "__main__":
    main()
