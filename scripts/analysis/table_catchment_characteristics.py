"""
Table A1: descriptive statistics (median, interquartile range, range) of the characteristics of the
1,186 study catchments, from the same catchment tables the model and the surrogates use.

  drainage area, perimeter       geodesic, from catchment_geometry.npz (GRDC-Caravan morphology)
  elevation, slope, snow fraction, land cover, aridity index (PET/P), clay and sand fractions
                                 the model's static attributes (basin_attributes.build_basin_tables;
                                 clay and sand are the SoilGrids fractions new_cly_pc / new_snd_pc)
  mean annual precipitation, potential evapotranspiration, mean air temperature
                                 HydroATLAS sub-basin attributes (pre_mm_syr, pet_mm_syr, tmp_dc_syr)

Land-cover fractions are capped at 100 % (area-weighted aggregation gives up to 101 % in a few
catchments). Prints the table and writes table_A1_catchment_characteristics.csv.

Usage:
  python scripts/analysis/table_catchment_characteristics.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basin_attributes import build_basin_tables, HYDROATLAS_PATH

from project_paths import REPO
FIG = os.path.join(REPO, "results_lateral_experiments", "figures")
GEOMETRY_NPZ = os.path.join(FIG, "catchment_geometry.npz")
OUT_CSV = os.path.join(FIG, "table_A1_catchment_characteristics.csv")


def _rows():
    """(label, values, decimals for median/IQR, decimals for the range) for every table row."""
    geo = dict(np.load(GEOMETRY_NPZ, allow_pickle=True))
    ids = [str(g) for g in geo["gauge_id"]]
    _, df = build_basin_tables(ids)
    df = df.set_index("gauge_id").reindex(ids)
    ha = pd.read_csv(HYDROATLAS_PATH).set_index("gauge_id").reindex(ids)
    cover = lambda c: np.minimum(df[c].values.astype(float), 100.0)
    return [
        ("Drainage area (km²)", geo["area_geod_km2"], 0, 0),
        ("Perimeter (km)", geo["perimeter_geod_km"], 0, 0),
        ("Mean elevation (m)", df["ele_mt_sav"], 0, 0),
        ("Mean slope (°)", df["slp_dg_sav"] / 10.0, 1, 1),     # HydroATLAS stores slope x10
        ("Mean annual precipitation (mm yr⁻¹)", ha["pre_mm_syr"], 0, 0),
        ("Potential evapotranspiration (mm yr⁻¹)", ha["pet_mm_syr"], 0, 0),
        ("Aridity index PET/P (–)", df["ari_ix_conv"], 1, 1),
        ("Mean air temperature (°C)", ha["tmp_dc_syr"] / 10.0, 1, 1),   # stored in 0.1 degC
        ("Snow fraction (%)", df["snw_pc_syr"], 1, 1),
        ("Forest cover (%)", cover("cover_forest"), 1, 0),
        ("Grass cover (%)", cover("cover_grass"), 1, 0),
        ("Crop cover (%)", cover("cover_crop"), 1, 0),
        ("Shrub cover (%)", cover("cover_shrub"), 1, 0),
        ("Clay fraction (%)", df["new_cly_pc"], 1, 0),
        ("Sand fraction (%)", df["new_snd_pc"], 1, 0),
    ], len(ids)


def main():
    """Print Table A1 and write it as CSV."""
    rows, n = _rows()
    out = []
    for label, values, dq, dr in rows:
        x = np.asarray(values, float)
        x = x[np.isfinite(x)]
        med, q1, q3 = np.percentile(x, [50, 25, 75])
        out.append({"Characteristics": label, "Median": f"{med:.{dq}f}",
                    "IQR (25–75)": f"{q1:.{dq}f} – {q3:.{dq}f}",
                    "Range (min–max)": f"{x.min():.{dr}f} – {x.max():.{dr}f}", "n": len(x)})
    table = pd.DataFrame(out)
    print(f"Table A1. Descriptive statistics of catchment characteristics (n = {n}).")
    print(table.to_string(index=False))
    table.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
