"""
Build basin coordinate/attribute tables for the GRDC-global lateral-flow experiments.

Mirrors the attribute handling in libs/data.py's AttributeProcessor (same cover_*
land-cover-fraction derivation from glc_pc_s01-22) and the aridity
classification of the manuscript, but reads the raw (non-normalized) attribute CSVs directly, and uses
gauge station coordinates (lat_pp/long_pp) instead of basin-polygon centroids since no
GRDC basin shapefile is used here.

Returns (df_coord, df_features) both indexed by gauge_id, restricted to valid_ids:
  df_coord    : gauge_id, basin_lon, basin_lat, aridity_category (1=arid..5=hyper-humid)
  df_features : df_coord + all raw attribute columns + derived cover_* fractions
"""
import numpy as np
import pandas as pd

CATEGORY_NAMES = {1: "arid", 2: "semi-arid", 3: "semi-humid", 4: "humid", 5: "hyper-humid"}
CATEGORY_COLORS = {1: "#8c510a", 2: "#d8b365", 3: "#808080", 4: "#5ab4ac", 5: "#01665e"}

# ISO country codes present in the GRDC attribute table (see country value_counts)
NORTH_AMERICA_COUNTRIES = {"CA", "US"}
EUROPE_COUNTRIES = {"DE", "NO", "ES", "AT", "FI", "SE", "CH", "BE", "EE", "IE", "CZ", "IS", "LU"}

import os
from project_paths import GRDC_ATTR_DIR, KG_NC_PATH
HYDROATLAS_PATH = os.path.join(GRDC_ATTR_DIR, "attributes_hydroatlas_grdc.csv")
ADDITIONAL_PATH = os.path.join(GRDC_ATTR_DIR, "attributes_additional_grdc.csv")
SOIL_PATH = os.path.join(GRDC_ATTR_DIR, "attributes_soil_grdc.csv")

# Köppen-Geiger classification (Beck et al. 2023, https://www.nature.com/articles/s41597-023-02549-6),
# 0.1-degree native grid. Fine-class table and 8-major-group reclassification as in the MOREDO analysis
# notebooks of the ML4HES group, for consistency with the MOREDO figures.

KG_CLASSES = {
    1:  {'code': 'Af',  'description': 'Tropical, rainforest'},
    2:  {'code': 'Am',  'description': 'Tropical, monsoon'},
    3:  {'code': 'Aw',  'description': 'Tropical, savannah'},
    4:  {'code': 'BWh', 'description': 'Arid, desert, hot'},
    5:  {'code': 'BWk', 'description': 'Arid, desert, cold'},
    6:  {'code': 'BSh', 'description': 'Arid, steppe, hot'},
    7:  {'code': 'BSk', 'description': 'Arid, steppe, cold'},
    8:  {'code': 'Csa', 'description': 'Temperate, dry summer, hot summer'},
    9:  {'code': 'Csb', 'description': 'Temperate, dry summer, warm summer'},
    10: {'code': 'Csc', 'description': 'Temperate, dry summer, cold summer'},
    11: {'code': 'Cwa', 'description': 'Temperate, dry winter, hot summer'},
    12: {'code': 'Cwb', 'description': 'Temperate, dry winter, warm summer'},
    13: {'code': 'Cwc', 'description': 'Temperate, dry winter, cold summer'},
    14: {'code': 'Cfa', 'description': 'Temperate, no dry season, hot summer'},
    15: {'code': 'Cfb', 'description': 'Temperate, no dry season, warm summer'},
    16: {'code': 'Cfc', 'description': 'Temperate, no dry season, cold summer'},
    17: {'code': 'Dsa', 'description': 'Cold, dry summer, hot summer'},
    18: {'code': 'Dsb', 'description': 'Cold, dry summer, warm summer'},
    19: {'code': 'Dsc', 'description': 'Cold, dry summer, cold summer'},
    20: {'code': 'Dsd', 'description': 'Cold, dry summer, very cold winter'},
    21: {'code': 'Dwa', 'description': 'Cold, dry winter, hot summer'},
    22: {'code': 'Dwb', 'description': 'Cold, dry winter, warm summer'},
    23: {'code': 'Dwc', 'description': 'Cold, dry winter, cold summer'},
    24: {'code': 'Dwd', 'description': 'Cold, dry winter, very cold winter'},
    25: {'code': 'Dfa', 'description': 'Cold, no dry season, hot summer'},
    26: {'code': 'Dfb', 'description': 'Cold, no dry season, warm summer'},
    27: {'code': 'Dfc', 'description': 'Cold, no dry season, cold summer'},
    28: {'code': 'Dfd', 'description': 'Cold, no dry season, very cold winter'},
    29: {'code': 'ET',  'description': 'Polar, tundra'},
    30: {'code': 'EF',  'description': 'Polar, frost'},
}

KG_MAJOR = {
    1: {'description': 'Tropical',              'code': 'A',  'color': "#e7b5ad", 'ids': [1, 2, 3]},
    2: {'description': 'Arid',                   'code': 'B',  'color': "#d7dab7", 'ids': [4, 5, 6, 7]},
    3: {'description': 'Temp. (dry summer)',     'code': 'Cs', 'color': "#9dc8ae", 'ids': [8, 9, 10]},
    4: {'description': 'Temp. (dry winter)',     'code': 'Cw', 'color': "#acc2bd", 'ids': [11, 12, 13]},
    5: {'description': 'Temp. (no dry season)',  'code': 'Cf', 'color': "#a7dfe5", 'ids': [14, 15, 16]},
    6: {'description': 'Cold (dry summer)',      'code': 'Ds', 'color': "#9fc0e3", 'ids': [17, 18, 19, 20]},
    7: {'description': 'Cold (dry winter)',      'code': 'Dw', 'color': "#ceb7dd", 'ids': [21, 22, 23, 24]},
    8: {'description': 'Cold (no dry season)',   'code': 'Df', 'color': "#d6ccdb", 'ids': [25, 26, 27, 28]},
    # Polar (29, 30) intentionally excluded, as in those notebooks -- not present among GRDC gauges anyway.
}
KG_MAJOR_NAMES = {k: v['code'] for k, v in KG_MAJOR.items()}
KG_MAJOR_COLORS = {k: v['color'] for k, v in KG_MAJOR.items()}
KG_TO_MAJOR = {fine_id: major_id for major_id, v in KG_MAJOR.items() for fine_id in v['ids']}


def load_kg_major_grid(downsample=4):
    """Global Koppen-Geiger grid (fine class -> 8 major groups), for use as a light background
    layer behind basin scatter maps. Native resolution is 0.1 deg (3600x1800); downsampled by
    simple slicing (map-background use only, not for analysis) since full resolution is overkill
    at map-figure scale and considerably slower to render.

    Returns (lon [1D], lat [1D], major_id grid [lat, lon], NaN where no/polar class).
    """
    import xarray as xr

    kg = xr.open_dataset(KG_NC_PATH)['kg_class']
    if downsample > 1:
        kg = kg.isel(lon=slice(None, None, downsample), lat=slice(None, None, downsample))

    lut = np.full(31, np.nan)
    for fine_id, major_id in KG_TO_MAJOR.items():
        lut[fine_id] = major_id

    fine_vals = kg.values
    valid = np.isfinite(fine_vals) & (fine_vals >= 1) & (fine_vals <= 30)
    major_vals = np.full(fine_vals.shape, np.nan)
    major_vals[valid] = lut[fine_vals[valid].astype(int)]

    return kg.lon.values, kg.lat.values, major_vals


def _add_cover_fractions(df):
    """Aggregate the 22 GLC land-cover classes into forest, shrub, grass, crop, and other fractions."""
    df = df.copy()
    df['cover_forest'] = df[[f'glc_pc_s{str(i).zfill(2)}' for i in range(1, 11)]].sum(axis=1)
    df['cover_shrub']  = df[[f'glc_pc_s{str(i).zfill(2)}' for i in [11, 12, 15]]].sum(axis=1)
    df['cover_grass']  = df[[f'glc_pc_s{str(i).zfill(2)}' for i in [13, 14]]].sum(axis=1)
    df['cover_crop']   = df[[f'glc_pc_s{str(i).zfill(2)}' for i in [16, 17, 18]]].sum(axis=1)
    df['cover_others'] = df[[f'glc_pc_s{str(i).zfill(2)}' for i in [19, 20, 21, 22]]].sum(axis=1)
    return df


def _classify_aridity(ari_ix_sav):
    """ari_ix_sav is the HydroATLAS aridity index P/PET * 100 (higher = wetter).
    Converted to the PET/P ratio (1/ari_ix_sav)*100, then binned into the five aridity categories."""
    ai = (1.0 / ari_ix_sav) * 100
    conditions = [
        (ai >= 0)   & (ai < 0.5),
        (ai >= 0.5) & (ai <= 1.0),
        (ai >= 1.0) & (ai <= 1.5),
        (ai >= 1.5) & (ai <= 2.5),
        (ai > 2.5),
    ]
    choices = [5, 4, 3, 2, 1]
    return np.select(conditions, choices, default=np.nan), ai


def _lookup_koppen_major(lons, lats):
    """Nearest-neighbor point lookup of each gauge's Koppen-Geiger fine class from the 0.1-deg
    grid, reclassified to the 8 major groups (KG_MAJOR). Returns (major_id array, major_code array)
    with NaN/None for gauges that fall on a nodata/polar/ocean cell."""
    import xarray as xr

    kg = xr.open_dataset(KG_NC_PATH)['kg_class']
    lon_da = xr.DataArray(np.asarray(lons, dtype=float), dims='gauge')
    lat_da = xr.DataArray(np.asarray(lats, dtype=float), dims='gauge')
    fine = kg.sel(lon=lon_da, lat=lat_da, method='nearest').values
    kg.close()

    major_id = np.array([KG_TO_MAJOR.get(int(f), np.nan) if np.isfinite(f) else np.nan for f in fine])
    major_code = np.array([KG_MAJOR[int(m)]['code'] if np.isfinite(m) else None for m in major_id])
    return major_id, major_code


def build_basin_tables(valid_ids):
    """Build the coordinate table (df_coord) and the attribute/feature table (df_features) for the
    given catchment ids by joining HydroATLAS, soil, and additional attributes and deriving the
    land-cover fractions, aridity class, and Koppen-Geiger major class used across the analyses."""
    hydro = pd.read_csv(HYDROATLAS_PATH, index_col='gauge_id')
    addl = pd.read_csv(ADDITIONAL_PATH, index_col='gauge_id')
    soil = pd.read_csv(SOIL_PATH, index_col='gauge_id')

    df = hydro.join(soil, how='left').join(
        addl[['lat_pp', 'long_pp', 'country', 'area_shp']], how='left')  # area_shp: catchment area (km2)
    df = df.loc[df.index.intersection(valid_ids)].reindex(valid_ids)

    df = _add_cover_fractions(df)
    df['aridity_category'], df['ari_ix_conv'] = _classify_aridity(df['ari_ix_sav'])
    df['kg_major_id'], df['kg_major_code'] = _lookup_koppen_major(df['long_pp'], df['lat_pp'])

    df_features = df.reset_index().rename(columns={'index': 'gauge_id'})
    if 'gauge_id' not in df_features.columns:
        df_features = df_features.rename(columns={df_features.columns[0]: 'gauge_id'})
    df_features['basin_lon'] = df_features['long_pp']
    df_features['basin_lat'] = df_features['lat_pp']

    df_coord = df_features[['gauge_id', 'basin_lon', 'basin_lat', 'aridity_category',
                             'kg_major_id', 'kg_major_code', 'country']].copy()

    return df_coord, df_features


if __name__ == "__main__":
    import pickle
    import sys
    pickle_path = sys.argv[1] if len(sys.argv) > 1 else None
    if pickle_path:
        with open(pickle_path, 'rb') as f:
            d = pickle.load(f)
        valid_ids = d['valid_ids']
    else:
        raise SystemExit("usage: python basin_attributes.py <averaged_full_output.pickle>")

    df_coord, df_features = build_basin_tables(valid_ids)
    print(df_coord.head())
    print("n basins:", len(df_coord), " missing coords:", df_coord['basin_lon'].isna().sum())
    print(df_coord['aridity_category'].value_counts().sort_index())
