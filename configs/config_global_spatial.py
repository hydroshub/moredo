from config_base import *

DATA_DIR = "/path/to/GRDC_Caravan/processed_merged/"
ATTR_PATHS = [
    "/path/to/GRDC_Caravan/attributes/grdc/attributes_hydroatlas_grdc.csv",
    "/path/to/GRDC_Caravan/attributes/grdc/attributes_additional_grdc.csv",
    "/path/to/GRDC_Caravan/attributes/grdc/attributes_soil_grdc.csv",    
]


PARAM_CONFIG = {
    "snow_tsnow":      ([-5.0, 0.0],    'static',  TERRAIN_ATTRS),             # snowfall temperature threshold (°C)
    "snow_train":      ([0.0,  5.0],    'static',  TERRAIN_ATTRS),             # rainfall temperature threshold (°C)
    "snow_fmt":        ([0.5,  8.0],    'static',  TERRAIN_ATTRS),             # melt factor (mm/°C/day)
    "split_k":         ([0.1,  10.0],   'static',  VEG_ATTRS + TERRAIN_ATTRS), # available-water split shape (1=linear, >1=convex, <1=concave)
    "avai_cap_base":   ([0.0,  2000.0], 'static',  VEG_ATTRS + TERRAIN_ATTRS), # available water capacity base (mm)
    "avai_wetpoint":   ([0.01, 0.99],   'static',  VEG_ATTRS + TERRAIN_ATTRS), # relative soil moisture at which ET reaches PET
    "avai_stress_exp": ([0.1,  1.0],    'static',  VEG_ATTRS + TERRAIN_ATTRS), # f_wetness stress exponent (<1 convex: wet areas stay near 1)
    "avai_beta":       ([0.05, 0.95],   'static',  VEG_ATTRS + TERRAIN_ATTRS), # LAI phenology response curve
    "fast_kf":         ([0.05, 0.95],   'static',  TERRAIN_ATTRS),             # fast-flow recession coefficient
    "fast_perc":       ([0.1,  20.0],   'static',  TERRAIN_ATTRS),             # percolation rate (mm/day)
    "slow_ks":         ([1e-4, 1e-1],   'static',  TERRAIN_ATTRS),             # slow-flow recession rate
}

TARGET_COLS_MAP = {
    "q":    "streamflow",
    "et":   "fluxcom_E",
    "swe":  "snow_depth_water_equivalent_mean", # change nsidc_SWE to snow_depth_water_equivalent_mean
    "twsa": "grace_TWSA",
}

TARGET_DAILY_COLS   = [v for k, v in TARGET_COLS_MAP.items() if k != "twsa"]
TARGET_MONTHLY_COLS = [TARGET_COLS_MAP["twsa"]]