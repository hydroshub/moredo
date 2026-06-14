# -----------------------------------------------------------------------------
# Data paths
# -----------------------------------------------------------------------------
DATA_DIR = "/path/to/GRDC_Caravan/processed_merged/"
ATTR_PATHS = [
    "/path/to/GRDC_Caravan/attributes/grdc/attributes_hydroatlas_grdc.csv",
    "/path/to/GRDC_Caravan/attributes/grdc/attributes_additional_grdc.csv",
    "/path/to/GRDC_Caravan/attributes/grdc/attributes_soil_grdc.csv",    
]

# -----------------------------------------------------------------------------
# Column mappings
#   DRIVER_COLS_MAP  : logical driver name  → CSV column (time-varying)
#   ATTR_COLS_MAP    : logical attr name    → CSV column (static, named lookup)
#   TARGET_COLS_MAP  : logical target name  → CSV column
# -----------------------------------------------------------------------------
DRIVER_COLS_MAP = {
    "P":      "total_precipitation_sum",
    "T":      "temperature_2m_mean",
    "RAD_SW": "surface_net_solar_radiation_mean",
    "RAD_LW": "surface_net_thermal_radiation_mean",
    "LAI":    "lai_GIMMS_filled",
}

ATTR_COLS_MAP = {
    "SLOPE":   "slp_dg_sav",
    "LAI_AVG": "lai_avg",   # long-term mean LAI  (also an NN input via ATTR_NN_COLS)
    "LAI_MAX": "lai_max",   # long-term 99th-pct LAI (passthrough only)
    "LAI_MIN": "lai_min",   # long-term  1st-pct LAI (passthrough only)
}

TARGET_COLS_MAP = {
    "q":    "streamflow",
    "et":   "fluxcom_E",
    "swe":  "snow_depth_water_equivalent_mean",
    "twsa": "grace_TWSA",
}

# Derived — do not edit directly
INPUT_COLS          = list(DRIVER_COLS_MAP.values())
TARGET_DAILY_COLS   = [v for k, v in TARGET_COLS_MAP.items() if k != "twsa"]
TARGET_MONTHLY_COLS = [TARGET_COLS_MAP["twsa"]]

# -----------------------------------------------------------------------------
# Time range
#   SPINUP_START_YEAR : data loading begins here (spinup period)
#   TRAIN_START_YEAR  : spinup ends, CV period begins (= spinup_until)
#   TRAIN_END_YEAR    : data loading ends here
# -----------------------------------------------------------------------------
SPINUP_START_YEAR = 1996
TRAIN_START_YEAR  = 2001
TRAIN_END_YEAR    = 2020

# -----------------------------------------------------------------------------
# Attribute groups
#
#   VEG_ATTRS      : vegetation attributes (land cover fractions + mean LAI)
#   TERRAIN_ATTRS  : terrain / soil / climate attributes
#
#   ATTR_NN_COLS          : all attrs entering the NN as {col}_normed
#   ATTR_PASSTHROUGH_COLS : attrs kept as raw {col} in x for model formula use
#                           (lai_avg is in both — gets lai_avg_normed for NN
#                            AND lai_avg raw for the avai_cap formula)
#   ATTR_COLS             : full list of raw columns to load from CSV (no duplicates)
# -----------------------------------------------------------------------------
VEG_ATTRS = [
    "cover_forest",
    "cover_shrub",
    "cover_grass",
    "cover_crop",
    "cover_others",
    "lai_avg",
]

TERRAIN_ATTRS = [
    "slp_dg_sav",
    "ele_mt_sav",
    "new_cly_pc",
    "new_snd_pc",
    "snw_pc_syr",
    "ari_ix_sav",
]

ATTR_NN_COLS = VEG_ATTRS + TERRAIN_ATTRS

ATTR_PASSTHROUGH_COLS = [
    "lai_avg",
    "lai_max",
    "lai_min",
    "ele_mt_sav",
]

ATTR_COLS = ATTR_NN_COLS + [c for c in ATTR_PASSTHROUGH_COLS if c not in ATTR_NN_COLS]

# -----------------------------------------------------------------------------
# Parameter configuration
#   Each entry: param_name → ([low, high], mode, attrs)
#     mode = 'static'                → MLP(attrs_normed) — time-invariant
# -----------------------------------------------------------------------------
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

# avai_cap calculation mode:
#   "informed" — cap = avai_cap_base * f(LAI) * g(slope) + 50   (uses the
#                "avai_cap_base" entry in PARAM_CONFIG above)
#   "direct"   — cap is learned directly from attributes, with no LAI/slope
#                modulation → see Blougouras et al. WRR '26.
#                To use it, REPLACE the "avai_cap_base" entry in PARAM_CONFIG
#                with one named "avai_cap", e.g.:
#                  "avai_cap": ([50.0, 2000.0], 'static', VEG_ATTRS + TERRAIN_ATTRS),
#                (the model reads params['avai_cap'] directly in this mode —
#                 keeping only "avai_cap_base" will raise KeyError: 'avai_cap')
AVAI_CAP_MODE = "informed"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
BATCH_SIZE          = 2048
NUM_EPOCHS          = 100   # overridable via train.py --num_epochs
LEARNING_RATE       = 5e-2
EARLY_STOP_PATIENCE = 10
LOSS_WEIGHTS        = [1.0, 1.0, 1.0, 1.0]  # [q, et, swe, twsa]
CV_MODE             = "spatial"             
NUM_FOLDS           = 5
SPATIAL_CV_METHOD   = "random"              # "random"  or  path to a .py file defining FOLD_DICT
SPATIAL_CV_SEED     = 42                    # controls basin fold assignments — keep fixed across training seeds

# -----------------------------------------------------------------------------
# Inference
#   INFER_START_YEAR / INFER_END_YEAR : inference time range (can differ from training)
#   INFER_SAVE_DAILY   : variables saved at daily resolution   (None = all, [] = skip)
#   INFER_SAVE_MONTHLY : variables saved at monthly resolution (None = all, [] = skip)
#   Variable names follow the format "{group}_{varname}", e.g. "fluxes_et", "states_S_AVAI"
# -----------------------------------------------------------------------------
INFER_SPINUP_YEAR   = 1995  # None = no spinup; set to e.g. 1995 to cycle that year before INFER_START_YEAR
INFER_SPINUP_CYCLES = 5

INFER_START_YEAR = 1996  # first year with saved output; if using spinup, set to INFER_SPINUP_YEAR + 1
INFER_END_YEAR   = 2020

INFER_SAVE_DAILY = []

INFER_SAVE_MONTHLY = None

# Static attribute file for grid inference
INFER_ATTR_PATH = "/path/to/attributes/combined_variables.nc"

# Driver file path templates — keys match DRIVER_COLS_MAP logical names
# Each entry is either:
#   "path_template"             — file is a single-variable DataArray
#   ("path_template", "varname")— file is a Dataset; varname selects the variable
# Use {year} and {month:02d} as placeholders in path_template.
INFER_DRIVER_PATHS = {
    "P":      "/path/to/ERA5_Land/tp/{year}{month:02d}.nc",
    "T":      "/path/to/ERA5_Land/t2m/{year}{month:02d}.nc",
    "RAD_SW": "/path/to/ERA5_Land/ssr/{year}{month:02d}.nc",
    "RAD_LW": "/path/to/ERA5_Land/str/{year}{month:02d}.nc",
    "LAI":    ("/path/to/LAI_gimms4g_0d25/LAI_daily_{year}_{month:02d}.nc", "LAI"),
}


# -----------------------------------------------------------------------------
# Fixed normalization: x_norm = (x - offset) / scaler
# Not fitted to training data — better for global extrapolation.
# -----------------------------------------------------------------------------

# Static NN attributes (ATTR_NN_COLS = VEG_ATTRS + TERRAIN_ATTRS)
ATTR_NORM = { # based on HydroATLAS
    # VEG_ATTRS
    "cover_forest":  (0.0, 100.0),   # fraction [0, 1]
    "cover_shrub":   (0.0, 100.0),   # fraction [0, 1]
    "cover_grass":   (0.0, 100.0),   # fraction [0, 1]
    "cover_crop":    (0.0, 100.0),   # fraction [0, 1]
    "cover_others":  (0.0, 100.0),   # fraction [0, 1]
    "lai_avg":       (1.2, 1.4),     # m²/m²
    # TERRAIN_ATTRS
    "slp_dg_sav":    (46.7, 53.9),   # degree * 10
    "ele_mt_sav":    (21.4, 13.2),   # sqrt_m (NOTE: sqrt is applied to elevation columns before normalization in AttributeProcessor.transform())
    "new_cly_pc":    (26.9, 8.8),    # % clay
    "new_snd_pc":    (42.9, 12.4),   # % sand
    "snw_pc_syr":    (27.1, 31.7),   # % snow cover
    "ari_ix_sav":    (106.7, 177.8), # aridity index * 100
}
