# MOREDO

<img src="moredo_logo.jpg" align="right" width="250" style="padding-left:10px;"/>

**M**ulti-**O**bservation **R**oot-zone **E**cohydrology **D**iagn**O**stic Model.


- fully differentiable hybrid model 🔁
- transfer across scales for global inference (e.g., basin to grid) 🌍
- diagnose 🔍 root-zone ecohydrology 💧🌱

---

## 📁 Directory structure

```
moredo/
├── configs/
│   ├── config_base.py               # Configuration (all settings, with comments)
│   └── spatial_cv_folds_example.py  # Example predefined fold dict for spatial CV
├── libs/
│   ├── model.py                     # HybridModel definition
│   ├── data.py                      # Data loading, caching, CV classes
│   └── utils.py                     # Training, evaluation, loss functions
├── scripts/
│   ├── train.py                     # Training + evaluation entry point
│   ├── inference_basin.py           # Inference on basin time series
│   └── inference_grid.py            # Inference on spatial grid (NetCDF output)
├── requirements.txt
└── CHANGELOG.md
```

---

## ✅ Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

GPU users should install a CUDA-matched `torch` build (see the note in
`requirements.txt`); the code also runs on CPU, which is the default for inference.

---

## 🚀 Running

`scripts/train.py` trains one (seed, fold) model and evaluates it on the held-out test
basins. Run from the repository root with `PYTHONPATH` set so `libs/` is importable:

```bash
PYTHONPATH=. python scripts/train.py \
    --basin_list /path/to/basin_list.txt \
    --config     configs/config_base.py \
    --fold_id    0 \
    --seed       100 \
    --save_dir   results/ \
    --device     cuda:0       # or cpu
```

Inference then reuses the saved checkpoint:

```bash
# Basin time series  → results/results_fold{fold}_seed{seed}.pickle
PYTHONPATH=. python scripts/inference_basin.py \
    --model_dir results/ --fold 0 --seed 100 \
    --config configs/config_base.py --basin_list /path/to/basin_list.txt --device cpu

# Spatial grid       → NetCDF per year under the output dir
PYTHONPATH=. python scripts/inference_grid.py \
    --model_dir results/ --fold 0 --seed 100 \
    --config configs/config_base.py --output_dir results/grid/ --device cpu
```

Training and inference are run per (seed, fold); loop over `--seed` and `--fold_id`
to produce a full ensemble (e.g. 5 seeds × 5 folds).

---

## ⚙️ Configuration

All settings live in `configs/config_base.py`, which is the authoritative, commented
source — the excerpt below is for orientation only. Create a new experiment by
inheriting and overriding:

```python
# configs/config_myexperiment.py
from config_base import *

DATA_DIR   = "/path/to/my/data"
ATTR_PATHS = ["/path/to/my/attributes.csv"]
SPATIAL_CV_SEED = 7
```

Only override what differs — everything else inherits from `config_base.py`.

### Key settings

```python
# Column mappings  (logical name → CSV column name)
DRIVER_COLS_MAP = {
    "P":      "total_precipitation_sum",
    "T":      "temperature_2m_mean",
    "RAD_SW": "surface_net_solar_radiation_mean",
    "RAD_LW": "surface_net_thermal_radiation_mean",
    "LAI":    "lai_GIMMS_filled",
}
TARGET_COLS_MAP = {
    "q":    "streamflow",
    "et":   "fluxcom_E",
    "swe":  "snow_depth_water_equivalent_mean",
    "twsa": "grace_TWSA",   # monthly
}

# Attribute groups → NN inputs as {col}_normed; passthrough kept as raw {col}
VEG_ATTRS     = ["cover_forest", "cover_shrub", "cover_grass",
                 "cover_crop", "cover_others", "lai_avg"]
TERRAIN_ATTRS = ["slp_dg_sav", "ele_mt_sav", "new_cly_pc",
                 "new_snd_pc", "snw_pc_syr", "ari_ix_sav"]
ATTR_NN_COLS          = VEG_ATTRS + TERRAIN_ATTRS
ATTR_PASSTHROUGH_COLS = ["lai_avg", "lai_max", "lai_min", "ele_mt_sav"]
# Note: "ele" columns are sqrt-transformed before normalization.

# Parameter configuration — param_name: ([low, high], mode, attrs)
#   mode = 'static'  →  MLP(attrs_normed), time-invariant per basin/pixel
PARAM_CONFIG = {
    "snow_tsnow":      ([-5.0, 0.0],    'static',  TERRAIN_ATTRS),
    "snow_train":      ([0.0,  5.0],    'static',  TERRAIN_ATTRS),
    "snow_fmt":        ([0.5,  8.0],    'static',  TERRAIN_ATTRS),
    "split_k":         ([0.1,  10.0],   'static',  VEG_ATTRS + TERRAIN_ATTRS),
    "avai_cap_base":   ([0.0,  2000.0], 'static',  VEG_ATTRS + TERRAIN_ATTRS),
    "avai_wetpoint":   ([0.01, 0.99],   'static',  VEG_ATTRS + TERRAIN_ATTRS),
    "avai_stress_exp": ([0.1,  1.0],    'static',  VEG_ATTRS + TERRAIN_ATTRS),
    "avai_beta":       ([0.05, 0.95],   'static',  VEG_ATTRS + TERRAIN_ATTRS),
    "fast_kf":         ([0.05, 0.95],   'static',  TERRAIN_ATTRS),
    "fast_perc":       ([0.1,  20.0],   'static',  TERRAIN_ATTRS),
    "slow_ks":         ([1e-4, 1e-1],   'static',  TERRAIN_ATTRS),
}

# avai_cap mode: "informed" = avai_cap_base · f(LAI) · g(slope) + 50;
#                "direct"   = cap learned directly (see config_base.py for details)
AVAI_CAP_MODE = "informed"

# Training
BATCH_SIZE = 2048; NUM_EPOCHS = 100; LEARNING_RATE = 5e-2
EARLY_STOP_PATIENCE = 10; LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  # [q, et, swe, twsa]
CV_MODE = "spatial"; NUM_FOLDS = 5
SPATIAL_CV_METHOD = "random"   # "random" or path to a .py defining FOLD_DICT
SPATIAL_CV_SEED   = 42

# Grid inference
INFER_SPINUP_YEAR   = 1995   # None = cold start; else cycle this year before INFER_START_YEAR
INFER_SPINUP_CYCLES = 5
INFER_START_YEAR    = 1996
INFER_END_YEAR      = 2020
```

Normalization (`ATTR_NORM`) uses fixed offset/scaler pairs rather than scalers fitted
to the training data, which keeps inference well-behaved when extrapolating to unseen
regions in global grid runs.

### Grid-inference units

Grid drivers are fed to the physics in the **same units as the training CSVs**:
precipitation and snowmelt in mm/day, temperature in °C, net radiation in W/m². The
`INFER_DRIVER_PATHS` files must already be in these units — no unit conversion is
applied when reading the NetCDFs. Raw ERA5-Land variables (`tp` in m, `t2m` in K,
`ssr`/`str` in J/m²) must be converted beforehand.

---

## 🤝 Contact

Ideas, questions or bugs?! We would love to hear from you!
- **Issues / feature requests** → [open an issue](https://github.com/hydroshub/moredo/issues)
- **Reach us directly** →  The model is developed by the [ML4HES](https://www.bgc-jena.mpg.de/en/bgi/ml4hes) group at the Max Planck Institute for Biogeochemistry and ELLIS Unit Jena. Feel free to reach out!

---

## 📝 Citation

_Preprint in preparation — citation details to follow._

---

## License

MOREDO is free and open-source software, licensed under the [European Union Public Licence v1.2 (EUPL)](./LICENSE).

You are free to copy, modify, and redistribute the code, and to use it in both commercial and non-commercial contexts. If you redistribute a modified version — excluding changes made solely for interoperability — you must do so under the EUPL v1.2 or a compatible licence.

This software is provided in the hope that it will be useful, but **without any warranty**, including, without limitation, the implied warranties of merchantability or fitness for a particular purpose.

---
**Copyright © 2026** Max Planck Institute for Biogeochemistry