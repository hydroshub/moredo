# Relaxed-bound sensitivity test: memoryless MLP(A,F) with the capacity-normalized subsidy
# coefficient bound doubled from [0, 0.01] to [0, 0.02]. Everything else is identical to
# config_lateral_mlp_af.py. Companion of config_lateral_lstm_af_cap002.py for the instantaneous encoder.
from config_global_spatial import *

PARAM_CONFIG = {
    **PARAM_CONFIG,
    "lat_alpha": ([0.0, 0.02], 'dynamic_driver_state', VEG_ATTRS + TERRAIN_ATTRS),
}
