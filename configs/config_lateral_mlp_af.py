# Experiment 2: lateral flow signal MLP(A,F) — memoryless MLP(attrs, drivers) -> lat_alpha.
from config_global_spatial import *

PARAM_CONFIG = {
    **PARAM_CONFIG,
    "lat_alpha": ([0.0, 0.01], 'dynamic_driver_state', VEG_ATTRS + TERRAIN_ATTRS),
}
