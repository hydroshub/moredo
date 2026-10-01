# Experiment 1: lateral flow signal MLP(A) — static attrs -> lat_alpha (time-invariant).
from config_global_spatial import *

PARAM_CONFIG = {
    **PARAM_CONFIG,
    "lat_alpha": ([0.0, 0.01], 'static', VEG_ATTRS + TERRAIN_ATTRS),
}
