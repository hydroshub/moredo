# Relaxed-bound sensitivity test: recurrent LSTM(A,F) with the capacity-normalized subsidy
# coefficient bound doubled from [0, 0.01] to [0, 0.02]. Everything else is identical to
# config_lateral_lstm_af.py. Used to check whether the learned F_add moves when the bound moves
# (bound-insensitivity check for the [0, 0.01] range).
from config_global_spatial import *

PARAM_CONFIG = {
    **PARAM_CONFIG,
    "lat_alpha": ([0.0, 0.02], 'dynamic_lstm_state', VEG_ATTRS + TERRAIN_ATTRS),
}
