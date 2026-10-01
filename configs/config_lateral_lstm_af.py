# Experiment 3: lateral flow signal LSTM(A,F) — LSTMCell(attrs, drivers) -> lat_alpha.
from config_global_spatial import *

PARAM_CONFIG = {
    **PARAM_CONFIG,
    "lat_alpha": ([0.0, 0.01], 'dynamic_lstm_state', VEG_ATTRS + TERRAIN_ATTRS),
}
