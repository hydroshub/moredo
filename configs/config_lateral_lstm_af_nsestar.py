# Recurrent latent-subsidy model (config_lateral_lstm_af.py) trained with the basin-normalized
# NSE* loss (Kratzert et al., 2019) instead of the pooled NSE. Everything else is unchanged.
# eps per target and the valid-basin rule are set in config_base (NSE_STAR_EPS, NSE_MIN_OBS).
# First trained 2026-09-28 (lstm_af_nsestar).
from config_lateral_lstm_af import *

LOSS_TYPE = "nse_star"
