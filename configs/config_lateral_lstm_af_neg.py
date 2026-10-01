# Recurrent latent-subsidy model (lstm_af) with a SIGNED subsidy coefficient.
#
# The capacity-normalized coefficient psi (lat_alpha) is allowed to be negative as well as positive,
# so the latent flux F_add = psi * avai_cap can act as a lateral LOSS as well as a gain. The bound is
# widened symmetrically from the non-negative [0, 0.01] to [-0.01, 0.01] day^-1; the parameter map
# sigmoid(x)*(high-low)+low then spans negative values with 0 as the neutral midpoint.
#
# A negative F_add is capped inside the model (libs/model.py:avai_bucket) so it can drain at most the
# water currently in the ecosystem-accessible bucket, never below zero, keeping ET non-negative. That
# guard is inactive for the default non-negative lat_alpha, so all other experiments are unaffected.
#
# Everything else (architecture, drivers, attributes, basin list, seeds/folds, CV) is identical to
# config_lateral_lstm_af, so the run is directly comparable to the main recurrent model.

from config_lateral_lstm_af import *

PARAM_CONFIG = {
    **PARAM_CONFIG,
    "lat_alpha": ([-0.01, 0.01], 'dynamic_lstm_state', VEG_ATTRS + TERRAIN_ATTRS),
}
