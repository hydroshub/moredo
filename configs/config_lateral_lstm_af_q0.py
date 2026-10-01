# Recurrent latent-subsidy model (lstm_af) retrained with the streamflow (Q) loss weight set to zero.
#
# Companion to the TWSA=0 sensitivity run: here the streamflow term is removed from the objective
# (weight 0) while ET, SWE and TWSA keep weight 1. This shows how much the runoff constraint drives
# the learned F_add and the held-out skill, i.e. whether F_add and the process parameters are shaped
# mainly by matching streamflow.
#
# Everything else (architecture, drivers, attributes, basin list, seeds/folds, CV) is identical to
# config_lateral_lstm_af, so the run is directly comparable to the main recurrent model.

from config_lateral_lstm_af import *

# LOSS_WEIGHTS order is [q, et, swe, twsa]; drop the streamflow term only.
LOSS_WEIGHTS = [0.0, 1.0, 1.0, 1.0]
