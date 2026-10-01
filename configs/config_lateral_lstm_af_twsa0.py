# Recurrent latent-subsidy model (lstm_af) retrained with the GRACE TWSA loss weight set to zero.
#
# This is the TWSA scale-mismatch sensitivity test (Fig. S16): GRACE mascons are far
# coarser than the 100-2500 km2 catchments, so TWSA is a weak, smeared constraint. Removing it from
# the objective (weight 0) while keeping streamflow, ET and SWE at weight 1 shows how much the TWSA
# term actually drives the learned F_add and the held-out ET/TWSA skill.
#
# Everything else (architecture, drivers, attributes, basin list, seeds/folds, CV) is identical to
# config_lateral_lstm_af, so the run is directly comparable to the main recurrent model.

from config_lateral_lstm_af import *

# LOSS_WEIGHTS order is [q, et, swe, twsa]; drop the TWSA term only.
LOSS_WEIGHTS = [1.0, 1.0, 1.0, 0.0]
