# Baseline: original MOREDO physical model with NO lateral-flow / F_add term at all.
# 'lat_alpha' is absent from PARAM_CONFIG entirely, so avai_bucket's
# params.get('lat_alpha', 0.0) defaults to 0 and f_lat is identically zero everywhere.
# Used as the reference "static/instantaneous/recurrent vs. no-subsidy-at-all" column in
# performance_comparison.png.
from config_global_spatial import *
