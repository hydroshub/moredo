# Custom spatial CV fold definitions
#
# FOLD_DICT maps each fold index to the list of basin IDs assigned to the TEST set.
# Train and val basins are drawn from the remaining basins (same logic as RandomSpatialCV).
#
# To use this file, set in config_base.py (or any child config):
#   SPATIAL_CV_METHOD = "configs/spatial_cv_folds_example.py"

FOLD_DICT = {
    0: ["camels_01013500", "camels_01022500", "camels_01031500"],
    1: ["camels_01047000", "camels_01052500", "camels_01054200"],
    2: ["camels_01055000", "camels_01057000", "camels_01073000"],
    3: ["camels_01078000", "camels_01118300", "camels_01123000"],
    4: ["camels_01134500", "camels_01137500", "camels_01139000"],
}
