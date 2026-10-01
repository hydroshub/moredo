"""
Locations used by the analysis scripts, defined in one place so no script carries a
machine-specific path.

  REPO         repository root, found as the nearest parent directory of this file that
               contains libs/ (this file sits in <repo>/scripts/analysis/)
  RESULTS_DIR  trained models, averaged outputs and cached arrays, <repo>/results_lateral_experiments
               (cached arrays in its figures/ subdirectory)
  DATA_ROOT    input data, default <repo>/data, expected to contain
                 GRDC_Caravan/attributes/grdc/attributes_{hydroatlas,additional,soil,morphology}_grdc.csv
                 KG_Beck23/koppen_geiger_nc/1991_2020/koppen_geiger_0p1.nc
  BASIN_LIST   the study catchments, <repo>/configs/basin_list_global_grdc.txt

DATA_ROOT and BASIN_LIST can be set for a specific machine in an optional project_paths_local.py
next to this file (not part of the release), or with the environment variables MOREDO_DATA_ROOT
and MOREDO_BASIN_LIST, which take precedence.
"""
import os


def _find_repo():
    """Nearest parent directory of this file that contains libs/."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "libs")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project_paths: no parent directory with libs/ found")
        d = parent


REPO = _find_repo()
RESULTS_DIR = os.path.join(REPO, "results_lateral_experiments")
DATA_ROOT = os.path.join(REPO, "data")
BASIN_LIST = os.path.join(REPO, "configs", "basin_list_global_grdc.txt")

try:
    import project_paths_local as _local
    DATA_ROOT = getattr(_local, "DATA_ROOT", DATA_ROOT)
    BASIN_LIST = getattr(_local, "BASIN_LIST", BASIN_LIST)
except ImportError:
    pass
DATA_ROOT = os.environ.get("MOREDO_DATA_ROOT", DATA_ROOT)
BASIN_LIST = os.environ.get("MOREDO_BASIN_LIST", BASIN_LIST)

GRDC_ATTR_DIR = os.path.join(DATA_ROOT, "GRDC_Caravan", "attributes", "grdc")
KG_NC_PATH = os.path.join(DATA_ROOT, "KG_Beck23", "koppen_geiger_nc", "1991_2020", "koppen_geiger_0p1.nc")
