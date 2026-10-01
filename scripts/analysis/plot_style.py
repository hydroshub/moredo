"""Shared ultraplot/seaborn styling of the manuscript figures."""
import os
import sys

# PROJ database of the active (conda) environment, needed by cartopy
_PROJ_DIR = os.path.join(sys.prefix, "share", "proj")
if os.path.isdir(_PROJ_DIR):
    os.environ.setdefault("PROJ_DATA", _PROJ_DIR)
    os.environ.setdefault("PROJ_LIB", _PROJ_DIR)

import ultraplot as uplt
import seaborn as sns
import cartopy.feature as cfeature

uplt.rc.style = "seaborn-v0_8-paper"
uplt.rc.update({
    "tick.len": 3.0,
    "tick.dir": "in",
    "font.largesize": 14,
    "font.smallsize": 12,
    "mathtext.it": "sans",
})
# NOTE: deliberately NOT calling sns.set_theme() — it overrides matplotlib rcParams
# globally and conflicts with ultraplot's own layout engine (caused severe panel/text
# overlap). seaborn plotting functions (sns.boxplot etc.) work fine without it.

LABELSIZE = 15
TICKLABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 12

CATEGORY_NAMES = {1: "arid", 2: "semi-arid", 3: "semi-humid", 4: "humid", 5: "hyper-humid"}
CATEGORY_COLORS = {1: "#8c510a", 2: "#d8b365", 3: "#808080", 4: "#5ab4ac", 5: "#01665e"}
EXP_LABELS = {"baseline": "baseline", "mlp_a": "MLP(A)", "mlp_af": "MLP(A,F)", "lstm_af": "LSTM(A,F)"}
EXP_COLORS = {"baseline": "#7f7f7f", "mlp_a": "#d7191c", "mlp_af": "#1a9641", "lstm_af": "#2c7bb6"}


def format_map(ax, title="", lonlim=(-180, 180), latlim=(-60, 85)):
    """Apply the shared map style to a GeoAxes: coastlines and borders, fixed extent, no frame or grid."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.format(title=title, titlesize=TITLESIZE, lonlim=lonlim, latlim=latlim,
              labels=False, grid=False, linewidth=0)
    for s in ax.spines.values():
        s.set_visible(False)


def right_colorbar(fig, ax, sc, label, width=0.15, length=0.8, align=None,
                   labelsize=None, ticklabelsize=None, extend=None):
    """Attach a right-hand colorbar to ax with the shared width, length, and font sizes."""
    kw = dict(align=align) if align is not None else {}
    if extend is not None:
        kw["extend"] = extend
    ax.colorbar(sc, loc="r", width=width, length=length, label=label,
                labelsize=LABELSIZE - 1 if labelsize is None else labelsize,
                ticklabelsize=TICKLABELSIZE - 1 if ticklabelsize is None else ticklabelsize, **kw)
