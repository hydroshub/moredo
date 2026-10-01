"""
Figs. S5 and S6 from fig_si_repr_heldout.npz (fig_si_repr_heldout.py) and the averaged pickles.
Light (no torch inference), safe on the login node.

S5  Representation comparison: 2001-2020 ensemble-mean catchment-mean F_add for the static
    (mlp_a), instantaneous (mlp_af) and recurrent (lstm_af) encoders.
S6  Strictly held-out recurrent F_add (test-fold-only estimates), the held-out control of Fig. 5.
    Its spatial correlation with the Fig. 5 map (mean of the 25 model instances, from
    fig_si_stability.npz) is printed.
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE, EXP_COLORS, format_map, right_colorbar
import cartopy.crs as ccrs
from basin_attributes import build_basin_tables
from residual_utils import ACTIVE_START_YEAR

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")


def _active_mean_fadd(exp):
    """Return the 2001-2020 active-period catchment-mean F_add and valid ids for one experiment."""
    d = pickle.load(open(os.path.join(REPO, f"results_lateral_experiments/{exp}/averaged_full_output.pickle"), "rb"))
    idx = pd.DatetimeIndex(d["idx_daily"]); active = idx.year >= ACTIVE_START_YEAR
    return np.squeeze(d["fluxes"]["f_lat"])[:, active].mean(1), list(d["valid_ids"])


def main():
    """Render Figs. S5 and S6: the three representation maps and the strictly held-out F_add map."""
    d = np.load(os.path.join(OUT_DIR, "fig_si_repr_heldout.npz"), allow_pickle=True)
    fadd_static = d["fadd_static"]; fadd_heldout = d["fadd_heldout"]; vids = list(d["gauge_id"])
    fadd_inst, _ = _active_mean_fadd("mlp_af")
    fadd_rec, _ = _active_mean_fadd("lstm_af")
    dfc, dff = build_basin_tables(vids)
    lon, lat = dfc["basin_lon"].values, dfc["basin_lat"].values

    # ---------- S5: three representation maps, stacked vertically ----------
    vmax = float(np.nanpercentile(np.concatenate([fadd_static, fadd_inst, fadd_rec]), 95))
    fig = uplt.figure(figwidth="150mm", tight=True, spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=3, ncols=1, hspace=3)
    maps = [("static", fadd_static), ("instantaneous", fadd_inst), ("recurrent", fadd_rec)]
    sc = None
    for i, (name, m) in enumerate(maps):
        ax = fig.add_subplot(gs[i, 0], proj=ccrs.PlateCarree(), abc=True, abcloc="l")
        sc = ax.scatter(lon, lat, c=m, s=8, cmap="Blues", vmin=0, vmax=vmax,
                        edgecolor="grey", linewidth=0.15, transform=ccrs.PlateCarree())
        format_map(ax, title=name)
    fig.colorbar(sc, loc="r", width=0.22, length=0.6, extend="max",
                 label="Mean $F_{add}$ (mm day$^{-1}$)",
                 labelsize=LABELSIZE - 1, ticklabelsize=TICKLABELSIZE - 1)
    out2 = os.path.join(OUT_DIR, "fig_si_representation_maps.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight", pad_inches=0.15); print(f"Saved -> {out2}")

    # ---------- S6: held-out-only recurrent F_add (single map) ----------
    st = np.load(os.path.join(OUT_DIR, "fig_si_stability.npz"), allow_pickle=True)
    assert [str(g) for g in st["gauge_id"]] == [str(g) for g in vids]
    fadd_all25 = np.nanmean(st["inst_maps"], axis=0)  # Fig. 5 map (all 25 instances)
    v = np.isfinite(fadd_heldout) & np.isfinite(fadd_all25)
    r = np.corrcoef(fadd_all25[v], fadd_heldout[v])[0, 1]  # reported in caption/log, not drawn
    vmax3 = float(np.nanpercentile(fadd_heldout[v], 95))
    fig = uplt.figure(figwidth="180mm", tight=True, spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())
    sc = ax.scatter(lon[v], lat[v], c=fadd_heldout[v], s=12, cmap="Blues", vmin=0, vmax=vmax3,
                    edgecolor="grey", linewidth=0.2, transform=ccrs.PlateCarree())
    format_map(ax, title="")
    right_colorbar(fig, ax, sc, "Held-out mean $F_{add}$ (mm day$^{-1}$)", extend="max")
    out3 = os.path.join(OUT_DIR, "fig_si_heldout_subsidy_map.png")
    fig.savefig(out3, dpi=200, bbox_inches="tight", pad_inches=0.25)
    print(f"Saved -> {out3}  (held-out vs 25-instance mean (Fig. 5) spatial r = {r:.3f}, n = {int(v.sum())})")


if __name__ == "__main__":
    main()
