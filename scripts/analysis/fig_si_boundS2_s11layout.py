"""
Fig. S2: sensitivity of the learned subsidy to the coefficient bound, two square panels
(figwidth 200mm x figheight 92mm, the same footprint as Fig. S10, fig_morph_geometry.py).
  (a) catchment-mean F_add of the recurrent representation under the default bound [0, 0.01]
      against the doubled bound [0, 0.02], coloured by Koppen-Geiger class, with the 1:1 line and
      Pearson r.
  (b) distribution of the learned subsidy coefficient alpha = F_add/theta_cap, as a fraction of the
      imposed bound, for the three encoders re-trained with the doubled bound [0, 0.02]
      (mlp_a_cap002, mlp_af_cap002, lstm_af_cap002), linear y-scale.
Reads the averaged outputs of lstm_af, lstm_af_cap002, mlp_a_cap002 and mlp_af_cap002.

Usage: python scripts/analysis/fig_si_boundS2_s11layout.py
"""
import os
import sys
import pickle

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE
from residual_utils import ACTIVE_START_YEAR
from basin_attributes import build_basin_tables, KG_MAJOR_NAMES, KG_MAJOR_COLORS
from fadd_bound_sensitivity import load_fadd, EXP_A, EXP_B

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT = os.path.join(RD, "figures", "fig_si_boundS2_s11layout.png")

CONFIGS_002 = [
    ("mlp_a_cap002", "Static", "#d7191c", 0.02),
    ("mlp_af_cap002", "Instantaneous", "#1a9641", 0.02),
    ("lstm_af_cap002", "Recurrent", "#2c7bb6", 0.02),
]


def _panel_a(ax):
    """Figure S2, unchanged: recurrent F_add, bound [0,0.01] vs [0,0.02], by KG class."""
    fa, ids_a = load_fadd(EXP_A)
    fb, ids_b = load_fadd(EXP_B)
    if ids_a != ids_b:
        common = [i for i in ids_a if i in set(ids_b)]
        fa = fa[[ids_a.index(i) for i in common]]
        fb = fb[[ids_b.index(i) for i in common]]
        ids = common
    else:
        ids = ids_a
    ma, mb = np.nanmean(fa, axis=1), np.nanmean(fb, axis=1)

    dfc, _ = build_basin_tables(list(ids))
    kg = dfc["kg_major_id"].values

    v = np.isfinite(ma) & np.isfinite(mb)
    r, _ = stats.pearsonr(ma[v], mb[v])
    hi = float(np.nanpercentile(np.concatenate([ma[v], mb[v]]), 99.5)) * 1.05
    present = [g for g in sorted(KG_MAJOR_NAMES) if np.any(kg == g)]

    for g in present:
        m = (kg == g) & v
        ax.scatter(ma[m], mb[m], s=20, color=KG_MAJOR_COLORS[g], edgecolor="k",
                  linewidth=0.2, alpha=0.85, zorder=3)
    ax.plot([0, hi], [0, hi], color="k", ls="--", lw=1.0, zorder=2)
    ax.text(0.96, 0.04, f"r = {r:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=TICKLABELSIZE - 1)
    ax.format(
        xlabel="$F_{add}$ (mm day$^{-1}$), bound [0, 0.01]",
        ylabel="$F_{add}$ (mm day$^{-1}$), bound [0, 0.02]",
        xlim=(0, hi), ylim=(0, hi),
        xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
        xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False,
    )
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)

    handles = [Line2D([], [], marker="o", linestyle="None", markersize=7,
                      markerfacecolor=KG_MAJOR_COLORS[g], markeredgecolor="k",
                      markeredgewidth=0.2, label=KG_MAJOR_NAMES[g]) for g in present]
    ax.legend(handles, [h.get_label() for h in handles], loc="b",
              ncols=4, frame=False, prop={"size": LEGENDSIZE - 3})
    print(f"  (a) r = {r:.3f}   mean F_add: {np.nanmean(ma):.4f} vs {np.nanmean(mb):.4f} mm/day  (n={v.sum()})")


def coeff_fraction(exp, cap):
    """Learned capacity-normalized coefficient (alpha = F_add/theta_cap) as a fraction of the
    imposed bound, for one experiment. Matches fig_si_coef_bound.coeff_fraction()."""
    r = pickle.load(open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb"))
    la = np.squeeze(np.asarray(r["params"]["lat_alpha"]))
    if la.ndim == 2 and la.shape[1] == len(r["idx_daily"]):
        # daily coefficients: keep the 2001-2020 analysis period, drop the 1996-2000 spin-up
        active = np.asarray([d.year >= ACTIVE_START_YEAR for d in r["idx_daily"]])
        la = la[:, active]
    la = la[np.isfinite(la)]
    return la / cap


def _panel_b(ax):
    """New: alpha distribution for the doubled bound [0, 0.02], linear y-scale."""
    bins = np.linspace(0, 1, 61)
    for exp, label, color, cap in CONFIGS_002:
        frac = coeff_fraction(exp, cap)
        ax.hist(frac, bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=0.35, lw=0.0, zorder=2)
        ax.hist(frac, bins=bins, density=True, histtype="step",
                color=color, lw=1.6, zorder=3, label=label)
        print(f"  (b) {label:14s} median={np.median(frac)*100:5.1f}%  "
              f"p99={np.percentile(frac, 99)*100:5.1f}%  max={frac.max()*100:5.1f}%")
    ax.axvline(1.0, color="k", ls="--", lw=1.2, zorder=4)
    ax.text(0.995, 0.97, "imposed bound", transform=ax.get_xaxis_transform(),
            rotation=90, ha="right", va="top", fontsize=TICKLABELSIZE - 2, color="k")
    ax.format(yscale="linear", xlim=(0, 1.03),
              xlabel=r"Coefficient $\alpha$ (fraction of bound)",
              ylabel="Probability density",
              xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
              xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False)
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    ax.legend(loc="b", ncols=3, frame=False, prop={"size": LEGENDSIZE - 3})


def main():
    fig = uplt.figure(figwidth="200mm", figheight="92mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=1, ncols=2, wspace=0.05, wratios=(1, 1))

    axa = fig.add_subplot(gs[0, 0], abc=True, abcloc="ul")
    _panel_a(axa)

    axb = fig.add_subplot(gs[0, 1], abc=True, abcloc="ul")
    _panel_b(axb)

    # The one-line y-label of (a) is longer than the square axes and overhangs the top edge, which
    # get_tightbbox leaves out. Save with the tight box extended upward to include the full label.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    label = axa.yaxis.label.get_window_extent(rend).transformed(fig.dpi_scale_trans.inverted())
    bb = Bbox.union([fig.get_tightbbox(rend), label])
    fig.savefig(OUT, dpi=200, bbox_inches=Bbox.from_extents(bb.x0, bb.y0, bb.x1, bb.y1 + 0.05))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
