"""
SI figure: distribution of the learned subsidy coefficient alpha = F_add / theta_cap, expressed as
a fraction of the imposed upper bound (0.01 day^-1), for the three encoder configurations. Shows
that the optimizer parks the coefficient well below the ceiling on its own, so the inferred subsidy
is data-constrained rather than bound-constrained. Pre-empts the reviewer question about the range.

Usage:
  python scripts/analysis/fig_si_coef_bound.py
"""
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE
from residual_utils import ACTIVE_START_YEAR

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT = os.path.join(RD, "figures", "fig_si_coef_bound.png")

CAP = 0.01  # imposed upper bound on the coefficient (day^-1)
CONFIGS = [
    ("mlp_a", "Static", "#d7191c"),
    ("mlp_af", "Instantaneous", "#1a9641"),
    ("lstm_af", "Recurrent", "#2c7bb6"),
]


def coeff_fraction(exp):
    """Return the learned capacity-normalized coefficient (alpha = F_add/theta_cap) as a fraction of the imposed bound for one experiment."""
    r = pickle.load(open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb"))
    la = np.squeeze(np.asarray(r["params"]["lat_alpha"]))
    if la.ndim == 2 and la.shape[1] == len(r["idx_daily"]):
        # daily coefficients: keep the 2001-2020 analysis period, drop the 1996-2000 spin-up
        active = np.asarray([d.year >= ACTIVE_START_YEAR for d in r["idx_daily"]])
        la = la[:, active]
    la = la[np.isfinite(la)]
    return la / CAP  # fraction of the bound


def _render(out_path, yscale):
    """Draw the coefficient-fraction histogram for the three encoders on the given y-scale."""
    fig, ax = uplt.subplots(nrows=1, ncols=1, figwidth="150mm", figheight="105mm", share=False)
    bins = np.linspace(0, 1, 61)
    for exp, label, color in CONFIGS:
        frac = coeff_fraction(exp)
        ax.hist(frac, bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=0.35, lw=0.0, zorder=2)
        ax.hist(frac, bins=bins, density=True, histtype="step",
                color=color, lw=1.6, zorder=3,
                label=f"{label}")

    ax.axvline(1.0, color="k", ls="--", lw=1.2, zorder=4)
    ax.text(0.995, 0.97, "imposed bound", transform=ax.get_xaxis_transform(),
            rotation=90, ha="right", va="top", fontsize=TICKLABELSIZE - 2, color="k")

    ax.format(
        yscale=yscale,
        xlim=(0, 1.03), xlabel=r"Learned coefficient $\alpha$ as fraction of the bound",
        ylabel="Probability density", xlabelsize=LABELSIZE, ylabelsize=LABELSIZE,
        xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False,
    )
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="b", ncols=len(CONFIGS), frame=False, prop={"size": LEGENDSIZE - 3})

    fig.save(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


def main():
    """Render Figure S1 (linear y-scale) and print the per-config summary statistics."""
    _render(OUT, "linear")
    for exp, label, _ in CONFIGS:
        frac = coeff_fraction(exp)
        print(f"  {label:14s} median={np.median(frac)*100:5.1f}%  p99={np.percentile(frac,99)*100:5.1f}%  "
              f"max={frac.max()*100:5.1f}%  frac>=90%={ (frac>=0.9).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
