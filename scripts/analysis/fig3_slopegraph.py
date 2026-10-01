"""
Figure 3: held-out performance of the four latent-subsidy representations, as a paired
slopegraph (NOT a boxplot -- with only 5 CV replicates a boxplot's quartiles are unstable and
would misrepresent the spread; connecting matched seeds shows the within-seed improvement, which
is the actual evidence).

Reads fig3_replicate_nse.npz (per-replicate pooled NSE, written by eval_replicate_et_nse.py).
One panel per variable (ET, TWSA, SWE, Q); x-axis ordered by increasing temporal information
(no-subsidy -> static -> instantaneous -> recurrent); one thin line per seed connecting its four
configurations; the replicate-median drawn as a heavy marker.

Usage:
  python scripts/analysis/fig3_slopegraph.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE

from project_paths import REPO
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")

VAR_TITLE = {"et": "ET", "twsa": "TWSA", "swe": "SWE", "q": "Q"}
LABEL_FIX = {"no-subsidy": "baseline"}


def main(npz_name="fig3_replicate_nse.npz", out_name="fig3_performance_slopegraph.png",
         grid=False, var_order=("et", "q", "swe", "twsa"), title_prefix="",
         single_ylabel=False, label_fix=None):
    """Render Figure 3: the per-replicate pooled-NSE slopegraph (1x4 or 2x2) from the cached npz.

    label_fix optionally extends the module-level LABEL_FIX renaming (e.g. spelling out a config
    name on the x-axis) without changing the default figure."""
    fix = {**LABEL_FIX, **(label_fix or {})}
    d = np.load(os.path.join(OUT_DIR, npz_name), allow_pickle=True)
    labels = [fix.get(l, l) for l in d["labels"]]   # ordered configs
    seeds = list(d["seeds"])
    varlist = [v for v in var_order if v in d.files]
    x = np.arange(len(labels))

    seed_colors = uplt.Colormap("viridis")(np.linspace(0.1, 0.9, len(seeds)))

    # Keep the per-panel size (and therefore the visual weight of fonts and lines) identical
    # between the two layouts: the 1x4 gives ~82 mm/panel (330/4), so the 2x2 uses ~85 mm/panel
    # (~175 mm figwidth) rather than a wider figure that would shrink the type relative to the axes.
    if grid:  # 2x2 layout
        fig, axs = uplt.subplots(nrows=2, ncols=2, figwidth="175mm", wspace=6, hspace=9,
                                 share=False, abc=True, abcloc="l")
    else:     # 1x4 layout
        fig, axs = uplt.subplots(nrows=1, ncols=len(varlist), figwidth="330mm", wspace=6,
                                 share=False, abc=True, abcloc="l")
    xrot, xtick_sz = 25, TICKLABELSIZE - 1   # same in both layouts
    for i, (ax, v) in enumerate(zip(axs, varlist)):
        # 2x2: left column keeps the y-label. 1x4: every panel by default, or only the leftmost
        # panel when single_ylabel is set.
        left_col = (i % 2 == 0) if grid else (i == 0 if single_ylabel else True)
        M = d[v]                        # [n_config, n_seed]
        for si, seed in enumerate(seeds):
            ax.plot(x, M[:, si], color=seed_colors[si], lw=1.0,
                    marker="o", markersize=4, alpha=0.85,
                    label=f"seed {seed}" if v == varlist[0] else None)
        med = np.median(M, axis=1)
        ax.plot(x, med, color="k", lw=2.6, marker="D", markersize=7,
                zorder=5, label="median" if v == varlist[0] else None)
        # Tight per-panel y-limits fitted to the actual point spread (each variable lives in
        # its own narrow band), so within-panel differences among replicates stay visible.
        lo, hi = np.nanmin(M), np.nanmax(M)
        pad = 0.10 * (hi - lo)
        ax.format(
            title=f"{title_prefix}{VAR_TITLE[v]}", titlesize=TITLESIZE,
            xlim=(-0.3, len(labels) - 0.7), ylim=(lo - pad, hi + pad),
            xticks=list(x), xticklabels=labels, xrotation=xrot,
            ylabel="Pooled NSE (-)" if left_col else "", xlabel="",
            xlabelsize=LABELSIZE, ylabelsize=LABELSIZE,
            xticklabelsize=xtick_sz, yticklabelsize=TICKLABELSIZE, grid=False,
        )
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2)

    # Shared legend centered below all panels (6 entries: 5 seeds + median).
    handles, lbls = axs[0].get_legend_handles_labels()
    leg_ncols = 3 if grid else 6   # 2x2 is narrower -> wrap the legend to two rows
    fig.legend(handles, lbls, loc="b", ncols=leg_ncols, frame=False,
               prop={"size": LEGENDSIZE + (0 if grid else 2)}, order="F")
    out = os.path.join(OUT_DIR, out_name)
    fig.save(out, dpi=200)
    print(f"Saved -> {out}")

    print("\nmedian per-replicate pooled NSE:")
    for i, lab in enumerate(labels):
        print(f"  {lab:15s} " + "  ".join(f"{v}={np.median(d[v][i]):.3f}" for v in varlist))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="fig3_replicate_nse.npz")
    p.add_argument("--out", default="fig3_performance_slopegraph.png")
    p.add_argument("--grid", action="store_true", help="2x2 layout instead of 1x4")
    p.add_argument("--order", default="et,q,swe,twsa", help="comma-separated variable order")
    p.add_argument("--no-nse", action="store_true", help="drop the 'NSE — ' title prefix")
    p.add_argument("--single-ylabel", action="store_true",
                   help="1x4 only: keep the y-label on the leftmost panel only")
    a = p.parse_args()
    main(npz_name=a.npz, out_name=a.out, grid=a.grid,
         var_order=tuple(a.order.split(",")), title_prefix="" if a.no_nse else "NSE — ",
         single_ylabel=a.single_ylabel)
