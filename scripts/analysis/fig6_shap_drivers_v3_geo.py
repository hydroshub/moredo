"""
Fig. S18: SHAP dependence scatters for all 18 predictors of the F_add surrogate of Fig. 6 (base 14
+ karst, area, perimeter, lithology), in a four-column grid ordered by descending relative SHAP
contribution (the ranking of Fig. 6a). Per-point median SHAP with IQR whiskers across the 100x5-fold
CV repeats, batlowK colour = mean ET, one shared colorbar and one shared SHAP y-range.

Pure re-render: reuses the cached 100x5-fold CV SHAP arrays for the extended surrogate
(fig6_shap_ext_cache_geo.npz), so no RF re-fitting. The aridity axis is relabelled PET/P.

Usage:
  python scripts/analysis/fig6_shap_drivers_v3_geo.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
FIG = os.path.join(RD, "figures")
OUT = FIG
CACHE = os.path.join(FIG, "fig6_shap_ext_cache_geo.npz")

NCOLS = 4
CBAR_VMAX = 2.4
CBAR_LABEL = "Mean ET (mm day$^{-1}$)"


def main(out_name="fig_si_shap_dependence_all.png"):
    """Render the all-predictor, four-column SHAP dependence grid (Fig. S18) into figures/."""
    cz = np.load(CACHE, allow_pickle=True)
    selected = [str(c) for c in cz["cols"]]
    shap_med, q1, q3 = cz["shap_median"], cz["shap_q1"], cz["shap_q3"]
    rank_med = np.asarray(cz["rank_med"], float)
    X_proc = pd.DataFrame(cz["Xvals"], columns=selected)
    et_color = np.asarray(cz["et_color"], float)
    # long axis labels from the cache (ari_ix_conv is PET/P, not P/PET)
    legend = {str(k): str(v).replace("P/PET", "PET/P")
              for k, v in zip(cz["legend_keys"], cz["legend_vals"])}
    legend["pci"] = "PCI (-)"   # abbreviation, explained in the caption (as in Fig. 6)

    full_order = np.argsort(rank_med)[::-1]
    feat_order = [selected[i] for i in full_order]

    cmap = uplt.Colormap("batlowK")
    ndep = len(selected)
    nrows_dep = int(np.ceil(ndep / NCOLS))

    fig = uplt.figure(figwidth="250mm", figheight="330mm", spanx=False, spany=False,
                      sharex=False, sharey=False, tight=True)
    gs = uplt.GridSpec(nrows=nrows_dep, ncols=NCOLS, hspace=3.5, wspace=4.5)

    # --- shared SHAP y-range across all dependence panels (median +/- IQR) ---
    lo = [np.nanmin(q1[:, selected.index(f)]) for f in feat_order]
    hi = [np.nanmax(q3[:, selected.index(f)]) for f in feat_order]
    ymin, ymax = float(np.nanmin(lo)), float(np.nanmax(hi))
    pad = 0.05 * (ymax - ymin)
    shared_ylim = (ymin - pad, ymax + pad)

    # --- a, b, ... dependence scatters, row-major in the NCOLS-wide grid ---
    positions = [(r, c) for r in range(nrows_dep) for c in range(NCOLS)]
    sc = None
    for idx, f in enumerate(feat_order):
        r, c = positions[idx]
        ax = fig.add_subplot(gs[r, c], abc=True, abcloc="l")
        j = selected.index(f)
        sfeat = shap_med[:, j]
        fvals = X_proc[f].values
        ax.axhline(0, ls="--", color="orange", lw=1, zorder=0)
        ylo = sfeat - q1[:, j]; yhi = q3[:, j] - sfeat
        ax.errorbar(fvals, sfeat, yerr=np.clip(np.vstack([ylo, yhi]), 0, None),
                    fmt="none", ecolor="0.6", elinewidth=0.6, alpha=0.7, zorder=1)
        sc = ax.scatter(fvals, sfeat, c=et_color, cmap=cmap, s=20, vmin=0, vmax=CBAR_VMAX,
                        edgecolor="k", lw=0.15, zorder=2)
        ax.set_box_aspect(1)
        ax.format(xlabel=legend.get(f, f), xlabelsize=LABELSIZE - 1,
                  ylabel="SHAP value" if c == 0 else "", ylim=shared_ylim,
                  ylabelsize=LABELSIZE, xticklabelsize=TICKLABELSIZE - 1,
                  yticklabelsize=TICKLABELSIZE - 1, grid=False)

    # --- colorbar in the leftover bottom-right cell ---
    if sc is not None and ndep < len(positions):
        er, ec = positions[ndep]
        fig.canvas.draw()                    # realise the tight layout so cell positions are final
        pos = gs[er, ec].get_position(fig)   # figure-fraction bbox of the leftover cell
        cax = fig.add_axes([pos.x0 + 0.06 * pos.width, pos.y0 + 0.5 * pos.height,
                            0.88 * pos.width, 0.035 * pos.height])
        cax.set_in_layout(False)
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="max",
                          ticks=[0, round(CBAR_VMAX / 2, 1), round(CBAR_VMAX, 1)])
        cb.set_label(CBAR_LABEL, size=LABELSIZE)
        cb.ax.tick_params(labelsize=TICKLABELSIZE - 1)

    out = os.path.join(OUT, out_name)
    fig.savefig(out, dpi=200)
    print(f"Saved -> {out}  ({ndep} predictors, {NCOLS} cols, {nrows_dep} rows)")


if __name__ == "__main__":
    main()
