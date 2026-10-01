"""
Fig. S4: cross-instance stability of the catchment-mean F_add spatial pattern, as the 25x25 matrix
of pairwise spatial Pearson correlations between the trained model instances (blocks of 5 folds per
random seed). Plots fig_si_stability.npz (fig_si_stability.py); light, safe on the login node.

Produces: fig_si_stability_corr.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE

from project_paths import RESULTS_DIR
OUT_DIR = os.path.join(RESULTS_DIR, "figures")


def main():
    """Render Fig. S4: the 25x25 cross-instance correlation matrix of catchment-mean F_add."""
    d = np.load(os.path.join(OUT_DIR, "fig_si_stability.npz"), allow_pickle=True)
    C = d["corr25"]; inst_seed = d["inst_seed"]
    seeds = [int(s) for s in d["seeds"]]

    # --- 25x25 correlation matrix ---
    n = len(C)
    off = C[~np.eye(n, dtype=bool)]
    same = inst_seed[:, None] == inst_seed[None, :]
    od = ~np.eye(n, dtype=bool)
    fig = uplt.figure(figwidth="150mm", tight=True, spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot()
    # Floor at 0.80 (not 0.90) so the real off-diagonal spread is visible; the few pairs below
    # 0.80 (min ~0.78) are indicated by the colorbar's bottom extend arrow.
    im = ax.imshow(C, cmap="viridis", vmin=0.80, vmax=1.0)
    ax.set_box_aspect(1)
    # seed block separators every 5
    for b in range(1, len(seeds)):
        ax.axhline(b * 5 - 0.5, color="w", lw=1.0)
        ax.axvline(b * 5 - 0.5, color="w", lw=1.0)
    ax.format(
        xticks=[i * 5 + 2 for i in range(len(seeds))], yticks=[i * 5 + 2 for i in range(len(seeds))],
        xticklabels=[str(s) for s in seeds], yticklabels=[str(s) for s in seeds],
        xlabel="random seed (blocks of 5 folds)", ylabel="random seed (blocks of 5 folds)",
        xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
        xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE,
    )
    fig.colorbar(im, ax=ax, loc="r", width=0.22, length=0.95, pad="1.5em", extend="min",
                 label="Pearson $r$",
                 labelsize=LABELSIZE - 2, ticklabelsize=TICKLABELSIZE - 2)
    ax.format(title="")
    out2 = os.path.join(OUT_DIR, "fig_si_stability_corr.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved -> {out2}")


if __name__ == "__main__":
    main()
