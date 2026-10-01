"""Catchment geometry does not organize the learned subsidy.
  (a) direct scatter of catchment-mean F_add against the perimeter-to-area ratio;
  (b) partial correlation of mean F_add with area, perimeter and their interaction, from a multiple
      regression on standardized log-area, log-perimeter and their product (controls for the r=0.97
      collinearity between area and perimeter).
Reads catchment_geometry.npz; no pickle load. Usage: python scripts/analysis/fig_morph_geometry.py"""
import os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE

from project_paths import RESULTS_DIR as RD
NPZ = os.path.join(RD, "figures", "catchment_geometry.npz")
OUT = os.path.join(RD, "figures", "fig_morph_geometry.png")

TERMS = ["Area", "Perimeter", "Area $\\times$\nPerimeter"]
TERM_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def _z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def partial_corr(y, area, perim):
    zA, zP = _z(np.log1p(area)), _z(np.log1p(perim))
    zI = _z(zA * zP)
    X = np.column_stack([np.ones_like(y), zA, zP, zI])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    covb = ((resid @ resid) / dof) * np.linalg.inv(X.T @ X)
    t = beta / np.sqrt(np.diag(covb))
    pr = t / np.sqrt(t ** 2 + dof)
    pval = 2 * stats.t.sf(np.abs(t), dof)
    return pr[1:], pval[1:], np.corrcoef(zA, zP)[0, 1]


def main():
    c = dict(np.load(NPZ, allow_pickle=True))
    f = c["fadd_lstm_af"].astype(float)
    area, perim = c["area_geod_km2"].astype(float), c["perimeter_geod_km"].astype(float)
    v = np.isfinite(f) & np.isfinite(area) & np.isfinite(perim)

    fig = uplt.figure(figwidth="200mm", figheight="92mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=1, ncols=2, wspace=6, wratios=(1, 1))

    # --- a: F_add vs perimeter-to-area ratio ---
    la = perim[v] / area[v]
    x, y = la, f[v]
    r, p = stats.pearsonr(x, y); rho, ps = stats.spearmanr(x, y)
    b = np.polyfit(x, y, 1); xx = np.array([x.min(), x.max()])
    axa = fig.add_subplot(gs[0, 0], abc=True, abcloc="ul")
    axa.scatter(x, y, s=13, color="#4C72B0", alpha=0.45, edgecolor="k", lw=0.2, zorder=2)
    axa.plot(xx, np.polyval(b, xx), color="k", lw=1.8, zorder=3)
    axa.text(0.96, 0.96, f"r = {r:+.2f}\n$\\rho$ = {rho:+.2f}", transform=axa.transAxes,
             ha="right", va="top", fontsize=TICKLABELSIZE - 1)
    axa.format(xlabel="Perimeter-to-area ratio (km$^{-1}$)",
               ylabel="Catchment-mean $F_{add}$ (mm day$^{-1}$)",
               xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
               xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False)
    axa.set_box_aspect(1)
    for s in axa.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)

    # --- b: partial correlations ---
    pr, pval, coll = partial_corr(_z(f[v]), area[v], perim[v])
    xpos = np.arange(len(TERMS))
    axb = fig.add_subplot(gs[0, 1], abc=True, abcloc="ul")
    axb.bar(xpos, pr, width=0.6, color=TERM_COLORS, edgecolor="k", lw=0.4)
    for xp, pp, pv in zip(xpos, pr, pval):
        if pv < 0.05:
            axb.text(xp, pp + (0.006 if pp >= 0 else -0.006), "*", ha="center",
                     va="bottom" if pp >= 0 else "top", fontsize=TICKLABELSIZE)
    axb.axhline(0, color="0.35", lw=1)
    axb.format(xlocator=xpos, xformatter=TERMS, ylim=(-0.14, 0.14),
               ylabel="Partial correlation with mean $F_{add}$", xlabel="",
               ylabelsize=LABELSIZE - 1, xticklabelsize=TICKLABELSIZE - 1,
               yticklabelsize=TICKLABELSIZE, grid=False)
    axb.set_box_aspect(1)
    for s in axb.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    axb.text(0.03, 0.03, "* p < 0.05", transform=axb.transAxes, ha="left", va="bottom",
             fontsize=TICKLABELSIZE - 2, color="0.4")

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"area-perimeter collinearity(logA,logP) r={coll:+.3f}")
    print(f"(a) F_add vs perimeter-to-area ratio: r={r:+.3f} (p={p:.2g})  rho={rho:+.3f} (p={ps:.2g})  n={int(v.sum())}")
    for t, rr, pv in zip(["Area", "Perimeter", "Interaction"], pr, pval):
        print(f"(b) {t:11s}: partial r={rr:+.3f}  p={pv:.2g}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
