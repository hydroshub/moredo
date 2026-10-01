"""Fig. S16: loss-weight ablation of the recurrent model:
  (a) held-out per-basin NSE (ET, Q, SWE, TWSA) for main vs TWSA-weight-0 vs Q-weight-0;
  (b) catchment-mean F_add, main vs TWSA-weight-0;
  (c) catchment-mean F_add, main vs Q-weight-0.
Reads loss_weight_compare.npz and catchment_geometry.npz (catchment area). Usage: python scripts/analysis/fig_wcmp_combined.py"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, sns, LABELSIZE, TICKLABELSIZE, LEGENDSIZE

from project_paths import RESULTS_DIR as RD
NPZ = os.path.join(RD, "figures", "loss_weight_compare.npz")
MORPH = os.path.join(RD, "figures", "catchment_geometry.npz")
OUT = os.path.join(RD, "figures", "fig_wcmp_combined.png")

VARS = ["ET", "Q", "SWE", "TWSA"]
RUNS = [("lstm_af", "Default (all weights = 1)", "#9aa0a6"),
        ("lstm_af_twsa0", "TWSA weight = 0", "#8172B3"),
        ("lstm_af_q0", "Q weight = 0", "#C44E52")]
SCATTER = [("lstm_af_twsa0", "TWSA weight = 0", "#8172B3"),
           ("lstm_af_q0", "Q weight = 0", "#C44E52")]


def main():
    c = dict(np.load(NPZ, allow_pickle=True))
    # join catchment area (km2) by gauge_id from catchment_geometry.npz
    m = dict(np.load(MORPH, allow_pickle=True))
    area_by_id = {str(g): a for g, a in zip(m["gauge_id"], m["area_geod_km2"].astype(float))}
    area = np.array([area_by_id.get(str(g), np.nan) for g in c["gauge_id"]], float)

    fig = uplt.figure(figwidth="175mm", figheight="185mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=2, ncols=2, hratios=(1, 1.15), wspace=6, hspace=0.3)

    # --- a: held-out NSE boxplots (spans both columns) ---
    recs = []
    for var in VARS:
        for exp, lab, _ in RUNS:
            a = c[f"nse_{var}_{exp}"].astype(float)
            a = a[np.isfinite(a)]
            recs += [{"var": var, "run": lab, "nse": x} for x in a]
    df = pd.DataFrame(recs)
    pal = {lab: col for _, lab, col in RUNS}
    order_runs = [lab for _, lab, _ in RUNS]

    axa = fig.add_subplot(gs[0, :], abc=True, abcloc="ul")
    sns.boxplot(data=df, x="var", y="nse", hue="run", order=VARS, hue_order=order_runs,
                palette=pal, legend=False, width=0.7, linewidth=1.0, showfliers=False, ax=axa)
    axa.axhline(0, color="0.4", ls="--", lw=1, alpha=0.7, zorder=0)
    axa.format(ylim=(-6, 1.2), xlabel="", ylabel="Per-catchment NSE (held-out)",
               ylabelsize=LABELSIZE, xticklabelsize=LABELSIZE - 1, yticklabelsize=TICKLABELSIZE, grid=False)
    axa.set_yticks([-6, -5, -4, -3, -2, -1, 0, 1])
    for s in axa.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=col, edgecolor="0.3", label=lab) for _, lab, col in RUNS]
    axa.legend(handles=handles, loc="lr", ncols=1, frame=False, prop={"size": LEGENDSIZE - 1})

    # --- b, c: F_add scatter vs main, coloured by catchment area ---
    fo = c["fadd_lstm_af"]
    hi = np.nanpercentile(np.concatenate([fo, c["fadd_lstm_af_twsa0"], c["fadd_lstm_af_q0"]]), 99)
    norm = LogNorm(vmin=100, vmax=2500)
    cmap = "viridis"
    sc = None
    scax = []
    for i, (exp, lab, col) in enumerate(SCATTER):
        f = c[f"fadd_{exp}"]
        v = np.isfinite(fo) & np.isfinite(f) & np.isfinite(area)
        r = stats.pearsonr(fo[v], f[v])[0]
        ax = fig.add_subplot(gs[1, i], abc=True, abcloc="ul")
        ax.plot([0, hi], [0, hi], color="0.4", ls="--", lw=1.2, zorder=1)
        sc = ax.scatter(fo[v], f[v], c=area[v], s=15, cmap=cmap, norm=norm,
                        alpha=0.75, edgecolor="k", lw=0.2, zorder=2)
        ax.text(0.96, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=TICKLABELSIZE - 1)
        ax.format(xlim=(0, hi), ylim=(0, hi), xlabel="Mean $F_{add}$, default (mm day$^{-1}$)",
                  ylabel=f"Mean $F_{{add}}$, {lab} (mm day$^{{-1}}$)",
                  xlabelsize=LABELSIZE - 4, ylabelsize=LABELSIZE - 4,
                  xticklabelsize=TICKLABELSIZE - 2, yticklabelsize=TICKLABELSIZE - 2)
        ax.set_box_aspect(1)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2)
        scax.append(ax)

    # shared colorbar below the two scatter panels
    fig.canvas.draw()
    p0, p1 = scax[0].get_position(), scax[1].get_position()
    cw = 0.5 * (p1.x1 - p0.x0)
    cbax = fig.add_axes([p0.x0 + (p1.x1 - p0.x0 - cw) / 2, p0.y0 - 0.055, cw, 0.012])
    cb = fig.colorbar(sc, cax=cbax, orientation="horizontal")
    cb.set_label("Catchment area (km$^2$)", size=LABELSIZE - 3)
    cb.ax.tick_params(labelsize=TICKLABELSIZE - 2)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
