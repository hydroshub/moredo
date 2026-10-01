"""SI figure: sensitivity of the recurrent subsidy's effective memory to LSTM hidden size.
Overlays the median normalized truncation error E_s(k) for hidden sizes 16, 32 (default), 64,
with the eps=0.10 threshold; legend reports median L_s and mean F_add per variant."""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, TITLESIZE, LEGENDSIZE

from project_paths import REPO
FIG = os.path.join(REPO, "results_lateral_experiments", "figures")
RD = os.path.join(REPO, "results_lateral_experiments")
ACTIVE_START = 2001
# All three from the full 25-instance ensemble (5 seeds x 5 folds).
VARIANTS = [("16", "memory_truncation_h16.npz", "lstm_af_h16", "#4C72B0"),
            ("32", "memory_truncation.npz",     "lstm_af",     "#000000"),
            ("48", "memory_truncation_h48.npz", "lstm_af_h48", "#DD8452")]


def mean_fadd(exp):
    with open(os.path.join(RD, exp, "averaged_full_output.pickle"), "rb") as f:
        r = pickle.load(f)
    fl = np.squeeze(r["fluxes"]["f_lat"])
    idx = pd.DatetimeIndex(r["idx_daily"]); act = idx.year >= ACTIVE_START
    return float(np.nanmean(fl[:, act]))


def main():
    fig, ax = uplt.subplots(refwidth="110mm", refaspect=1.35, share=False, top="6mm")
    eps = 0.10
    for h, npz, exp, col in VARIANTS:
        p = os.path.join(FIG, npz)
        if not os.path.exists(p):
            print(f"  skip hidden={h}: {npz} missing"); continue
        d = np.load(p, allow_pickle=True)
        k = d["k_values"]; med_e = np.nanmedian(d["E_sk"], axis=0)
        Ls = np.nanmedian(d["L"])
        try: mf = mean_fadd(exp)
        except Exception: mf = np.nan
        lab = f"{h} units ($L_s$={Ls:.0f} d, $\\overline{{F_{{add}}}}$={mf:.2f})"
        lw = 2.6 if h == "32" else 1.8
        ax.plot(k, med_e, color=col, lw=lw, marker="o", markersize=4, label=lab, zorder=4)
    ax.axhline(eps, color="0.4", ls="--", lw=1.2, zorder=2, label=f"$\\epsilon$ = {eps:.2f}")
    ax.format(xlabel="History length $k$ (days)",
              ylabel="Normalized truncation error $E_s(k)$",
              xlim=(0, 366), ylim=(0, None),
              xlabelsize=LABELSIZE, ylabelsize=LABELSIZE,
              xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    ax.legend(loc="ur", ncols=1, frame=False, prop={"size": LEGENDSIZE})
    out = os.path.join(FIG, "fig_si_memory_hidden_sweep.png")
    fig.save(out, dpi=200)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
