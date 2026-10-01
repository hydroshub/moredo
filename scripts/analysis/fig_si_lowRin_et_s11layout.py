"""
Fig. S14: held-out ET gain of the recurrent representation over the no-subsidy baseline in the
catchments with the smallest long-term water-balance non-closure |R_in|, two square panels
(figwidth 200mm x figheight 92mm, the same footprint as Fig. S10, fig_morph_geometry.py).
(a) per-catchment Delta ET NSE against |R_in|, smallest-|R_in| quartile highlighted.
(b) distribution of Delta ET NSE within that quartile. Reads the averaged outputs of lstm_af and
baseline. Usage: python scripts/analysis/fig_si_lowRin_et_s11layout.py
"""
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE, EXP_COLORS
from basin_attributes import build_basin_tables
from residual_utils import compute_residual_vs_fadd
from metrics import basin_nse_table

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
OUT_DIR = os.path.join(RD, "figures")
OUT = os.path.join(OUT_DIR, "fig_si_lowRin_et_s11layout.png")


def _load():
    """Exact data prep as fig_si_s6_s7._load(): recurrent + baseline ET NSE and the R_in residual."""
    lstm = pickle.load(open(os.path.join(RD, "lstm_af", "averaged_full_output.pickle"), "rb"))
    base = pickle.load(open(os.path.join(RD, "baseline", "averaged_full_output.pickle"), "rb"))
    vids = list(lstm["valid_ids"])
    dfc, dff = build_basin_tables(vids)
    res = compute_residual_vs_fadd(lstm, dfc)
    Rin = -res["residual_corrected"]; valid = res["valid"]
    et_b = basin_nse_table(base, target_channels=("et",))["et"]
    et_l = basin_nse_table(lstm, target_channels=("et",))["et"]
    return Rin, valid, et_b, et_l


def main():
    Rin, valid, et_b, et_l = _load()
    absR = np.abs(Rin)
    m = valid & np.isfinite(et_b) & np.isfinite(et_l) & np.isfinite(absR)
    q1 = np.nanpercentile(absR[m], 25)
    inq = m & (absR <= q1)
    dnse = et_l - et_b
    frac = np.mean(dnse[inq] > 0)
    print(f"n(smallest-|R_in| quartile)={inq.sum()}, fraction recurrent>baseline={frac*100:.0f}%")

    # --- fixed layout (matches fig_morph_geometry / Fig. S10) ---
    fig = uplt.figure(figwidth="200mm", figheight="92mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=1, ncols=2, wspace=6, wratios=(1, 1))

    # --- a: delta ET NSE vs |R_in| ---
    axa = fig.add_subplot(gs[0, 0], abc=True, abcloc="ul")
    axa.scatter(absR[m & ~inq], dnse[m & ~inq], s=10, color="0.7", alpha=0.5,
               edgecolor="none", label="other catchments")
    axa.scatter(absR[inq], dnse[inq], s=13, color=EXP_COLORS["lstm_af"], alpha=0.8,
               edgecolor="k", lw=0.2, label="smallest-$|R_{in}|$ quartile")
    axa.axhline(0, color="gray", ls="--", lw=1, zorder=1)
    axa.axvline(q1, color="k", ls=":", lw=1, zorder=1)
    axa.format(xlim=(0, np.nanpercentile(absR[m], 90)),
               ylim=(np.nanpercentile(dnse[m], 2), np.nanpercentile(dnse[m], 98)),
               xlabel="$|R_{in}|$ (mm day$^{-1}$)", ylabel="$\\Delta$ET NSE (recurrent $-$ baseline)",
               xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
               xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False)
    axa.set_box_aspect(1)
    for s in axa.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    axa.legend(loc="ur", ncols=1, frame=True, prop={"size": LEGENDSIZE - 2})

    # --- b: distribution of delta ET NSE within the lowest-|R_in| quartile ---
    axb = fig.add_subplot(gs[0, 1], abc=True, abcloc="ul")
    axb.hist(np.clip(dnse[inq], -0.3, 0.3), bins=30, color=EXP_COLORS["lstm_af"],
             edgecolor="k", lw=0.4)
    axb.axvline(0, color="k", ls="--", lw=1.2, zorder=2)
    axb.format(xlabel="$\\Delta$ET NSE (recurrent $-$ baseline)", ylabel="Catchments",
               xlabelsize=LABELSIZE - 1, ylabelsize=LABELSIZE - 1,
               xticklabelsize=TICKLABELSIZE, yticklabelsize=TICKLABELSIZE, grid=False)
    axb.set_box_aspect(1)
    for s in axb.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"n = {int(m.sum())}  quartile n = {int(inq.sum())}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
