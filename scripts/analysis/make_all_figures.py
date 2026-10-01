"""
Single entry point that renders every code-generated MAIN-TEXT and APPENDIX figure of the
manuscript (Figs. 1 and 3-10, and the appendix map of the study catchments) into
results_lateral_experiments/figures_manuscript_final/. Figure 2 is the framework schematic, drawn
by hand, and is not produced here.

Each figure's drawing logic lives in its own module. This file only orchestrates: it calls each
producer (which writes to its own default path, usually results_lateral_experiments/figures/) and
copies the result into figures_manuscript_final/ under a fig_<N>_<name>.png name. The figures are
rendered from cached arrays, whose producers are listed in the README.

  Fig 1   fig_1_nonclosure                <- fig1_nonclosure_inset.main
  Fig 3   fig_3_heldout_performance       <- fig3_slopegraph.main (pooled held-out NSE per replicate,
                                             five configurations, fig3_replicate_nse_frozen.npz)
  Fig 4   fig_4_paired_nse                <- fig_paired_nse_scatter.main
  Fig 5   fig_5_spatial_subsidy           <- manuscript_figures.fig5_spatial_fadd_inset (mean of the 25
                                             instance maps in fig_si_stability.npz)
  Fig 6   fig_6_shap_drivers              <- fig6_shap_drivers_v2layout_geo.main (fig6_shap_ext_cache_geo.npz)
  Fig 7   fig_7_seasonal_context          <- regen_seasonality_merged.main
  Fig 8   fig_8_effective_memory          <- fig8_memory_inset.main
  Fig 9   fig_9_nonclosure_relation       <- fig9_relation_2panel.main
  Fig 10  fig_10_fadd_contribution        <- fig_fadd_contribution_compare.main
  Fig A   fig_A_study_catchments          <- fig_catchment_kg_map.main

Usage:
  python scripts/analysis/make_all_figures.py [--only 1,3,A]
"""
import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from project_paths import REPO
RES = os.path.join(REPO, "results_lateral_experiments")
FIG = os.path.join(RES, "figures")
OUT = os.path.join(RES, "figures_manuscript_final")
os.makedirs(OUT, exist_ok=True)


def _copy(src, dst_name):
    dst = os.path.join(OUT, dst_name)
    shutil.copy2(src, dst)
    print(f"Saved -> {dst}")


# ---------------------------------------------------------------- Fig 1
def fig1():
    """Figure 1: non-closure R_in/P world map + Budyko-space inset."""
    import fig1_nonclosure_inset as F
    F.main()  # already writes directly to figures_manuscript_final/fig_1_nonclosure_inset.png
    src = os.path.join(OUT, "fig_1_nonclosure_inset.png")
    if os.path.exists(src):
        os.replace(src, os.path.join(OUT, "fig_1_nonclosure.png"))
        print(f"Saved -> {os.path.join(OUT, 'fig_1_nonclosure.png')}")


# ---------------------------------------------------------------- Fig 3
def fig3():
    """Figure 3: held-out pooled-NSE slopegraph (ET, Q, SWE, TWSA) across the five configurations
    (no-subsidy, static, instantaneous, recurrent, recurrent with frozen decoder)."""
    import fig3_slopegraph as F
    F.main(npz_name="fig3_replicate_nse_frozen.npz",
           out_name=os.path.join(OUT, "fig_3_heldout_performance.png"),
           single_ylabel=True)


# ---------------------------------------------------------------- Fig 4
def fig4():
    """Figure 4: per-catchment recurrent-vs-baseline NSE scatter, one panel per target, KG-coloured."""
    import fig_paired_nse_scatter as F
    F.main(out_name=os.path.join(OUT, "fig_4_paired_nse.png"), row_layout=True)


# ---------------------------------------------------------------- Fig 5
def fig5():
    """Figure 5: global map of catchment-mean recurrent F_add (2001-2020), averaged across the 25
    model instances (5 seeds x 5 folds, every catchment in every instance), with a per-KG-class
    boxplot inset. The strictly held-out counterpart is Fig. S5."""
    from manuscript_figures import fig5_spatial_fadd_inset
    from basin_attributes import build_basin_tables
    d = np.load(os.path.join(FIG, "fig_si_stability.npz"), allow_pickle=True)
    ids = [str(g) for g in d["gauge_id"]]
    fadd_mean = np.nanmean(d["inst_maps"], axis=0)          # [25, B] -> [B]
    df_coord, _ = build_basin_tables(ids)
    df_coord = df_coord.set_index("gauge_id").reindex(ids).reset_index()
    fig5_spatial_fadd_inset(None, df_coord, os.path.join(OUT, "fig_5_spatial_subsidy.png"),
                            fadd_mean=fadd_mean)


# ---------------------------------------------------------------- Fig 6
def fig6():
    """Figure 6: SHAP ranking and dependence panels of the 18-predictor surrogate, rendered from the
    cached 100x5-fold CV SHAP arrays (fig6_shap_ext_cache_geo.npz, written by shap_fig6_cache.py)."""
    import fig6_shap_drivers_v2layout_geo as F
    F.main()  # writes directly to figures_manuscript_final/fig_6_shap_drivers_v4.png
    src = os.path.join(OUT, "fig_6_shap_drivers_v4.png")
    if os.path.exists(src):
        os.replace(src, os.path.join(OUT, "fig_6_shap_drivers.png"))
        print(f"Saved -> {os.path.join(OUT, 'fig_6_shap_drivers.png')}")


# ---------------------------------------------------------------- Fig 7
def fig7():
    """Figure 7: seasonal climatologies (P, PET, ET, F_add, LAI) per KG class, season-aligned."""
    import regen_seasonality_merged as R
    R.main(out_name=os.path.join(OUT, "fig_7_seasonal_context.png"))


# ---------------------------------------------------------------- Fig 8
def fig8():
    """Figure 8: truncation-error curves and effective-history map, with the per-KG-class mean L_s inset."""
    import fig8_memory_inset as F
    F.main()
    _copy(os.path.join(FIG, "fig8_memory_k365_curve_Ls_inset.png"), "fig_8_effective_memory.png")


# ---------------------------------------------------------------- Fig 9
def fig9():
    """Figure 9: catchment-mean F_add against R_in with per-class OLS fits, and the per-class slopes."""
    import fig9_relation_2panel as F
    F.main()  # writes directly to figures_manuscript_final/fig_9_nonclosure_relation_2panel.png
    os.replace(os.path.join(OUT, "fig_9_nonclosure_relation_2panel.png"),
               os.path.join(OUT, "fig_9_nonclosure_relation.png"))
    print(f"Saved -> {os.path.join(OUT, 'fig_9_nonclosure_relation.png')}")


# ---------------------------------------------------------------- Fig 10
def fig10():
    """Figure 10: whole-period ET-contribution map + per-Koppen-Geiger-class inset boxplot."""
    import fig_fadd_contribution_compare as F
    F.main()
    _copy(os.path.join(FIG, "fig_fadd_contribution_compare.png"), "fig_10_fadd_contribution.png")


# ---------------------------------------------------------------- Appendix
def figA():
    """Appendix figure: study catchments over the Koppen-Geiger classes, with the per-class count inset."""
    import fig_catchment_kg_map as F
    F.main()
    _copy(os.path.join(FIG, "fig_catchment_kg_map.png"), "fig_A_study_catchments.png")


FIGURES = {"1": fig1, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8, "9": fig9,
           "10": fig10, "A": figA}


def main():
    """Build the requested figures (all by default, or a subset via --only 1,3,A)."""
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="comma-separated figure numbers to build, A for the appendix map (default: all)")
    a = p.parse_args()
    want = ([x.strip().upper() for x in a.only.split(",")] if a.only else list(FIGURES))
    for n in want:
        print(f"\n=== Figure {n} ===")
        FIGURES[n]()


if __name__ == "__main__":
    main()
