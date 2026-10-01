"""
Single entry point that renders every SUPPLEMENTARY figure (Figs. S1-S18) into
results_lateral_experiments/figures_si_final/.

Each figure's drawing logic lives in its own module. This file only orchestrates: it calls each
producer (which writes to its own default path, usually results_lateral_experiments/figures/) and
copies the result into figures_si_final/ under a fig_si_<NN>_<name>.png name. Some producer file
names carry older S-numbers or layout tags (e.g. fig_si_s10_... renders Fig. S9, fig_si_s12_...
renders S11, *_s11layout files render S2 and S14); the numbers below are the manuscript's.

  S1  fig_si_01_coefficient_bound      <- fig_si_coef_bound.main
  S2  fig_si_02_bound_sensitivity      <- fig_si_boundS2_s11layout.main
  S3  fig_si_03_loss_normalization     <- fig_si_loss_normalization.main
  S4  fig_si_04_ensemble_stability     <- fig_si_stability_plot.main
  S5  fig_si_05_representation_maps     \\
  S6  fig_si_06_heldout_subsidy_map      >- fig_si_repr_heldout_plot.main (one run, two outputs)
  S7  fig_si_07_large_catchments       <- fig_si_largebasins.main
  S8  fig_si_08_grouped_shap           <- fig_si_grouped_shap.main
  S9  fig_si_09_heldout_shap           <- fig_si_s10_heldout_shap_vertical.main
  S10 fig_si_10_morphology             <- fig_morph_geometry.main
  S11 fig_si_11_seasonal_shap          <- fig_si_s12_seasonal_shap_vertical.main
  S12 fig_si_12_antecedent_correlation <- memory_truncation_plot_v2.build_map_only
  S13 fig_si_13_hidden_size_sweep      <- fig_si_memory_hidden_sweep.main
  S14 fig_si_14_lowclosure_et_gain     <- fig_si_lowRin_et_s11layout.main
  S15 fig_si_15_signed_fadd            <- fig_neg_fadd.main
  S16 fig_si_16_loss_weight_ablation   <- fig_wcmp_combined.main
  S17 fig_si_17_relative_subsidy       <- fig_fadd_relative_maps.main
  S18 fig_si_18_shap_dependence        <- fig6_shap_drivers_v3_geo.main

Usage:
  python scripts/analysis/make_all_si_figures.py [--only 1,7,16]
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from project_paths import REPO
FIG = os.path.join(REPO, "results_lateral_experiments", "figures")
SI = os.path.join(REPO, "results_lateral_experiments", "figures_si_final")
os.makedirs(SI, exist_ok=True)


def _copy(src_name_or_path, dst_name, from_fig_dir=True):
    src = os.path.join(FIG, src_name_or_path) if from_fig_dir else src_name_or_path
    dst = os.path.join(SI, dst_name)
    shutil.copy2(src, dst)
    print(f"Saved -> {dst}")


# ---------------------------------------------------------------- S1
def s1():
    """S1: distribution of the learned subsidy coefficient as a fraction of its imposed bound."""
    import fig_si_coef_bound as M
    M.main()
    _copy("fig_si_coef_bound.png", "fig_si_01_coefficient_bound.png")


# ---------------------------------------------------------------- S2
def s2():
    """S2: (a) recurrent F_add under the default [0,0.01] vs doubled [0,0.02] coefficient bound,
    (b) coefficient distribution of the three encoders re-trained with the doubled bound."""
    import fig_si_boundS2_s11layout as M
    M.main()
    _copy("fig_si_boundS2_s11layout.png", "fig_si_02_bound_sensitivity.png")


# ---------------------------------------------------------------- S3
def s3():
    """S3: recurrent representation trained with the cross-catchment pooled loss vs the
    catchment-normalized loss (held-out NSE, F_add scatter, two F_add maps)."""
    import fig_si_loss_normalization as M
    M.main()
    _copy("fig_si_loss_normalization.png", "fig_si_03_loss_normalization.png")


# ---------------------------------------------------------------- S4
def s4():
    """S4: cross-instance spatial stability of catchment-mean F_add (25x25 correlation matrix)."""
    import fig_si_stability_plot as M
    M.main()
    _copy("fig_si_stability_corr.png", "fig_si_04_ensemble_stability.png")


# ---------------------------------------------------------------- S5 / S6
def s5_s6():
    """S5: representation comparison (static/instantaneous/recurrent F_add maps).
    S6: strictly held-out-only recurrent F_add."""
    import fig_si_repr_heldout_plot as M
    M.main()
    _copy("fig_si_representation_maps.png", "fig_si_05_representation_maps.png")
    _copy("fig_si_heldout_subsidy_map.png", "fig_si_06_heldout_subsidy_map.png")


# ---------------------------------------------------------------- S7
def s7():
    """S7: recurrent representation retrained on the 362 catchments > 1000 km^2 against the main
    model (map, F_add scatter, held-out NSE)."""
    import fig_si_largebasins as M
    M.main()
    _copy("fig_si_largebasins.png", "fig_si_07_large_catchments.png")


# ---------------------------------------------------------------- S8
def s8():
    """S8: SHAP importance aggregated into physically related predictor groups."""
    import fig_si_grouped_shap as M
    M.main()
    _copy("fig_si_grouped_shap.png", "fig_si_08_grouped_shap.png")


# ---------------------------------------------------------------- S9
def s9():
    """S9: SHAP surrogate fitted to the strictly held-out F_add estimate."""
    import fig_si_s10_heldout_shap_vertical as M
    M.main()
    _copy("fig_si_s10_heldout_shap_vertical.png", "fig_si_09_heldout_shap.png")


# ---------------------------------------------------------------- S10
def s10():
    """S10: F_add vs perimeter-to-area ratio + partial correlations of F_add with catchment area/perimeter."""
    import fig_morph_geometry as M
    M.main()
    _copy("fig_morph_geometry.png", "fig_si_10_morphology.png")


# ---------------------------------------------------------------- S11
def s11():
    """S11: seasonal SHAP importance (JJA vs DJF) for Northern-Hemisphere catchments."""
    import fig_si_s12_seasonal_shap_vertical as M
    M.main()
    _copy("fig_si_s12_seasonal_shap_vertical.png", "fig_si_11_seasonal_shap.png")


# ---------------------------------------------------------------- S12
def s12():
    """S12: map of deseasonalized F_add vs antecedent P-PET correlation over the effective history."""
    import memory_truncation_plot_v2 as M
    K, E_sk, L, r, eps, lon, lat = M._load()
    M.build_map_only(K, E_sk, L, r, eps, lon, lat, out_name="fig_si_memory_ac_map.png")
    _copy("fig_si_memory_ac_map.png", "fig_si_12_antecedent_correlation.png")


# ---------------------------------------------------------------- S13
def s13():
    """S13: sensitivity of effective memory to LSTM hidden size (16 vs 32 vs 48)."""
    import fig_si_memory_hidden_sweep as M
    M.main()
    _copy("fig_si_memory_hidden_sweep.png", "fig_si_13_hidden_size_sweep.png")


# ---------------------------------------------------------------- S14
def s14():
    """S14: held-out ET gain over the no-subsidy baseline in the lowest-|R_in| catchments."""
    import fig_si_lowRin_et_s11layout as M
    M.main()
    _copy("fig_si_lowRin_et_s11layout.png", "fig_si_14_lowclosure_et_gain.png")


# ---------------------------------------------------------------- S15
def s15():
    """S15: signed F_add (non-negativity relaxed) -- net sink in snow-dominated catchments."""
    import fig_neg_fadd as M
    M.main()
    _copy("fig_neg_fadd.png", "fig_si_15_signed_fadd.png")


# ---------------------------------------------------------------- S16
def s16():
    """S16: loss-weight ablation (TWSA-off, Q-off) -- F_add pattern retained, skill shifts."""
    import fig_wcmp_combined as M
    M.main()
    _copy("fig_wcmp_combined.png", "fig_si_16_loss_weight_ablation.png")


# ---------------------------------------------------------------- S17
def s17():
    """S17: catchment-mean F_add relative to P, PET and ET (three maps, one shared colour scale)."""
    import fig_fadd_relative_maps as M
    M.main()
    _copy("fig_fadd_relative_maps.png", "fig_si_17_relative_subsidy.png")


# ---------------------------------------------------------------- S18
def s18():
    """S18: SHAP dependence of all 18 predictors of the Fig. 6 F_add surrogate."""
    import fig6_shap_drivers_v3_geo as M
    M.main()
    _copy("fig_si_shap_dependence_all.png", "fig_si_18_shap_dependence.png")


FIGURES = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5_s6, 6: s5_s6, 7: s7, 8: s8, 9: s9, 10: s10,
           11: s11, 12: s12, 13: s13, 14: s14, 15: s15, 16: s16, 17: s17, 18: s18}


def main():
    """Build the requested SI figures (all by default, or a subset via --only 1,7,16)."""
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="comma-separated S-numbers (default: all)")
    a = p.parse_args()
    want = ([int(x) for x in a.only.split(",")] if a.only else sorted(FIGURES))
    ran = set()
    for n in want:
        fn = FIGURES[n]
        if fn in ran:
            continue           # S5/S6 share one producer, only run once
        ran.add(fn)
        print(f"\n=== Figure S{n} ===")
        fn()


if __name__ == "__main__":
    main()
