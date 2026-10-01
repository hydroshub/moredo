# Code for "Physics-constrained latent process learning for hidden subsurface water subsidies to ecosystems"

Greiner et al. This directory contains the code that reproduces the experiments, figures, tables
and quoted numbers of the manuscript and its Supplement, one version of each: the experiment
configurations, the training and averaging entry points, the producers of every cached array the
figures read, and the figure and table scripts.

The workflow has three stages:

1. **Train** the differentiable ecohydrological model (MOREDO) with a subsidy encoder, over a
   5-seed x 5-fold spatial cross-validation, for each experiment (`scripts/train.py`,
   `configs/`, `slurm/`, `libs/`).
2. **Average and diagnose**: average the 25 trained instances into one ensemble output per
   experiment (`scripts/analysis/build_averaged_output.py`), then run the producers that write the
   cached arrays (`scripts/analysis/`, see below).
3. **Plot**: render the figures and tables from those cached arrays (`scripts/analysis/` figure
   scripts and the two drivers).

---

## Directory layout

```
libs/                MOREDO differentiable model + data loading + training utilities
                     (model.py, data.py, utils.py) -- from Blougouras et al. (2026)
configs/             configurations of the reported experiments and the basin lists
slurm/               HPC (SLURM) job scripts for training, averaging and the analyses
scripts/train.py     training entry point (one seed x fold instance)
scripts/analysis/    averaging (build_averaged_output.py), analysis and figure code
                     (flat, so intra-imports resolve)
```

`scripts/analysis/` is kept as a single flat directory on purpose: the modules import each other
by bare name (`from plot_style import ...`), so they must sit side by side. The logical grouping is
documented below rather than expressed as subfolders. All commands below are run from the
repository root.

---

## Environment

Python 3 with the packages and versions in `requirements.txt` (`torch`, `numpy`, `pandas`,
`xarray`, `netCDF4`, `scipy`, `scikit-learn`, `shap`, `matplotlib`, `ultraplot`, `cartopy`,
`seaborn`). Training and the scripts that re-run the trained models require a CUDA GPU.

> **Paths.** No script contains a machine-specific path. All locations are defined in
> `scripts/analysis/project_paths.py`: the repository root is found automatically, trained models,
> averaged outputs and cached arrays are written under `results_lateral_experiments/` (cached
> arrays in its `figures/` subdirectory), and the study catchments are read from
> `configs/basin_list_global_grdc.txt`. The input data are expected under `data/` in the
> repository root (set `MOREDO_DATA_ROOT` to use another location), with the GRDC-Caravan
> attribute tables in `GRDC_Caravan/attributes/grdc/` and the Koppen-Geiger map (Beck et al., 2023)
> in `KG_Beck23/koppen_geiger_nc/1991_2020/`. The model data directory and attribute files are set
> in `configs/config_base.py` (`DATA_DIR`, `ATTR_PATHS`, `/path/to/` placeholders). The SLURM
> scripts take the repository root from the submission directory and use the active Python
> environment; machine-specific values can be set in an optional `slurm/site_local.sh` (see
> *SLURM scripts*). Data inputs (GRDC-Caravan, HydroATLAS, ERA5-Land, GIMMS LAI4g, FLUXCOM X-BASE,
> GRACE) are the public sources cited in the manuscript and are not redistributed here.

---

## Reproducing the results

1. **Train** every configuration in `configs/` (5 seeds x 5 folds) with `slurm/submit_train_global.sh`.
   The frozen-decoder run of Fig. 3 and Table B1 is trained with `slurm/submit_train_frozen_v2.sh`
   (the MOREDO parameter networks are initialized from the no-subsidy baseline and kept fixed, only
   the subsidy encoder is trained).
2. **Average** each run into `<experiment>/averaged_full_output.pickle` with
   `slurm/submit_build_averaged.sh`. For the hidden-size runs, `slurm/submit_hsweep_mem_full.sh`
   averages and runs the history-truncation experiment in one job.
3. **Produce the cached arrays** with the scripts in the next table. Run `catchment_geometry.py`
   first (the surrogate and geometry figures read it), `eval_replicate_et_nse.py` before
   `eval_replicate_frozen.py`, and `nonclosure_relation_cache.py` before `rin_surrogate_cv.py`.
   Scripts that re-run the trained models need a GPU (`slurm/submit_analysis_gpu.sh`), the
   random-forest surrogates are run with `slurm/submit_analysis_cpu.sh`.
4. **Render** with the two drivers and print Table B1 (see *Figures* and *Tables*).

### Producers of the cached arrays (`scripts/analysis/`)
| file | writes | used for |
|---|---|---|
| `catchment_geometry.py` | `catchment_geometry.npz` (perimeter, area, mean F_add) | Figs. 6, S9, S10, S11, S16 |
| `nonclosure_cache.py` | `fig_budyko_nonclosure_cache.npz` | Figs. 1, S17 |
| `eval_replicate_et_nse.py` | `fig3_replicate_nse.npz` | input to `eval_replicate_frozen.py` |
| `eval_replicate_frozen.py` | `fig3_replicate_nse_frozen.npz` | Fig. 3, Table B1 |
| `shap_fig6_cache.py` | `fig6_shap_ext_cache_geo.npz` | Figs. 6, S18 and the surrogate R^2 (Sect. 3.2) |
| `memory_truncation.py` | `memory_truncation.npz` (`_h16`, `_h48` with `--tag`) | Figs. 8, S12, S13 |
| `nonclosure_relation_cache.py` | `fig10_nonclosure_v2_cache.pkl` | Fig. 9 |
| `rin_surrogate_cv.py` | `rin_surrogate_cv_cache.npz` | R_in surrogate R^2 (Sect. 3.4) |
| `loss_normalization_compare.py` | `loss_normalization_compare.npz` (pooled-loss and catchment-normalized-loss recurrent models, held-out F_add and NSE) | Fig. S3 |
| `largebasins_compare.py` | `largebasins_compare.npz` (main model and the retrain on the 362 catchments > 1000 km², held-out F_add and NSE) | Fig. S7 |
| `fadd_wholeperiod_counterfactual.py` | `fadd_wholeperiod_counterfactual.npz` | Fig. 10 (ET contribution of F_add, Methods Eq. 12) |
| `fig_si_stability.py` | `fig_si_stability.npz` (catchment-mean F_add of each of the 25 recurrent instances) | Figs. 5 (their mean), S4, the F_add target of the surrogate in Figs. 6, S8 and S18, and the Fig. 5 vs held-out comparison of Fig. S6 |
| `fig_si_repr_heldout.py` | `fig_si_repr_heldout.npz` | Figs. S5, S6, S9 |
| `fig_si_grouped_shap.py` | `fig_si_grouped_shap_cv_cache.npz` (18 predictors, 7 groups, renders S8 as well) | Fig. S8 |
| `shap_seasonal_cache.py` | `fig_si_s12_seasonal_geo_cv_cache.npz` | Fig. S11 |
| `neg_fadd_compare.py` | `neg_fadd_compare.npz` | Fig. S15 |
| `loss_weight_compare.py` | `loss_weight_compare.npz` | Fig. S16 |

`fadd_bound_sensitivity.py` reads the default- and doubled-bound recurrent runs for Fig. S2a
(imported by `fig_si_boundS2_s11layout.py`, and prints the bound-insensitivity correlations when
run directly).

### Shared library (`scripts/analysis/`)
| file | purpose |
|---|---|
| `project_paths.py` | repository root, results directory, data root and basin list (see *Paths*) |
| `plot_style.py` | shared figure style constants, map formatting, colorbar helpers |
| `basin_attributes.py` | catchment coordinate and attribute tables, Koppen-Geiger classes |
| `residual_utils.py` | daily/monthly series helpers and the long-term water-balance residual |
| `metrics.py` | NSE and per-catchment skill tables |
| `run_analysis.py` | loading of the averaged outputs, the precipitation concentration index, SHAP plotting helpers |
| `manuscript_figures.py` | Fig. 5 drawing logic and the 14 base surrogate predictors |
| `run_io.py` | readers for the averaged outputs used by the post-hoc comparisons |

---

## Figures

Two drivers render all code-generated figures into the two final-figure directories:

```
python scripts/analysis/make_all_figures.py         # -> results_.../figures_manuscript_final/
python scripts/analysis/make_all_si_figures.py      # -> results_.../figures_si_final/
# subsets:  --only 3,6,A   /   --only 1,8,17
```

### Main text and appendix (`figures_manuscript_final/`)
| Fig | output | script (`scripts/analysis/`) | data |
|---|---|---|---|
| 1 | `fig_1_nonclosure.png` | `fig1_nonclosure_inset.py` | `fig_budyko_nonclosure_cache.npz` |
| 3 | `fig_3_heldout_performance.png` | `fig3_slopegraph.py` | `fig3_replicate_nse_frozen.npz` |
| 4 | `fig_4_paired_nse.png` | `fig_paired_nse_scatter.py` | averaged pickles (baseline, recurrent) |
| 5 | `fig_5_spatial_subsidy.png` | `manuscript_figures.fig5_spatial_fadd_inset` | `fig_si_stability.npz` (mean of the 25 recurrent instances) |
| 6 | `fig_6_shap_drivers.png` | `fig6_shap_drivers_v2layout_geo.py` | `fig6_shap_ext_cache_geo.npz` |
| 7 | `fig_7_seasonal_context.png` | `regen_seasonality_merged.py` | recurrent averaged pickle |
| 8 | `fig_8_effective_memory.png` | `fig8_memory_inset.py` (with `memory_truncation_plot_v2.py`) | `memory_truncation.npz` |
| 9 | `fig_9_nonclosure_relation.png` | `fig9_relation_2panel.py` | `fig10_nonclosure_v2_cache.pkl` |
| 10 | `fig_10_fadd_contribution.png` | `fig_fadd_contribution_compare.py` | `fadd_wholeperiod_counterfactual.npz` |
| A | `fig_A_study_catchments.png` | `fig_catchment_kg_map.py` | recurrent averaged pickle, Koppen-Geiger grid |

Fig. 2 (framework schematic) is drawn by hand and not produced by code.

### Supplement (`figures_si_final/`)
| Fig | output | script (`scripts/analysis/`) |
|---|---|---|
| S1 | `fig_si_01_coefficient_bound.png` | `fig_si_coef_bound.py` |
| S2 | `fig_si_02_bound_sensitivity.png` | `fig_si_boundS2_s11layout.py` |
| S3 | `fig_si_03_loss_normalization.png` | `fig_si_loss_normalization.py` |
| S4 | `fig_si_04_ensemble_stability.png` | `fig_si_stability_plot.py` |
| S5, S6 | `fig_si_05_representation_maps.png`, `fig_si_06_heldout_subsidy_map.png` | `fig_si_repr_heldout_plot.py` |
| S7 | `fig_si_07_large_catchments.png` | `fig_si_largebasins.py` |
| S8 | `fig_si_08_grouped_shap.png` | `fig_si_grouped_shap.py` |
| S9 | `fig_si_09_heldout_shap.png` | `fig_si_s10_heldout_shap_vertical.py` |
| S10 | `fig_si_10_morphology.png` | `fig_morph_geometry.py` |
| S11 | `fig_si_11_seasonal_shap.png` | `fig_si_s12_seasonal_shap_vertical.py` |
| S12 | `fig_si_12_antecedent_correlation.png` | `memory_truncation_plot_v2.build_map_only` |
| S13 | `fig_si_13_hidden_size_sweep.png` | `fig_si_memory_hidden_sweep.py` |
| S14 | `fig_si_14_lowclosure_et_gain.png` | `fig_si_lowRin_et_s11layout.py` |
| S15 | `fig_si_15_signed_fadd.png` | `fig_neg_fadd.py` |
| S16 | `fig_si_16_loss_weight_ablation.png` | `fig_wcmp_combined.py` |
| S17 | `fig_si_17_relative_subsidy.png` | `fig_fadd_relative_maps.py` |
| S18 | `fig_si_18_shap_dependence.png` | `fig6_shap_drivers_v3_geo.py` |

S-numbers follow the Supplement. Some script names carry older S-numbers or layout tags (e.g.
`fig_si_s10_...` renders Fig. S9, `fig_si_s12_...` renders Fig. S11, and the `*_s11layout` scripts
render Figs. S2 and S14).

### Tables
| table | script |
|---|---|
| A1 (catchment characteristics) | `scripts/analysis/table_catchment_characteristics.py` (also writes `table_A1_catchment_characteristics.csv`) |
| B1 (held-out pooled NSE per configuration) | `scripts/analysis/table_a1.py` (the file name keeps an earlier numbering) |

Table 1 (representation hypotheses) and Table S1 (list of surrogate predictors) are descriptive.

---

## Experiment configurations (`configs/`)

| config | experiment |
|---|---|
| `config_lateral_baseline.py` | no-subsidy baseline |
| `config_lateral_mlp_a.py` | static encoder (attributes only) |
| `config_lateral_mlp_af.py` | instantaneous encoder (attributes + concurrent forcing) |
| `config_lateral_lstm_af.py` | recurrent encoder (attributes + hydroclimatic history); also the frozen-decoder run |
| `config_lateral_lstm_af_h16.py`, `config_lateral_lstm_af_h48.py` | recurrent, hidden size 16 and 48 (Fig. S13) |
| `config_lateral_lstm_af_cap002.py` | recurrent, doubled coefficient bound [0, 0.02] (Fig. S2) |
| `config_lateral_mlp_a_cap002.py`, `config_lateral_mlp_af_cap002.py` | static and instantaneous, doubled coefficient bound [0, 0.02] (Fig. S2b) |
| `config_lateral_lstm_af_nsestar.py` | recurrent, catchment-normalized loss (NSE*, `LOSS_TYPE = "nse_star"`) (Fig. S3) |
| `config_lateral_lstm_af_neg.py` | recurrent, signed subsidy (Fig. S15) |
| `config_lateral_lstm_af_twsa0.py`, `config_lateral_lstm_af_q0.py` | recurrent, TWSA or discharge loss weight set to zero (Fig. S16) |
| `basin_list_global_grdc.txt` | the GRDC-Caravan catchments the models are trained and evaluated on (default basin list of the SLURM scripts and `project_paths.py`) |
| `basin_list_largebasins_ge1000km2.txt` | the 362 catchments with drainage area > 1000 km² (Fig. S7). The large-catchment model is `config_lateral_lstm_af.py` trained on this list (`slurm/submit_train_global.sh lstm_af_largebasins configs/config_lateral_lstm_af.py configs/basin_list_largebasins_ge1000km2.txt`) |
| `config_base.py`, `config_global_spatial.py` | shared settings and the GRDC-Caravan data setup imported by the configs above |

## SLURM scripts (`slurm/`)

Submit from the repository root (`sbatch slurm/<script>.sh ...`) with the Python environment
activated, after creating `results_lateral_experiments/logs/` once. Relative config and basin-list
arguments are resolved against the repository root. The data loader keeps its `cache/` in the
working directory. To set a fixed interpreter, a different working directory or another default
basin list, create `slurm/site_local.sh` defining `PYTHON_BIN`, `WORK_DIR`, `CODE_DIR` or
`BASIN_LIST_DEFAULT`; every job script sources it when present.

| script | job |
|---|---|
| `submit_train_global.sh` | train one configuration (array over seeds x folds) |
| `submit_train_frozen_v2.sh` | train the frozen-decoder recurrent run |
| `submit_build_averaged.sh` | average the 25 instances of one run |
| `submit_hsweep_mem_full.sh` | average a hidden-size run and run its history-truncation experiment |
| `submit_analysis_gpu.sh` | generic GPU wrapper for scripts that re-run the trained models |
| `submit_analysis_cpu.sh` | generic CPU wrapper for the surrogates and cache producers |
| `submit_grouped_shap.sh` | Fig. S8 grouped-SHAP surrogate |
