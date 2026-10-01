# Changelog

## [latent-subsidy] - 2026-10 - latent subsidy extension (Greiner et al.)

Code of Greiner et al., *Physics-constrained latent process learning for hidden subsurface water
subsidies to ecosystems* (in preparation). Without a `lat_alpha` entry in `PARAM_CONFIG` the model
is the 0.0.1 release model. The experiments and the analysis pipeline are described in
`README_latent_subsidy.md`.

### Model (`libs/model.py`)
- Learned latent subsidy flux `f_lat` (F_add) in `avai_bucket`: `f_lat = lat_alpha * avai_cap`,
  added to the ecosystem-available store before ET and runoff partitioning. `lat_alpha` is a
  bounded parameter in `PARAM_CONFIG` (bound [0, 0.01] in the experiments). A clamp at `-s_pre`
  keeps storage non-negative for the signed-subsidy variant and is inactive for the default
  non-negative subsidy.
- Parameter modes besides `static`: `constant` (global learnable scalar), `dynamic_physical_state`
  (MLP on the scaled states), `dynamic_driver_state` (MLP on the current drivers, the
  instantaneous encoder) and `dynamic_lstm_state` (LSTMCell over the driver sequence, the
  recurrent encoder). `HybridModel` takes `lstm_hidden_size`, and `forward(mode='full')` returns
  the LSTM hidden state, which can be passed back with `hidden=`.
- The static `lat_alpha` generator uses hidden width 32, like the dynamic encoders.
- Optional counterfactual hooks in `forward` (`fadd_gate`, `reset_mask`, `ref_S_AVAI`,
  `ref_relmoist`), no-ops by default.

### Data and training (`libs/data.py`, `libs/utils.py`, `scripts/`)
- `TimeBlockCV` for temporal cross-validation (`CV_MODE = "temporal"`), log1p normalization also
  for perimeter attributes.
- Optional catchment-normalized loss NSE* (Kratzert et al., 2019), `LOSS_TYPE = "nse_star"`. The
  default remains the pooled NSE loss.
- `evaluate_model` also returns the per-catchment test NSE (median and mean over valid
  catchments), written to `eval_basin_seed{seed}_fold{fold}.txt`.
- The optimizer only receives parameters with `requires_grad`.
- `train.py`: `--init_from`, `--freeze_subsidy_only` and `--zero_init_subsidy` for the
  frozen-decoder ablation, `--learning_rate` and `--early_stop_patience` overrides.
- `inference_basin.py` and `inference_grid.py` pass `lstm_hidden_size`, and the grid inference
  carries the LSTM hidden state across spin-up cycles and years.

### Configs (`configs/`)
- `config_base.py`: adds `LSTM_HIDDEN_SIZE`, `LOSS_TYPE`, `NSE_STAR_EPS`, `NSE_MIN_OBS`,
  `DRIVER_NORM` and `STATE_NORM`. The 0.0.1 defaults are unchanged.
- `config_global_spatial.py`: the GRDC-Caravan setup imported by the experiment configs.
- Experiment configs: no-subsidy baseline (`config_lateral_baseline.py`), static (`mlp_a`),
  instantaneous (`mlp_af`) and recurrent (`lstm_af`) encoders, the hidden-size sweep
  (`lstm_af_h16`, `lstm_af_h48`), the doubled coefficient bound (`*_cap002`), the signed subsidy
  (`lstm_af_neg`), the loss-weight ablations (`lstm_af_twsa0`, `lstm_af_q0`) and the
  catchment-normalized loss (`lstm_af_nsestar`).
- `basin_list_global_grdc.txt` (the study catchments) and `basin_list_largebasins_ge1000km2.txt`
  (the 362 catchments of the large-catchment retrain).

### Analysis (`scripts/analysis/`, `slurm/`)
- The diagnostic and figure pipeline of the manuscript and its Supplement, including
  `build_averaged_output.py` (averages the 25 seed x fold instances into one output per
  experiment). All locations are set in `scripts/analysis/project_paths.py`.
- SLURM job scripts for training, averaging and the analyses. They take the repository root from
  the submission directory, and machine-specific settings go into an optional
  `slurm/site_local.sh`.
- Analysis dependencies added to `requirements.txt`.

---

## [0.0.1] - 2026-06-12

- First public pre-release.
