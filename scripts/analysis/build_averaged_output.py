"""
Build a per-basin, out-of-sample "full" model output (fluxes/states/params), averaged
across the repeated spatial-CV seeds, for one experiment (one config + one save_dir).

Unlike temporal CV (where every basin appears in every fold, just at different time
windows), our spatial CV holds each basin out in exactly one fold *per seed*. Because
SPATIAL_CV_SEED is set equal to the training seed at submission time, each seed defines
its own independent 5-way basin partition (RandomSpatialCV reshuffles basins per seed).
So for a fixed seed, stitching together the 5 folds' test-set predictions gives one
complete out-of-sample prediction for every basin. Repeating this for all 5 seeds gives
5 independent out-of-sample coverages (repeated 5-fold spatial CV), which we then average.

Output: a pickle with the same {'fluxes':..., 'states':..., 'params':...} dict structure
as HybridModel.forward(mode='full'), plus '_std' variants (across the 5 seeds) and
valid_ids / idx_daily / idx_monthly for downstream analysis.
"""
import argparse
import pickle
import sys
import numpy as np
import torch


def get_full_out(x_all, idx_daily, fold, seed, device, model_dir, config):
    """Run one trained seed/fold model over the full 1996-2020 forcing and return its full output dict."""
    from libs.model import HybridModel

    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)

    model = HybridModel(
        config.PARAM_CONFIG,
        nn_attr_cols=config.ATTR_NN_COLS,
        passthrough_cols=config.ATTR_PASSTHROUGH_COLS,
        input_cols=config.INPUT_COLS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
    ).to(device)
    state_dict = torch.load(f"{model_dir}/best_model_seed{seed}_fold{fold}.pt",
                             weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        full_out = model(x_all.to(device), date_seq=date_seq, mode='full')
    return full_out


def to_cpu_numpy(x):
    """Recursively move tensors in a nested structure to CPU numpy arrays."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    """Average the 25 seed/fold model outputs into averaged_full_output.pickle for one experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.py")
    parser.add_argument("--basin_list", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True, help="Dir with best_model_seed{s}_fold{f}.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, required=True, help="Output pickle path")
    args = parser.parse_args()

    from libs.utils import load_config
    config = load_config(args.config)
    from libs.data import load_dataset_with_cache, RandomSpatialCV

    print("Loading dataset (reuses shared cache if available)...")
    x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids = load_dataset_with_cache(
        data_dir=config.DATA_DIR,
        basin_list_path=args.basin_list,
        attr_paths=config.ATTR_PATHS,
        input_cols=config.INPUT_COLS,
        target_daily_cols=config.TARGET_DAILY_COLS,
        target_monthly_cols=config.TARGET_MONTHLY_COLS,
        nn_attr_cols=config.ATTR_NN_COLS,
        start_year=config.INFER_START_YEAR,
        end_year=config.INFER_END_YEAR,
        passthrough_cols=config.ATTR_PASSTHROUGH_COLS,
    )
    n_basins = len(valid_ids)
    print(f"  n_basins={n_basins}, x_all={tuple(x_all.shape)}")

    per_seed_coverage = []  # list of dicts: {'fluxes': {...}, 'states': {...}, 'params': {...}}

    for seed in args.seeds:
        print(f"=== seed {seed} ===")
        cv = RandomSpatialCV(valid_ids, num_folds=args.num_folds,
                              spinup_until=int(config.TRAIN_START_YEAR), seed=seed)

        coverage = {'fluxes': {}, 'states': {}, 'diagnos': {}, 'params': {}}
        covered_mask = np.zeros(n_basins, dtype=bool)

        for fold in range(args.num_folds):
            _, _, test_idx = cv.get_spatial_indices(fold, valid_ids)
            if not test_idx:
                continue
            print(f"  fold {fold}: {len(test_idx)} test basins")

            full_out = get_full_out(x_all, idx_daily, fold, seed, args.device, args.model_dir, config)

            for group in ['fluxes', 'states', 'diagnos', 'params']:
                for key, val in full_out[group].items():
                    val_np = to_cpu_numpy(val)
                    if key not in coverage[group]:
                        coverage[group][key] = np.full((n_basins,) + val_np.shape[1:], np.nan, dtype=np.float32)
                    coverage[group][key][test_idx] = val_np[test_idx].astype(np.float32)

            covered_mask[test_idx] = True
            del full_out
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        n_missing = int((~covered_mask).sum())
        if n_missing > 0:
            print(f"  WARNING: {n_missing}/{n_basins} basins not covered by any fold for seed {seed}")
        per_seed_coverage.append(coverage)

    # Stack across seeds and average (+ std) per key
    print("Averaging across seeds...")
    averaged = {'fluxes': {}, 'states': {}, 'diagnos': {}, 'params': {}}
    averaged_std = {'fluxes': {}, 'states': {}, 'diagnos': {}, 'params': {}}
    for group in ['fluxes', 'states', 'diagnos', 'params']:
        keys = per_seed_coverage[0][group].keys()
        for key in keys:
            stacked = np.stack([cov[group][key] for cov in per_seed_coverage], axis=0)  # [n_seeds, basins, ...]
            averaged[group][key] = np.nanmean(stacked, axis=0)
            averaged_std[group][key] = np.nanstd(stacked, axis=0)

    result = {
        'fluxes': averaged['fluxes'],
        'states': averaged['states'],
        'diagnos': averaged['diagnos'],
        'params': averaged['params'],
        'fluxes_std': averaged_std['fluxes'],
        'states_std': averaged_std['states'],
        'diagnos_std': averaged_std['diagnos'],
        'params_std': averaged_std['params'],
        'valid_ids': valid_ids,
        'idx_daily': idx_daily,
        'idx_monthly': idx_monthly,
        'y_all': to_cpu_numpy(y_all),
        'z_all': to_cpu_numpy(z_all),
        'seeds': args.seeds,
        'num_folds': args.num_folds,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
