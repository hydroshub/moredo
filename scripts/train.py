import argparse
import importlib.util
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def parse_args():
    """Parse the training CLI arguments (config path and experiment name)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin_list", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config.py")
    parser.add_argument("--fold_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="model_output_test")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override config.NUM_EPOCHS")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--cv_seed", type=int, default=None, help="Override config.SPATIAL_CV_SEED")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cpu, cuda:0). Defaults to cuda if available.")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to a checkpoint to load (strict=False) before training. Used by the "
                             "frozen-decoder ablation: initialise the MOREDO parameter networks from a "
                             "trained no-subsidy baseline. Subsidy-encoder keys absent from the checkpoint "
                             "stay randomly initialised.")
    parser.add_argument("--freeze_subsidy_only", action="store_true",
                        help="Freeze every parameter except the lat_alpha subsidy encoder, so only the "
                             "latent subsidy is learned on top of a fixed decoder. Isolates the added "
                             "process from decoder reparameterisation.")
    parser.add_argument("--zero_init_subsidy", action="store_true",
                        help="Warm-start the lat_alpha output head so F_add begins near zero (bias set "
                             "strongly negative, head weights shrunk). Every fold then starts at the "
                             "baseline and only grows the subsidy where it reduces the loss, which "
                             "stabilises the frozen-decoder training (avoids the bad local optima).")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override config.LEARNING_RATE (e.g. 1e-3 for the stabilised frozen re-run).")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Override config.EARLY_STOP_PATIENCE (raise it so slow-but-correct descent "
                             "is not cut off).")
    return parser.parse_args()


def main():
    """Train the MOREDO + subsidy-encoder model instances (5 seeds x 5 folds) for the given config."""
    args = parse_args()
    from libs.utils import load_config, set_seed, train_model, evaluate_model, print_run_summary
    config = load_config(args.config)
    if args.cv_seed is not None:
        config.SPATIAL_CV_SEED = args.cv_seed
    from libs.data import (load_dataset_with_cache,
                           RandomSpatialCV, PredefinedSpatialCV, TimeBlockCV, UnifiedDataset)
    from libs.model import HybridModel

    # ----- Setup -----
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = args.num_epochs if args.num_epochs is not None else config.NUM_EPOCHS

    cv_mode = config.CV_MODE  # "spatial" or "temporal"

    # ----- Load data -----
    x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids = load_dataset_with_cache(
        data_dir=config.DATA_DIR,
        basin_list_path=args.basin_list,
        attr_paths=config.ATTR_PATHS,
        input_cols=config.INPUT_COLS,
        target_daily_cols=config.TARGET_DAILY_COLS,
        target_monthly_cols=config.TARGET_MONTHLY_COLS,
        nn_attr_cols=config.ATTR_NN_COLS,
        start_year=config.SPINUP_START_YEAR,
        end_year=config.TRAIN_END_YEAR,
        passthrough_cols=config.ATTR_PASSTHROUGH_COLS,
    )

    # ----- Init CV (both modes use valid_ids from load) -----
    if cv_mode == "spatial":
        if config.SPATIAL_CV_METHOD == "random":
            cv = RandomSpatialCV(valid_ids, num_folds=config.NUM_FOLDS, spinup_until=int(config.TRAIN_START_YEAR), seed=config.SPATIAL_CV_SEED)
        else:
            spec = importlib.util.spec_from_file_location("cv_folds", os.path.abspath(config.SPATIAL_CV_METHOD))
            fold_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fold_module)
            cv = PredefinedSpatialCV(valid_ids, fold_dict=fold_module.FOLD_DICT, spinup_until=int(config.TRAIN_START_YEAR))
    elif cv_mode == "temporal":
        cv = TimeBlockCV(years=idx_daily.year, num_folds=config.NUM_FOLDS, spinup_until=int(config.TRAIN_START_YEAR))
    else:
        raise ValueError(f"Unknown CV_MODE: '{cv_mode}'. Expected 'spatial' or 'temporal'.")

    cv.print_summary(args.fold_id, valid_ids, idx_daily)

    # ----- Generate time masks -----
    mask_daily   = cv.get_full_mask_dict(args.fold_id, idx_daily)
    mask_monthly = cv.get_full_mask_dict(args.fold_id, idx_monthly)

    # ----- Prepare datasets -----
    if cv_mode == "spatial":
        train_idx, val_idx, test_idx = cv.get_spatial_indices(args.fold_id, valid_ids)
        train_data = UnifiedDataset(x_all[train_idx], y_all[train_idx], z_all[train_idx], mask_daily["train_mask"], mask_monthly["train_mask"])
        val_data   = UnifiedDataset(x_all[val_idx],   y_all[val_idx],   z_all[val_idx],   mask_daily["val_mask"],   mask_monthly["val_mask"])
        test_data  = UnifiedDataset(x_all[test_idx],  y_all[test_idx],  z_all[test_idx],  mask_daily["test_mask"],  mask_monthly["test_mask"])
    else:
        train_data = UnifiedDataset(x_all, y_all, z_all, mask_daily["train_mask"], mask_monthly["train_mask"])
        val_data   = UnifiedDataset(x_all, y_all, z_all, mask_daily["val_mask"],   mask_monthly["val_mask"])
        test_data  = UnifiedDataset(x_all, y_all, z_all, mask_daily["test_mask"],  mask_monthly["test_mask"])

    # ----- DataLoaders -----
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=config.BATCH_SIZE, shuffle=False)

    # ----- Model & training -----
    model = HybridModel(
        param_config=config.PARAM_CONFIG,
        nn_attr_cols=config.ATTR_NN_COLS,
        passthrough_cols=config.ATTR_PASSTHROUGH_COLS,
        input_cols=config.INPUT_COLS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
    ).to(device)
    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)

    # ----- Frozen-decoder ablation: warm-start MOREDO params from a trained baseline, then train
    #       only the subsidy encoder so the added process cannot come from decoder reparameterisation.
    if args.init_from is not None:
        sd = torch.load(args.init_from, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        subsidy_missing = [k for k in missing if "lat_alpha" in k]
        decoder_missing = [k for k in missing if "lat_alpha" not in k]
        print(f"[init_from] loaded {args.init_from}")
        print(f"[init_from] {len(sd)} tensors; subsidy-encoder keys left random: {len(subsidy_missing)}")
        if decoder_missing:
            raise RuntimeError(f"[init_from] decoder keys missing from checkpoint (mismatch): {decoder_missing}")
        if unexpected:
            raise RuntimeError(f"[init_from] unexpected keys in checkpoint: {unexpected}")

    if args.freeze_subsidy_only:
        trainable, frozen = 0, 0
        for name, p in model.named_parameters():
            p.requires_grad = ("lat_alpha" in name)
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        if trainable == 0:
            raise RuntimeError("[freeze_subsidy_only] no lat_alpha parameters found to train.")
        print(f"[freeze_subsidy_only] trainable (subsidy encoder): {trainable}  frozen (decoder): {frozen}")

    if args.zero_init_subsidy:
        # Locate the lat_alpha subsidy generator regardless of encoder type and grab its final
        # linear layer (the LSTM encoder exposes it as `.head`, the static/instantaneous MLP
        # encoders as the last module of their `.network` Sequential).
        gen = None
        for mod in (model.param_dyn_lstm, model.param_dyn_driver, model.param_generators):
            if "lat_alpha" in mod:
                gen = mod["lat_alpha"]; break
        if gen is None:
            raise RuntimeError("[zero_init_subsidy] no lat_alpha generator found.")
        if hasattr(gen, "head"):
            out_lin = gen.head                       # LSTM encoder
        elif hasattr(gen, "network"):
            out_lin = gen.network[-1]                # static / instantaneous MLP encoder
        else:
            raise RuntimeError("[zero_init_subsidy] lat_alpha output layer not found.")
        with torch.no_grad():
            out_lin.weight.mul_(0.1)   # keep gradients flowing to the encoder, but small
            out_lin.bias.fill_(-4.0)   # sigmoid(-4)~=0.018 -> F_add starts near zero
        print(f"[zero_init_subsidy] lat_alpha output layer: weight*=0.1, bias=-4.0 "
              f"({type(gen).__name__}, F_add starts ~0)")

    n_total = sum(1 for line in open(args.basin_list) if line.strip())
    print_run_summary(x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids, n_total, model)

    model_path = save_dir / f"best_model_seed{args.seed}_fold{args.fold_id}.pt"
    log_path   = save_dir / f"training_progress_seed{args.seed}_fold{args.fold_id}.txt"

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_path=model_path,
        log_path=log_path,
        num_epochs=num_epochs,
        seed=args.seed,
        early_stop_patience=(args.early_stop_patience if args.early_stop_patience is not None else config.EARLY_STOP_PATIENCE),
        learning_rate=(args.learning_rate if args.learning_rate is not None else config.LEARNING_RATE),
        loss_weights=config.LOSS_WEIGHTS,
        date_seq=date_seq,
        show_progress=args.show_progress,
    )

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    # ----- Evaluate best model on test set -----
    test_nse, test_nse_basin = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        date_seq=date_seq,
    )
    eval_path = save_dir / f"eval_seed{args.seed}_fold{args.fold_id}.txt"
    with open(eval_path, "w") as f:
        for k, v in test_nse.items():
            f.write(f"nse_{k}: {v:.4f}\n")
    # basin-averaged NSE over valid basins, in a separate file so the pooled-NSE readers are unchanged
    eval_basin_path = save_dir / f"eval_basin_seed{args.seed}_fold{args.fold_id}.txt"
    with open(eval_basin_path, "w") as f:
        for k, b in test_nse_basin.items():
            f.write(f"{k}: median={b['median']:.4f} mean={b['mean']:.4f} "
                    f"n_valid={b['n_valid']} n_total={b['n_total']}\n")


if __name__ == "__main__":
    main()
