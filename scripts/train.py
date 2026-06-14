import argparse
import importlib.util
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def parse_args():
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
    return parser.parse_args()


def main():
    args = parse_args()
    from libs.utils import load_config, set_seed, train_model, evaluate_model, print_run_summary
    config = load_config(args.config)
    if args.cv_seed is not None:
        config.SPATIAL_CV_SEED = args.cv_seed
    from libs.data import (load_dataset_with_cache,
                           RandomSpatialCV, PredefinedSpatialCV, UnifiedDataset)
    from libs.model import HybridModel

    # ----- Setup -----
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = args.num_epochs if args.num_epochs is not None else config.NUM_EPOCHS

    cv_mode = config.CV_MODE  # "spatial"

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
    else:
        raise ValueError(f"Unknown CV_MODE: '{cv_mode}'. Expected 'spatial'.")

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
        raise ValueError(f"Unknown CV_MODE: '{cv_mode}'. Expected 'spatial'.")

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
    ).to(device)
    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)

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
        early_stop_patience=config.EARLY_STOP_PATIENCE,
        learning_rate=config.LEARNING_RATE,
        loss_weights=config.LOSS_WEIGHTS,
        date_seq=date_seq,
        show_progress=args.show_progress,
    )

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    # ----- Evaluate best model on test set -----
    test_nse = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        date_seq=date_seq,
    )
    eval_path = save_dir / f"eval_seed{args.seed}_fold{args.fold_id}.txt"
    with open(eval_path, "w") as f:
        for k, v in test_nse.items():
            f.write(f"nse_{k}: {v:.4f}\n")


if __name__ == "__main__":
    main()
