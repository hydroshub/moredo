import os
import sys
import importlib
import torch
import random
import warnings
import numpy as np



def load_config(config_path):
    config_path = os.path.abspath(config_path)
    sys.path.insert(0, os.path.dirname(config_path))  # allow configs to inherit each other
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    sys.modules["config"] = config
    return config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_run_summary(x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids, n_total, model):
    import config
    W = 54
    SEP = "─" * W
    print(SEP)
    print("Data summary")
    print(f"  Basins loaded : {len(valid_ids)}  (skipped: {n_total - len(valid_ids)})")
    n_drivers = len(config.INPUT_COLS)
    n_normed  = len(config.ATTR_NN_COLS)
    n_raw     = len(config.ATTR_PASSTHROUGH_COLS)
    print(f"  x_all  : {tuple(x_all.shape)}  [{n_drivers} drivers | {n_normed} normed attrs | {n_raw} raw attrs]")
    print(f"  y_all  : {tuple(y_all.shape)}  [daily targets]")
    print(f"  z_all  : {tuple(z_all.shape)}  [monthly targets]")
    print(f"  Time (daily)  : {idx_daily[0].date()} → {idx_daily[-1].date()}")
    print(f"  Time (monthly): {idx_monthly[0].date()} → {idx_monthly[-1].date()}")
    print(SEP)
    print("Model summary")
    mode_groups = {"static": []}
    for pname, (bounds, mode, _) in config.PARAM_CONFIG.items():
        mode_groups[mode].append(f"{pname} {bounds}")
    labels = {
        "static":                 "Static parameter(s)      (MLP, attr → param)",
    }
    for mode, params in mode_groups.items():
        if params:
            print(f"  {labels[mode]} ×{len(params)}")
            for p in params:
                print(f"    {p}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")
    print(SEP)


def masked_nse_metric(pred, target, eps=1e-6):
    """NSE as a metric (higher = better). Accepts pre-concatenated tensors."""
    valid = torch.isfinite(target)
    pred   = pred[valid]
    target = target[valid]

    if target.numel() == 0:
        return float("nan")

    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1.0 - ss_res / (ss_tot + eps)).item()


def masked_nse_loss(pred, target, eps=1e-6):
    valid = torch.isfinite(target)  # more robust than ~torch.isnan
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        warnings.warn("masked_nse_loss: No valid elements! Returning 0.0")
        return torch.tensor(0.0, device=target.device)

    # If target is constant, R2 is defined as 0.0
    if torch.allclose(target, target.mean()):
        warnings.warn("masked_nse_loss: Target is constant! Returning 0.0")
        return torch.tensor(0.0, device=target.device)

    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)

    return ss_res / (ss_tot + eps)


def compute_losses(pred, y, z):
    import config
    m = config.TARGET_COLS_MAP
    i_q   = config.TARGET_DAILY_COLS.index(m["q"])
    i_et  = config.TARGET_DAILY_COLS.index(m["et"])
    i_swe = config.TARGET_DAILY_COLS.index(m["swe"])
    i_twsa = 0  # always single monthly target
    return {
        "q":    masked_nse_loss(pred["q"][:, :, 0],             y[:, :, i_q]),
        "et":   masked_nse_loss(pred["et"][:, :, 0],            y[:, :, i_et]),
        "swe":  masked_nse_loss(pred["swe"][:, :, 0],           y[:, :, i_swe]),
        "twsa": masked_nse_loss(pred["twsa"][:, :, 0],  z[:, :, i_twsa]),
    }


def train_model(
    model, train_loader, val_loader, device, model_path, log_path,
    num_epochs=1000, seed=42, early_stop_patience=10, learning_rate=1e-2,
    loss_weights=[1.0, 1.0, 1.0, 1.0], grad_clip=10.0, date_seq=None, show_progress=False
):
    # Set random seed
    set_seed(seed)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    no_improve_epochs = 0

    weights = torch.tensor(loss_weights, device=device)

    for epoch in range(1, num_epochs + 1):
        # -----
        # Training phase
        # -----
        model.train()
        train_sums = {k: 0.0 for k in ["q", "et", "swe", "twsa"]}
        total_train_loss = 0.0

        for xb, yb, zb in train_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

            optimizer.zero_grad()
            pred = model(xb, date_seq=date_seq, mode="supervised", show_progress=show_progress)

            # -----
            # Loss calculation
            # -----
            loss_dict = compute_losses(pred, yb, zb)

            loss = (
                weights[0] * loss_dict["q"] +
                weights[1] * loss_dict["et"] +
                weights[2] * loss_dict["swe"] +
                weights[3] * loss_dict["twsa"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()

            total_train_loss += loss.item()
            for k in train_sums:
                train_sums[k] += loss_dict[k].item()

        # -----
        # Validation phase
        # -----
        model.eval()
        val_sums = {k: 0.0 for k in train_sums}
        total_val_loss = 0.0

        with torch.inference_mode():
            for xb, yb, zb in val_loader:
                xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

                pred = model(xb, date_seq=date_seq, mode="supervised", show_progress=show_progress)

                # Validation loss - no spinup masking needed
                loss_dict = compute_losses(pred, yb, zb)

                loss = (
                    weights[0] * loss_dict["q"] +
                    weights[1] * loss_dict["et"] +
                    weights[2] * loss_dict["swe"] +
                    weights[3] * loss_dict["twsa"]
                )
                total_val_loss += loss.item()
                for k in val_sums:
                    val_sums[k] += loss_dict[k].item()

        # -----
        # Logging and scheduler step
        # -----
        n_train, n_val = len(train_loader), len(val_loader)
        train_avg = {k: v / n_train for k, v in train_sums.items()}
        val_avg = {k: v / n_val for k, v in val_sums.items()}
        train_loss = total_train_loss / n_train
        val_loss = total_val_loss / n_val

        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print("  -> " + " | ".join([f"{k}: {train_avg[k]:.4f}/{val_avg[k]:.4f}" for k in train_avg]))

        scheduler.step(val_loss)

        # -----
        # Write log to file
        # -----
        log_line = (
            f"Epoch {epoch:03d} | Train: loss={train_loss:.4f}, " +
            ", ".join([f"{k}={train_avg[k]:.4f}" for k in train_avg]) +
            f" | Val: loss={val_loss:.4f}, " +
            ", ".join([f"{k}={val_avg[k]:.4f}" for k in val_avg])
        )
        write_mode = "w" if epoch == 1 else "a"
        with open(log_path, write_mode) as f:
            f.write(log_line + "\n")

        # -----
        # Early stopping
        # -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
            print("  -> Best model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break


def evaluate_model(model, test_loader, device, date_seq):
    import config
    m = config.TARGET_COLS_MAP
    i_q    = config.TARGET_DAILY_COLS.index(m["q"])
    i_et   = config.TARGET_DAILY_COLS.index(m["et"])
    i_swe  = config.TARGET_DAILY_COLS.index(m["swe"])

    # Accumulate SS_res and SS_tot per target across all batches
    targets = {
        "q":    lambda yb, zb, pred: (pred["q"][:, :, 0],    yb[:, :, i_q]),
        "et":   lambda yb, zb, pred: (pred["et"][:, :, 0],   yb[:, :, i_et]),
        "swe":  lambda yb, zb, pred: (pred["swe"][:, :, 0],  yb[:, :, i_swe]),
        "twsa": lambda yb, zb, pred: (pred["twsa"][:, :, 0], zb[:, :, 0]),
    }
    preds_list  = {k: [] for k in targets}
    targets_list = {k: [] for k in targets}

    model.eval()
    with torch.inference_mode():
        for xb, yb, zb in test_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            pred = model(xb, date_seq=date_seq, mode="supervised")
            for k, fn in targets.items():
                p, t = fn(yb, zb, pred)
                preds_list[k].append(p.cpu())
                targets_list[k].append(t.cpu())

    nse = {}
    for k in targets:
        p_all = torch.cat(preds_list[k])
        t_all = torch.cat(targets_list[k])
        nse[k] = masked_nse_metric(p_all, t_all)

    print("\n===== Test Set NSE =====")
    for k, v in nse.items():
        print(f"  {k}: {v:.4f}")

    return nse

