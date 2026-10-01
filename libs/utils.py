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
    mode_groups = {"constant": [], "static": [], "dynamic_physical_state": [], "dynamic_driver_state": [], "dynamic_lstm_state": []}
    for pname, (bounds, mode, _) in config.PARAM_CONFIG.items():
        mode_groups[mode].append(f"{pname} {bounds}")
    labels = {
        "constant":               "Scalar parameter(s)      (global learnable)",
        "static":                 "Static parameter(s)      (MLP, attr → param)",
        "dynamic_physical_state": "Dynamic parameter(s)     (MLP, state+attr → param)",
        "dynamic_driver_state":   "Dynamic parameter(s)     (MLP, driver+attr → param)",
        "dynamic_lstm_state":     "Dynamic parameter(s)     (LSTM, driver+attr → param)",
    }
    for mode, params in mode_groups.items():
        if params:
            print(f"  {labels[mode]} ×{len(params)}")
            for p in params:
                print(f"    {p}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")
    print(SEP)
    print("Loss summary")
    loss_type = getattr(config, "LOSS_TYPE", "nse_pooled")
    print(f"  Loss type    : {loss_type}")
    if loss_type == "nse_star":
        print(f"  NSE* eps     : {config.NSE_STAR_EPS}")
        print(f"  Min obs/basin: {getattr(config, 'NSE_MIN_OBS', 10)}")
    print(f"  Loss weights : {dict(zip(['q', 'et', 'swe', 'twsa'], config.LOSS_WEIGHTS))}")
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


def valid_basin_stats(target, min_obs=10):
    """Per-basin statistics of a [basins, time] target over its finite time steps.

    A basin is valid for NSE when it has at least `min_obs` finite observations and a
    non-constant target (std > 0), the same rule as metrics.compute_basin_nse. Returns the
    element mask restricted to valid basins [B, T], the basin mask [B], and the per-basin
    mean and population std [B, 1] (detached)."""
    finite = torch.isfinite(target)
    n_b = finite.sum(dim=1, keepdim=True)
    t0 = torch.where(finite, target, torch.zeros_like(target))
    mean_b = t0.sum(dim=1, keepdim=True) / n_b.clamp(min=1)
    var_b = (torch.where(finite, target - mean_b, torch.zeros_like(target)) ** 2).sum(
        dim=1, keepdim=True) / n_b.clamp(min=1)
    s_b = var_b.sqrt().detach()
    basin_ok = (n_b >= min_obs) & (s_b > 0)                                   # [B, 1]
    return finite & basin_ok, basin_ok.squeeze(1), mean_b.detach(), s_b


def masked_nse_star_loss(pred, target, eps=0.1, min_obs=10):
    """Basin-normalized NSE* loss (Kratzert et al., 2019, Eq. 13).

    Squared errors are weighted by 1 / (s_b + eps)^2, where s_b is the std of the target in
    basin b over the time steps of this split, so every basin is scored relative to its own
    variability. The mean is taken over all valid (basin, time) elements, so basins with more
    observations weigh more, as in Kratzert et al. (2019). Only valid basins enter (at least
    `min_obs` observations and a non-constant target, see valid_basin_stats).

    pred, target: [basins, time]"""
    valid, basin_ok, _, s_b = valid_basin_stats(target, min_obs)
    if not valid.any():
        warnings.warn("masked_nse_star_loss: No valid basins! Returning 0.0")
        return torch.tensor(0.0, device=target.device)
    weights = 1.0 / (s_b + eps) ** 2                                           # [B, 1]
    sq_err = (pred - torch.where(valid, target, pred)) ** 2                    # 0 where invalid, no NaN in the graph
    return (weights * sq_err)[valid].mean()


def masked_nse_per_basin(pred, target, min_obs=10):
    """NSE of every basin (NaN for basins that are not valid, see valid_basin_stats).
    pred, target: [basins, time]. Returns a float tensor [basins]."""
    valid, basin_ok, mean_b, _ = valid_basin_stats(target, min_obs)
    t = torch.where(valid, target, torch.zeros_like(target))
    p = torch.where(valid, pred, torch.zeros_like(pred))
    ss_res = ((p - t) ** 2 * valid).sum(dim=1)
    ss_tot = ((t - mean_b) ** 2 * valid).sum(dim=1)
    nse = 1.0 - ss_res / ss_tot.clamp(min=1e-12)
    return torch.where(basin_ok, nse, torch.full_like(nse, float("nan")))


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

    loss_type = getattr(config, "LOSS_TYPE", "nse_pooled")
    if loss_type == "nse_star":
        eps = config.NSE_STAR_EPS
        min_obs = getattr(config, "NSE_MIN_OBS", 10)
        loss_fn = lambda p, t, k: masked_nse_star_loss(p, t, eps=eps[k], min_obs=min_obs)
    elif loss_type == "nse_pooled":
        loss_fn = lambda p, t, k: masked_nse_loss(p, t)
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {loss_type}")

    return {
        "q":    loss_fn(pred["q"][:, :, 0],    y[:, :, i_q],    "q"),
        "et":   loss_fn(pred["et"][:, :, 0],   y[:, :, i_et],   "et"),
        "swe":  loss_fn(pred["swe"][:, :, 0],  y[:, :, i_swe],  "swe"),
        "twsa": loss_fn(pred["twsa"][:, :, 0], z[:, :, i_twsa], "twsa"),
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
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=1e-5)
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

    min_obs = getattr(config, "NSE_MIN_OBS", 10)
    nse, nse_basin = {}, {}
    for k in targets:
        p_all = torch.cat(preds_list[k])
        t_all = torch.cat(targets_list[k])
        nse[k] = masked_nse_metric(p_all, t_all)
        per_basin = masked_nse_per_basin(p_all, t_all, min_obs=min_obs)
        ok = torch.isfinite(per_basin)
        nse_basin[k] = {
            "median": per_basin[ok].median().item() if ok.any() else float("nan"),
            "mean": per_basin[ok].mean().item() if ok.any() else float("nan"),
            "n_valid": int(ok.sum()), "n_total": int(per_basin.numel()),
        }

    print("\n===== Test Set NSE (pooled | basin median, mean over valid basins) =====")
    for k in targets:
        b = nse_basin[k]
        print(f"  {k}: {nse[k]:.4f} | {b['median']:.4f}, {b['mean']:.4f}  "
              f"(valid basins {b['n_valid']}/{b['n_total']})")

    return nse, nse_basin

