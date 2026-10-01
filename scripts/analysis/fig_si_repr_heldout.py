"""
Compute for SI figures S5 (representation comparison) and S6 (held-out-only F_add).

S5: 2001-2020 ensemble-mean catchment-mean F_add for the STATIC (mlp_a) encoder. mlp_a's
    lat_alpha is time-invariant, so F_add = lat_alpha * avai_cap is read from a short forward per
    checkpoint and averaged over the 25 instances. (The instantaneous mlp_af and recurrent lstm_af
    maps are read from their averaged pickles at plot time.)

S6: held-out-only catchment-mean F_add for the RECURRENT encoder. Each catchment is held out in
    exactly one fold per seed; for each (seed, fold) we compute the full-history F_add for that
    fold's test catchments only (LSTM replay), then average each catchment's five held-out
    estimates (one per seed). This is the strictly out-of-sample analogue of Fig. 5, and also the
    target used by SI S9 (SHAP on held-out F_add).

Device-aware; run on a GPU Slurm node. Saves fig_si_repr_heldout.npz.

Usage (Slurm GPU):
  sbatch --job-name=moredo-si-repr slurm/submit_analysis_gpu.sh fig_si_repr_heldout.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch

from project_paths import REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from memory_truncation import (SEEDS, FOLDS, ACTIVE_START_YEAR, load_data, build_model,
                               load_checkpoint, driver_and_attr_tensors, static_avai_cap)
from fig_si_stability import full_lat_alpha_active_mean
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")
CFG = {"mlp_a": "config_lateral_mlp_a.py", "lstm_af": "config_lateral_lstm_af.py"}


def short_forward_fadd(model, x_all, idx_daily, device, n=8):
    """Static F_add = lat_alpha * avai_cap from a short forward (valid when lat_alpha is static)."""
    date_seq = torch.tensor([d.toordinal() for d in idx_daily[:n]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x_all[:, :n, :].to(device), date_seq=date_seq, mode="full",
                    compute_twsa=False, show_progress=False)
    lat = out["params"]["lat_alpha"][:, 0, 0]
    avai = out["params"]["avai_cap"][:, 0, 0]
    return (lat * avai).cpu().numpy()


def main():
    """Compute the strictly held-out (test-fold-only) F_add map plus the static and instantaneous means on GPU and write fig_si_repr_heldout.npz."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from libs.utils import load_config
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    cfg_lstm = load_config(os.path.join(REPO, "configs", CFG["lstm_af"]))
    x_all, idx_daily, valid_ids = load_data(cfg_lstm)
    from libs.data import RandomSpatialCV
    years = np.array([d.year for d in idx_daily])
    active = years >= ACTIVE_START_YEAR
    B = x_all.shape[0]

    # ---- S5: static (mlp_a) ensemble-mean F_add ----
    cfg_static = load_config(os.path.join(REPO, "configs", CFG["mlp_a"]))
    m_static = build_model(cfg_static, device)
    acc = np.zeros(B); n = 0
    for seed in SEEDS:
        for fold in FOLDS:
            load_checkpoint_dir(m_static, "mlp_a", seed, fold, device)
            acc += short_forward_fadd(m_static, x_all, idx_daily, device); n += 1
    fadd_static = acc / n
    print(f"[S5] static mlp_a mean F_add = {np.nanmean(fadd_static):.3f}", flush=True)

    # ---- S6: recurrent (lstm_af) held-out-only F_add ----
    cfg = load_config(os.path.join(REPO, "configs", CFG["lstm_af"]))
    model = build_model(cfg, device)
    driver_norm, attr_norm = driver_and_attr_tensors(model, x_all, cfg)
    # The training runs use cv_seed = training seed (slurm/submit_train_global.sh), so the held-out
    # catchments of model (seed, fold) are the test fold of that seed's own partition.
    fold_test = {s: {f: RandomSpatialCV(valid_ids, num_folds=cfg.NUM_FOLDS,
                                        spinup_until=int(cfg.TRAIN_START_YEAR),
                                        seed=s).get_spatial_indices(f, valid_ids)[2]
                     for f in FOLDS} for s in SEEDS}

    heldout_seed = np.full((len(SEEDS), B), np.nan)
    for si, seed in enumerate(SEEDS):
        for fold in FOLDS:
            load_checkpoint(model, seed, fold, device)  # lstm_af dir (MODEL_DIR default)
            avai = static_avai_cap(model, x_all, idx_daily, device).numpy()
            lat = full_lat_alpha_active_mean(model, cfg, driver_norm, attr_norm, active, device)
            fadd_all = lat * avai
            idx = fold_test[seed][fold]
            heldout_seed[si, idx] = fadd_all[idx]   # keep only this fold's held-out catchments
        print(f"[S6] held-out seed={seed} assembled "
              f"({np.isfinite(heldout_seed[si]).sum()} catchments)", flush=True)
    fadd_heldout = np.nanmean(heldout_seed, axis=0)  # avg 5 held-out estimates per catchment
    print(f"[S6] held-out recurrent mean F_add = {np.nanmean(fadd_heldout):.3f}", flush=True)

    np.savez(os.path.join(OUT_DIR, "fig_si_repr_heldout.npz"),
             fadd_static=fadd_static, fadd_heldout=fadd_heldout,
             gauge_id=np.array(valid_ids, dtype=object))
    print(f"Saved -> {os.path.join(OUT_DIR, 'fig_si_repr_heldout.npz')}")


def load_checkpoint_dir(model, exp, seed, fold, device):
    """load_checkpoint variant for an arbitrary experiment directory (mlp_a here)."""
    path = os.path.join(REPO, "results_lateral_experiments", exp, f"best_model_seed{seed}_fold{fold}.pt")
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.eval()


if __name__ == "__main__":
    main()
