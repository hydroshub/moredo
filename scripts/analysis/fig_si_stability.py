"""
Fig. S4: cross-model stability of the catchment-mean F_add spatial pattern.

Computes the 2001-2020 catchment-mean F_add for EACH of the 25 trained instances (5 seeds x 5
folds), via full-history LSTM replay (validated to 1e-8 against the model's own lat_alpha). It
then forms:
  - a 25x25 matrix of pairwise spatial Pearson correlations between the instance patterns
    (ordered seed-major, fold-minor; blocks of 5 = one seed), and
  - five replicate maps (per-seed averages of the 5 folds) as small multiples.

COMPUTE ONLY here (device-aware, meant to run on a GPU Slurm node); saves fig_si_stability.npz.
Plotting is done separately by fig_si_stability_plot (no GPU needed).

Usage (Slurm GPU):
  sbatch --job-name=moredo-si-stability slurm/submit_analysis_gpu.sh fig_si_stability.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch

from project_paths import REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from memory_truncation import (CONFIG_PATH, SEEDS, FOLDS, ACTIVE_START_YEAR,
                               load_data, build_model, load_checkpoint,
                               driver_and_attr_tensors, static_avai_cap)
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")


def full_lat_alpha_active_mean(model, config, driver_norm, attr_norm, active, device, basin_chunk=600):
    """Full-history lat_alpha averaged over active days, per basin. [B]"""
    gen = model.param_dyn_lstm["lat_alpha"]
    lo, hi = config.PARAM_CONFIG["lat_alpha"][0]
    B, T, _ = driver_norm.shape
    a_idx = set(np.where(active)[0].tolist())
    out = torch.zeros(B)
    with torch.no_grad():
        for b0 in range(0, B, basin_chunk):
            b1 = min(b0 + basin_chunk, B); Bc = b1 - b0
            at = attr_norm[b0:b1].to(device)
            h = torch.zeros(Bc, model.lstm_hidden_size, device=device)
            c = torch.zeros_like(h)
            acc = torch.zeros(Bc, device=device); n = 0
            dn = driver_norm[b0:b1].to(device)
            for t in range(T):
                h, c = gen.lstm_cell(torch.cat([dn[:, t, :], at], dim=-1), (h, c))
                if t in a_idx:
                    raw = gen.head(h)[:, 0]
                    acc += torch.clamp(torch.sigmoid(raw) * (hi - lo) + lo, lo, hi)
                    n += 1
            out[b0:b1] = (acc / n).cpu()
    return out.numpy()


def main():
    """Compute the cross-instance stability of catchment-mean F_add across the 25 trained instances and write fig_si_stability.npz."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from libs.utils import load_config
    config = load_config(CONFIG_PATH)
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    x_all, idx_daily, valid_ids = load_data(config)
    years = np.array([d.year for d in idx_daily])
    active = years >= ACTIVE_START_YEAR
    model = build_model(config, device)
    driver_norm, attr_norm = driver_and_attr_tensors(model, x_all, config)

    B = x_all.shape[0]
    inst_maps = np.full((len(SEEDS) * len(FOLDS), B), np.nan)  # per-checkpoint catchment-mean F_add
    inst_seed = np.zeros(len(SEEDS) * len(FOLDS), dtype=int)
    inst_fold = np.zeros(len(SEEDS) * len(FOLDS), dtype=int)
    k = 0
    for seed in SEEDS:
        for fold in FOLDS:
            load_checkpoint(model, seed, fold, device)
            avai = static_avai_cap(model, x_all, idx_daily, device).numpy()
            lat = full_lat_alpha_active_mean(model, config, driver_norm, attr_norm, active, device)
            inst_maps[k] = lat * avai
            inst_seed[k], inst_fold[k] = seed, fold
            print(f"[stability] instance seed={seed} fold={fold} done "
                  f"(mean F_add={np.nanmean(inst_maps[k]):.3f})", flush=True)
            k += 1

    rep_maps = inst_maps.reshape(len(SEEDS), len(FOLDS), B).mean(axis=1)  # per-seed replicate maps
    C25 = np.corrcoef(inst_maps)
    C5 = np.corrcoef(rep_maps)
    off25 = C25[~np.eye(len(C25), dtype=bool)]
    # within-seed (same seed, different fold) vs across-seed blocks
    same_seed = inst_seed[:, None] == inst_seed[None, :]
    offdiag = ~np.eye(len(C25), dtype=bool)
    print(f"[stability] 25x25 mean off-diag r = {off25.mean():.3f} "
          f"(min {off25.min():.3f}); within-seed r = {C25[offdiag & same_seed].mean():.3f}; "
          f"across-seed r = {C25[offdiag & ~same_seed].mean():.3f}", flush=True)

    np.savez(os.path.join(OUT_DIR, "fig_si_stability.npz"),
             inst_maps=inst_maps, rep_maps=rep_maps, corr25=C25, corr5=C5,
             inst_seed=inst_seed, inst_fold=inst_fold, seeds=np.array(SEEDS),
             gauge_id=np.array(valid_ids, dtype=object))
    print(f"Saved -> {os.path.join(OUT_DIR, 'fig_si_stability.npz')}")


if __name__ == "__main__":
    main()
