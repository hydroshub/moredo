"""
Compute per-replicate pooled NSE for the frozen-decoder recurrent model (lstm_af_frozen_v2),
using EXACTLY the same evaluation as eval_replicate_et_nse.py (pool the five held-out folds per
training seed -> one aggregate NSE per replicate). Then merge it into the Fig. 3 npz as a fifth
configuration ("recurrent (frozen)"), writing a NEW npz so the original stays untouched.

Usage:
  python scripts/analysis/eval_replicate_frozen.py [--threads 8]
"""
import argparse
import os
import sys

import numpy as np
import torch

from project_paths import REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from project_paths import BASIN_LIST
SEEDS = [100, 200, 300, 400, 500]
FOLDS = [0, 1, 2, 3, 4]
FROZEN_EXP = "lstm_af_frozen_v2"
FROZEN_CFG = "config_lateral_lstm_af.py"
FROZEN_LABEL = "recurrent (frozen)"
VARS = ["et", "twsa", "swe", "q"]

IN_NPZ = os.path.join(REPO, "results_lateral_experiments", "figures", "fig3_replicate_nse.npz")
OUT_NPZ = os.path.join(REPO, "results_lateral_experiments", "figures", "fig3_replicate_nse_frozen.npz")


def masked_nse(pred, target, eps=1e-6):
    valid = np.isfinite(target)
    p, t = pred[valid], target[valid]
    if t.size == 0:
        return np.nan
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from libs.utils import load_config
    device = torch.device("cpu")
    cfg = load_config(os.path.join(REPO, "configs", FROZEN_CFG))
    from libs.data import load_dataset_with_cache, RandomSpatialCV
    from libs.model import HybridModel

    print("Loading dataset...", flush=True)
    x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids = load_dataset_with_cache(
        data_dir=cfg.DATA_DIR, basin_list_path=BASIN_LIST, attr_paths=cfg.ATTR_PATHS,
        input_cols=cfg.INPUT_COLS, target_daily_cols=cfg.TARGET_DAILY_COLS,
        target_monthly_cols=cfg.TARGET_MONTHLY_COLS, nn_attr_cols=cfg.ATTR_NN_COLS,
        start_year=cfg.INFER_START_YEAR, end_year=cfg.INFER_END_YEAR,
        passthrough_cols=cfg.ATTR_PASSTHROUGH_COLS,
    )
    x_all = x_all.float()
    valid_ids = list(valid_ids)
    m = cfg.TARGET_COLS_MAP
    ch = {"q": cfg.TARGET_DAILY_COLS.index(m["q"]),
          "et": cfg.TARGET_DAILY_COLS.index(m["et"]),
          "swe": cfg.TARGET_DAILY_COLS.index(m["swe"])}
    y_np = y_all.numpy()
    z_np = z_all[:, :, 0].numpy()
    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long)

    # The frozen runs use cv_seed = training seed (slurm/submit_train_frozen_v2.sh), so the held-out
    # catchments of model (seed, fold) are the test fold of that seed's own partition.
    cvs = {s: RandomSpatialCV(valid_ids, num_folds=cfg.NUM_FOLDS,
                              spinup_until=int(cfg.TRAIN_START_YEAR), seed=s) for s in SEEDS}
    fold_test_idx = {s: {f: cvs[s].get_spatial_indices(f, valid_ids)[2] for f in FOLDS} for s in SEEDS}
    cv = cvs[SEEDS[0]]   # the time masks do not depend on the catchment partition
    active_d = cv.get_full_mask_dict(0, idx_daily)["test_mask"]
    active_m = cv.get_full_mask_dict(0, idx_monthly)["test_mask"]
    print(f"  {len(valid_ids)} basins; active days {int(active_d.sum())}, "
          f"months {int(active_m.sum())}", flush=True)

    model = HybridModel(cfg.PARAM_CONFIG, nn_attr_cols=cfg.ATTR_NN_COLS,
                        passthrough_cols=cfg.ATTR_PASSTHROUGH_COLS, input_cols=cfg.INPUT_COLS,
                        lstm_hidden_size=cfg.LSTM_HIDDEN_SIZE).to(device)
    model_dir = os.path.join(REPO, "results_lateral_experiments", FROZEN_EXP)

    frozen = {v: np.full(len(SEEDS), np.nan) for v in VARS}
    for si, seed in enumerate(SEEDS):
        pool = {v: ([], []) for v in VARS}
        for fold in FOLDS:
            ckpt = os.path.join(model_dir, f"best_model_seed{seed}_fold{fold}.pt")
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            model.eval()
            idx = fold_test_idx[seed][fold]
            with torch.no_grad():
                out = model(x_all[idx].to(device), date_seq=date_seq, mode="supervised",
                            show_progress=False)
            for v in ["et", "q", "swe"]:
                pred = out[v][:, :, 0].cpu().numpy()
                tgt = y_np[idx][:, :, ch[v]]
                mask = active_d[None, :] & np.isfinite(tgt)
                pool[v][0].append(pred[mask]); pool[v][1].append(tgt[mask])
            pred_tw = out["twsa"][:, :, 0].cpu().numpy()
            tgt_tw = z_np[idx]
            mask_tw = active_m[None, :] & np.isfinite(tgt_tw)
            pool["twsa"][0].append(pred_tw[mask_tw]); pool["twsa"][1].append(tgt_tw[mask_tw])
        for v in VARS:
            frozen[v][si] = masked_nse(np.concatenate(pool[v][0]), np.concatenate(pool[v][1]))
        print(f"  [frozen] seed {seed}: " +
              " ".join(f"{v}={frozen[v][si]:.3f}" for v in VARS), flush=True)
    print("==> frozen medians: " +
          " ".join(f"{v}={np.median(frozen[v]):.3f}" for v in VARS), flush=True)

    # --- merge into the existing 4-config npz, appending frozen as the 5th row ---
    d = np.load(IN_NPZ, allow_pickle=True)
    labels = list(d["labels"]) + [FROZEN_LABEL]
    experiments = list(d["experiments"]) + [FROZEN_EXP]
    merged = {v: np.vstack([d[v], frozen[v][None, :]]) for v in VARS}
    np.savez(OUT_NPZ, experiments=np.array(experiments), labels=np.array(labels),
             seeds=d["seeds"], vars=d["vars"], **merged)
    print(f"\nSaved -> {OUT_NPZ}")
    print("\n===== median per-replicate pooled NSE (all 5 configs) =====")
    print("config              " + "".join(f"{v:>8s}" for v in VARS))
    for ei, lab in enumerate(labels):
        print(f"{lab:19s} " + "".join(f"{np.median(merged[v][ei]):8.3f}" for v in VARS))


if __name__ == "__main__":
    main()
