"""
Per-replicate pooled ET NSE for the four subsidy representations (Fig. 3 numbers).

Reproduces exactly the training-time evaluation (libs.utils.evaluate_model / masked_nse_metric):
for each spatial-CV replicate (= training seed) we pool the held-out predictions from its five
folds -- every catchment appears once -- and compute a single aggregate ET NSE over the active
period (years >= TRAIN_START_YEAR). The reported number is the median of those five per-replicate
values. Each training seed defines its own catchment partition (RandomSpatialCV with
cv_seed = seed, identical across experiments), so the held-out sets match those used in training.

Configs -> narrative label:
  baseline  = no-subsidy           mlp_a = static (temporally constant)
  mlp_af    = instantaneous        lstm_af = recurrent

Usage:
  python scripts/analysis/eval_replicate_et_nse.py [--threads 8]
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
EXPERIMENTS = {
    "baseline": "config_lateral_baseline.py",
    "mlp_a":    "config_lateral_mlp_a.py",
    "mlp_af":   "config_lateral_mlp_af.py",
    "lstm_af":  "config_lateral_lstm_af.py",
}
LABEL = {"baseline": "no-subsidy", "mlp_a": "static", "mlp_af": "instantaneous", "lstm_af": "recurrent"}
VARS = ["et", "twsa", "swe", "q"]
OUT_NPZ = os.path.join(REPO, "results_lateral_experiments", "figures", "fig3_replicate_nse.npz")


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
    base_cfg = load_config(os.path.join(REPO, "configs", "config_lateral_lstm_af.py"))
    # libs.data/model do `import config` at module load, so import them only AFTER load_config
    # has registered a config module in sys.modules.
    from libs.data import load_dataset_with_cache, RandomSpatialCV
    from libs.model import HybridModel

    print("Loading dataset...", flush=True)
    x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids = load_dataset_with_cache(
        data_dir=base_cfg.DATA_DIR, basin_list_path=BASIN_LIST, attr_paths=base_cfg.ATTR_PATHS,
        input_cols=base_cfg.INPUT_COLS, target_daily_cols=base_cfg.TARGET_DAILY_COLS,
        target_monthly_cols=base_cfg.TARGET_MONTHLY_COLS, nn_attr_cols=base_cfg.ATTR_NN_COLS,
        start_year=base_cfg.INFER_START_YEAR, end_year=base_cfg.INFER_END_YEAR,
        passthrough_cols=base_cfg.ATTR_PASSTHROUGH_COLS,
    )
    x_all = x_all.float()
    valid_ids = list(valid_ids)
    m = base_cfg.TARGET_COLS_MAP
    ch = {"q": base_cfg.TARGET_DAILY_COLS.index(m["q"]),
          "et": base_cfg.TARGET_DAILY_COLS.index(m["et"]),
          "swe": base_cfg.TARGET_DAILY_COLS.index(m["swe"])}
    y_np = y_all.numpy()          # daily targets [B, T, 3]
    z_np = z_all[:, :, 0].numpy() # monthly TWSA target [B, M]
    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long)

    # The training runs use cv_seed = training seed (slurm/submit_train_global.sh), so the held-out
    # catchments of model (seed, fold) are the test fold of that seed's own partition.
    cvs = {s: RandomSpatialCV(valid_ids, num_folds=base_cfg.NUM_FOLDS,
                              spinup_until=int(base_cfg.TRAIN_START_YEAR), seed=s) for s in SEEDS}
    fold_test_idx = {s: {f: cvs[s].get_spatial_indices(f, valid_ids)[2] for f in FOLDS} for s in SEEDS}
    cv = cvs[SEEDS[0]]   # the time masks do not depend on the catchment partition
    active_d = cv.get_full_mask_dict(0, idx_daily)["test_mask"]     # daily time mask
    active_m = cv.get_full_mask_dict(0, idx_monthly)["test_mask"]   # monthly time mask (TWSA)
    print(f"  {len(valid_ids)} basins; fold test sizes (seed {SEEDS[0]}): "
          f"{[len(fold_test_idx[SEEDS[0]][f]) for f in FOLDS]}; active days {int(active_d.sum())}, "
          f"months {int(active_m.sum())}", flush=True)

    # results[var] -> [n_exp, n_seed]
    results = {v: np.full((len(EXPERIMENTS), len(SEEDS)), np.nan) for v in VARS}
    exp_order = list(EXPERIMENTS.keys())
    for ei, exp in enumerate(exp_order):
        cfg = load_config(os.path.join(REPO, "configs", EXPERIMENTS[exp]))
        model_dir = os.path.join(REPO, "results_lateral_experiments", exp)
        model = HybridModel(cfg.PARAM_CONFIG, nn_attr_cols=cfg.ATTR_NN_COLS,
                            passthrough_cols=cfg.ATTR_PASSTHROUGH_COLS, input_cols=cfg.INPUT_COLS,
                            lstm_hidden_size=cfg.LSTM_HIDDEN_SIZE).to(device)
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
                pred_tw = out["twsa"][:, :, 0].cpu().numpy()   # [n_test, M]
                tgt_tw = z_np[idx]
                mask_tw = active_m[None, :] & np.isfinite(tgt_tw)
                pool["twsa"][0].append(pred_tw[mask_tw]); pool["twsa"][1].append(tgt_tw[mask_tw])
            for v in VARS:
                results[v][ei, si] = masked_nse(np.concatenate(pool[v][0]), np.concatenate(pool[v][1]))
            print(f"  [{exp:8s}] seed {seed}: " +
                  " ".join(f"{v}={results[v][ei, si]:.3f}" for v in VARS), flush=True)
        print(f"==> {exp:8s} ({LABEL[exp]:13s}) medians: " +
              " ".join(f"{v}={np.median(results[v][ei]):.3f}" for v in VARS), flush=True)

    np.savez(OUT_NPZ, experiments=np.array(exp_order), labels=np.array([LABEL[e] for e in exp_order]),
             seeds=np.array(SEEDS), vars=np.array(VARS),
             **{v: results[v] for v in VARS})
    print(f"\nSaved -> {OUT_NPZ}")
    print("\n===== median per-replicate pooled NSE =====")
    header = "config          " + "".join(f"{v:>8s}" for v in VARS)
    print(header)
    for ei, exp in enumerate(exp_order):
        print(f"{LABEL[exp]:15s} " + "".join(f"{np.median(results[v][ei]):8.3f}" for v in VARS))


if __name__ == "__main__":
    main()
