"""
Effective-memory experiment for the recurrent lateral subsidy F_add (LSTM(A,F)).

Rationale (from the model code): F_add = lat_alpha * avai_cap, where avai_cap is a STATIC
per-basin quantity and lat_alpha is produced solely by the per-basin LSTM encoder run over the
normalized drivers (P, T, RAD, LAI) plus static attributes -- it does NOT depend on the physical
water-balance buckets (see libs/model.py: avai_bucket / generate_params). Therefore truncating the
LSTM's history is equivalent to resetting its hidden/cell state at t-k and replaying the preceding
k days of driver+attribute inputs; the physical stores are irrelevant to F_add. We exploit this to
replay truncated histories through the extracted LSTM submodule (exactly the model's weights and
input normalization), which is far cheaper than re-running the full hydrology, and we VALIDATE that
the full-history replay reproduces the model's own logged lat_alpha before trusting the truncation.

For each catchment s and history length k in K = [0,7,14,28,56,84,120,180] days:
  F_add,k^(s,t) : reset LSTM state at t-k, replay the k preceding days (+ day t), read lat_alpha,
                  multiply by the run's static avai_cap.
  F_add,full    : full-history replay (= the model's normal output).
  E_s(k) = RMSE_t(F_add,k, F_add,full) / SD_t(F_add,full)                                   (Eq. 10)
  L_s    = shortest tested k with E_s(k) < eps (=0.10) AND E_s(k') < eps for all k' > k.

All quantities are ENSEMBLE-AVERAGED over the 25 seed x fold checkpoints (F_add,k and F_add,full
are averaged across runs at each (basin, day) before E_s(k) is formed), matching how the averaged
pickle is used elsewhere.

Interpretation: deseasonalize F_add and (P - PET) (remove the per-basin day-of-year climatology),
accumulate the (P-PET) anomaly over each catchment's L_s-day trailing window, and correlate it with
the concurrent F_add anomaly. r < 0 => subsidy rises after anomalously dry spells (buffering);
r > 0 => association with wetter antecedent conditions (recharge-like).

Outputs (results_lateral_experiments/figures/):
  memory_truncation.npz  : E_s(k) [B,K], L_s [B], antecedent r [B], k values, gauge ids
  memory_truncation.csv  : per-basin table
  (plotted by fig8_memory_inset.py for Fig. 8 and memory_truncation_plot_v2.build_map_only for Fig. S12)

Usage:
  python scripts/analysis/memory_truncation.py --validate      # quick correctness gate, 1 checkpoint
  python scripts/analysis/memory_truncation.py                 # full 25-run ensemble (heavy, CPU)
"""
import argparse
import os
import sys

import numpy as np
import torch

from project_paths import REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

CONFIG_PATH = os.path.join(REPO, "configs", "config_lateral_lstm_af.py")
MODEL_DIR = os.path.join(REPO, "results_lateral_experiments", "lstm_af")
from project_paths import BASIN_LIST
AVG_PICKLE = os.path.join(MODEL_DIR, "averaged_full_output.pickle")
OUT_DIR = os.path.join(REPO, "results_lateral_experiments", "figures")

SEEDS = [100, 200, 300, 400, 500]
FOLDS = [0, 1, 2, 3, 4]
K_VALUES = [0, 7, 14, 28, 56, 84, 120, 180, 240, 300, 365]
EPS = 0.10
ACTIVE_START_YEAR = 2001


def load_data(config):
    """Load and cache the dataset and forcing arrays for the truncation experiment from the config."""
    from libs.data import load_dataset_with_cache
    x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids = load_dataset_with_cache(
        data_dir=config.DATA_DIR, basin_list_path=BASIN_LIST, attr_paths=config.ATTR_PATHS,
        input_cols=config.INPUT_COLS, target_daily_cols=config.TARGET_DAILY_COLS,
        target_monthly_cols=config.TARGET_MONTHLY_COLS, nn_attr_cols=config.ATTR_NN_COLS,
        start_year=config.INFER_START_YEAR, end_year=config.INFER_END_YEAR,
        passthrough_cols=config.ATTR_PASSTHROUGH_COLS,
    )
    return x_all.float(), list(idx_daily), list(valid_ids)


def build_model(config, device):
    """Instantiate the HybridModel (MOREDO decoder + subsidy encoder) on the given device from the config."""
    from libs.model import HybridModel
    model = HybridModel(config.PARAM_CONFIG, nn_attr_cols=config.ATTR_NN_COLS,
                        passthrough_cols=config.ATTR_PASSTHROUGH_COLS, input_cols=config.INPUT_COLS,
                        lstm_hidden_size=config.LSTM_HIDDEN_SIZE).to(device)
    return model


def load_checkpoint(model, seed, fold, device):
    """Load the trained weights of one seed/fold checkpoint into model."""
    sd = torch.load(os.path.join(MODEL_DIR, f"best_model_seed{seed}_fold{fold}.pt"),
                    weights_only=True, map_location=device)
    model.load_state_dict(sd)
    model.eval()


def driver_and_attr_tensors(model, x_all, config):
    """Reproduce exactly what generate_params feeds the lat_alpha LSTM:
      driver_input = normalize(P, T, RAD_SW+RAD_LW, LAI) with DRIVER_NORM
      attr_input   = the already-normalized static-attr columns (mean over time)
    x_all columns are ordered as model.all_cols = INPUT_COLS + [c_normed...] + [c_raw...]."""
    col_idx = {c: i for i, c in enumerate(model.all_cols)}
    dm, dn = config.DRIVER_COLS_MAP, config.DRIVER_NORM

    def raw(name):
        """Return the raw (unnormalized) driver series for the named variable as [B, T]."""
        return x_all[:, :, col_idx[dm[name]]]  # [B, T]

    P = raw("P")
    T = raw("T")
    RAD = raw("RAD_SW") + raw("RAD_LW")
    LAI = raw("LAI")
    dn_ = {k: (float(v[0]), float(v[1])) for k, v in dn.items()}
    driver_norm = torch.stack([
        (P - dn_["P"][0]) / dn_["P"][1],
        (T - dn_["T"][0]) / dn_["T"][1],
        (RAD - dn_["RAD"][0]) / dn_["RAD"][1],
        (LAI - dn_["LAI"][0]) / dn_["LAI"][1],
    ], dim=-1)  # [B, T, 4]

    lat_attrs = config.PARAM_CONFIG["lat_alpha"][2]  # attribute list for lat_alpha
    attr_norm = torch.stack([x_all[:, :, col_idx[f"{c}_normed"]].mean(dim=1) for c in lat_attrs],
                            dim=-1)  # [B, A] (constant over time; mean matches model)
    return driver_norm, attr_norm


def replay_lat_alpha(model, config, driver_norm, attr_norm, k, S, E, device, basin_chunk=300):
    """Truncated-history lat_alpha for target days [S, E): reset LSTM state at t-k, replay days
    t-k..t. Vectorized over (basin x target-day); target days are contiguous so each replay step is
    a contiguous time slice. Returns [B, E-S]."""
    gen = model.param_dyn_lstm["lat_alpha"]
    lo, hi = config.PARAM_CONFIG["lat_alpha"][0]
    B = driver_norm.shape[0]
    n_tg = E - S
    out = torch.empty(B, n_tg, dtype=torch.float32)

    with torch.no_grad():
        for b0 in range(0, B, basin_chunk):
            b1 = min(b0 + basin_chunk, B)
            Bc = b1 - b0
            dn_c = driver_norm[b0:b1]                    # [Bc, T, 4]
            at_c = attr_norm[b0:b1].to(device)           # [Bc, A]
            at_step = at_c.unsqueeze(1).expand(Bc, n_tg, at_c.shape[-1]).reshape(Bc * n_tg, -1)
            h = torch.zeros(Bc * n_tg, model.lstm_hidden_size, device=device)
            c = torch.zeros_like(h)
            for j in range(k + 1):                       # steps t-k .. t  (k+1 steps)
                d_step = dn_c[:, S - k + j: E - k + j, :].reshape(Bc * n_tg, -1).to(device)
                inp = torch.cat([d_step, at_step], dim=-1)
                h, c = gen.lstm_cell(inp, (h, c))
            raw = gen.head(h)[:, 0]                       # dropout is identity in eval()
            lat = torch.clamp(torch.sigmoid(raw) * (hi - lo) + lo, min=lo, max=hi)
            out[b0:b1] = lat.reshape(Bc, n_tg).cpu()
    return out


def static_avai_cap(model, x_all, idx_daily, device):
    """avai_cap is static; read it from a short forward pass."""
    from datetime import date
    n = 8
    date_seq = torch.tensor([d.toordinal() for d in idx_daily[:n]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x_all[:, :n, :].to(device), date_seq=date_seq, mode="full",
                    compute_twsa=False, show_progress=False)
    return out["params"]["avai_cap"][:, 0, 0].cpu()  # [B]


def validate(model, config, x_all, idx_daily, driver_norm, attr_norm, device):
    """Full-history replay must reproduce the model's own logged lat_alpha."""
    from datetime import date
    load_checkpoint(model, SEEDS[0], FOLDS[0], device)
    n = 2000  # first ~5.5 years is enough to validate
    date_seq = torch.tensor([d.toordinal() for d in idx_daily[:n]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x_all[:, :n, :].to(device), date_seq=date_seq, mode="full",
                    compute_twsa=False, show_progress=False)
    lat_model = out["params"]["lat_alpha"][:, :, 0].cpu()  # [B, n]

    # Replay lat_alpha for every day d in [1, n): full history = reset at day 0, replay 0..d.
    gen = model.param_dyn_lstm["lat_alpha"]
    lo, hi = config.PARAM_CONFIG["lat_alpha"][0]
    B = x_all.shape[0]
    dn_c = driver_norm[:, :n, :].to(device)
    at = attr_norm.to(device)
    h = torch.zeros(B, model.lstm_hidden_size, device=device)
    c = torch.zeros_like(h)
    lat_replay = torch.empty(B, n)
    with torch.no_grad():
        for t in range(n):
            inp = torch.cat([dn_c[:, t, :], at], dim=-1)
            h, c = gen.lstm_cell(inp, (h, c))
            raw = gen.head(h)[:, 0]
            lat_replay[:, t] = torch.clamp(torch.sigmoid(raw) * (hi - lo) + lo, min=lo, max=hi).cpu()

    diff = (lat_replay - lat_model).abs()
    print(f"[validate] max |replay - model| lat_alpha = {diff.max().item():.3e}  "
          f"(mean {diff.mean().item():.3e}); lat_alpha range "
          f"[{lat_model.min().item():.4f}, {lat_model.max().item():.4f}]")
    ok = diff.max().item() < 1e-4
    print("[validate] PASS" if ok else "[validate] FAIL — extraction does not match the model")
    return ok


def run_ensemble(model, config, x_all, idx_daily, driver_norm, attr_norm, device):
    """Replay the trained encoder with reset-then-k-day histories across all seeds/folds and return the per-catchment normalized truncation-error curves."""
    years = np.array([d.year for d in idx_daily])
    S = int(np.argmax(years >= ACTIVE_START_YEAR))  # first active-period index
    E = len(idx_daily)
    n_tg = E - S
    B = x_all.shape[0]

    # Ensemble sums of F_add over the 25 runs, for full history and each truncation length.
    sum_full = torch.zeros(B, n_tg)
    sum_k = {k: torch.zeros(B, n_tg) for k in K_VALUES}
    n_runs = 0
    for seed in SEEDS:
        for fold in FOLDS:
            load_checkpoint(model, seed, fold, device)
            avai = static_avai_cap(model, x_all, idx_daily, device)          # [B]
            avai_col = avai.unsqueeze(1)
            # full history = replay with k reaching back to day 0 for the earliest target.
            lat_full = replay_lat_alpha(model, config, driver_norm, attr_norm, k=S, S=S, E=E, device=device)
            sum_full += lat_full * avai_col
            for k in K_VALUES:
                lat_k = replay_lat_alpha(model, config, driver_norm, attr_norm, k=k, S=S, E=E, device=device)
                sum_k[k] += lat_k * avai_col
            n_runs += 1
            print(f"[ensemble] done seed={seed} fold={fold}  ({n_runs}/25)", flush=True)

    fadd_full = (sum_full / n_runs).numpy()                    # [B, n_tg]
    fadd_k = {k: (sum_k[k] / n_runs).numpy() for k in K_VALUES}  # ensemble-mean F_add,k
    return fadd_full, fadd_k, S, E


def compute_memory(fadd_full, fadd_k):
    """E_s(k) and effective memory length L_s per basin."""
    sd = np.std(fadd_full, axis=1)                             # [B]
    B = fadd_full.shape[0]
    K = np.array(K_VALUES)
    E_sk = np.full((B, len(K)), np.nan)
    for j, k in enumerate(K):
        rmse = np.sqrt(np.mean((fadd_k[k] - fadd_full) ** 2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            E_sk[:, j] = np.where(sd > 0, rmse / sd, np.nan)

    L = np.full(B, np.nan)
    for b in range(B):
        e = E_sk[b]
        below = e < EPS
        # shortest k with E<eps that stays below for all longer k
        for j in range(len(K)):
            if below[j] and np.all(below[j:]):
                L[b] = K[j]
                break
    return E_sk, L, sd


def antecedent_correlation(fadd_full, L, idx_daily, S, E, valid_ids):
    """Deseasonalize F_add and (P-PET) by per-basin day-of-year climatology, accumulate the
    (P-PET) anomaly over each basin's L_s-day trailing window, correlate with the F_add anomaly."""
    import pandas as pd
    from residual_utils import _reconstruct_precip, _daily_series

    with open(AVG_PICKLE, "rb") as f:
        import pickle
        avg = pickle.load(f)
    # Align averaged-pickle basins to our dataset order.
    avg_ids = list(avg["valid_ids"])
    pos = {g: i for i, g in enumerate(avg_ids)}
    order = [pos[g] for g in valid_ids]
    P_full = _reconstruct_precip(avg)[order]                   # [B, T_full] rain+snow
    PET_full = _daily_series(avg, "pet", group="diagnos")[order]
    pmet_full = (P_full - PET_full)[:, S:E]                    # [B, n_tg]

    dates = [d for d in idx_daily[S:E]]
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])  # clip leap day 366->365

    def deseason(arr):
        """Remove the mean day-of-year seasonal cycle from a [B, T] series."""
        anom = np.empty_like(arr)
        for dd in range(1, 366):
            m = doy == dd
            if m.any():
                anom[:, m] = arr[:, m] - arr[:, m].mean(axis=1, keepdims=True)
        return anom

    fadd_anom = deseason(fadd_full)
    pmet_anom = deseason(pmet_full)

    B, n_tg = fadd_full.shape
    r = np.full(B, np.nan)
    csum = np.concatenate([np.zeros((B, 1)), np.cumsum(pmet_anom, axis=1)], axis=1)  # [B, n_tg+1]
    for b in range(B):
        Lb = L[b]
        if not np.isfinite(Lb) or Lb < 1:
            continue
        Lb = int(Lb)
        # trailing Lb-day sum ending at each day t>=Lb-1: csum has n_tg+1 entries, so
        # csum[Lb:] - csum[:-Lb] has length n_tg+1-Lb and aligns to fadd_anom[Lb-1:].
        ant = csum[b, Lb:] - csum[b, :-Lb]
        f = fadd_anom[b, Lb - 1:]
        m = np.isfinite(ant) & np.isfinite(f)
        if m.sum() > 365 and np.std(ant[m]) > 0 and np.std(f[m]) > 0:
            r[b] = np.corrcoef(ant[m], f[m])[0, 1]
    return r


def main():
    """Run the effective-history-length experiment and write memory_truncation.npz (input to Figure 8 and S12)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--from_raw", action="store_true",
                        help="skip the GPU ensemble; redo E_s(k)/L_s/correlation from the saved raw arrays")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=str, default=None, help="override config path (e.g. hidden-size variant)")
    parser.add_argument("--model_dir", type=str, default=None, help="override checkpoint dir")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds (e.g. 100 for a 1-seed sweep)")
    parser.add_argument("--tag", type=str, default="", help="suffix for output files, e.g. _h16")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    # Optional overrides for a hidden-size / architecture sensitivity run (leave defaults otherwise).
    global CONFIG_PATH, MODEL_DIR, SEEDS
    if args.config:
        CONFIG_PATH = args.config
    if args.model_dir:
        MODEL_DIR = args.model_dir
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",")]

    raw_path = os.path.join(OUT_DIR, f"memory_truncation{args.tag}_raw.npz")

    from libs.utils import load_config
    config = load_config(CONFIG_PATH)

    if args.from_raw:
        # Light post-processing only (no GPU): reload the expensive ensemble arrays.
        print(f"Loading raw ensemble arrays from {raw_path}", flush=True)
        raw = np.load(raw_path, allow_pickle=True)
        fadd_full = raw["fadd_full"]
        fadd_k = {int(k): raw[f"fadd_k_{int(k)}"] for k in raw["k_values"]}
        S, E = int(raw["S"]), int(raw["E"])
        valid_ids = list(raw["gauge_id"])
        _, idx_daily, _ = load_data(config)
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}", flush=True)
        print("Loading dataset...", flush=True)
        x_all, idx_daily, valid_ids = load_data(config)
        print(f"  x_all {tuple(x_all.shape)}, {len(valid_ids)} basins, {len(idx_daily)} days", flush=True)
        model = build_model(config, device)
        driver_norm, attr_norm = driver_and_attr_tensors(model, x_all, config)

        if args.validate:
            validate(model, config, x_all, idx_daily, driver_norm, attr_norm, device)
            return
        if not validate(model, config, x_all, idx_daily, driver_norm, attr_norm, device):
            raise SystemExit("Replay validation failed; aborting.")

        fadd_full, fadd_k, S, E = run_ensemble(model, config, x_all, idx_daily, driver_norm, attr_norm, device)
        # Checkpoint the EXPENSIVE arrays immediately, so any post-processing error is cheap to fix.
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez(raw_path, fadd_full=fadd_full, S=S, E=E, k_values=np.array(K_VALUES),
                 gauge_id=np.array(valid_ids, dtype=object),
                 **{f"fadd_k_{k}": fadd_k[k] for k in K_VALUES})
        print(f"Checkpointed raw ensemble -> {raw_path}", flush=True)

    E_sk, L, sd = compute_memory(fadd_full, fadd_k)
    r = antecedent_correlation(fadd_full, L, idx_daily, S, E, valid_ids)

    n_def = int(np.isfinite(L).sum())
    print(f"[result] L_s defined for {n_def}/{len(L)} basins; "
          f"median L_s = {np.nanmedian(L):.0f} d; median r = {np.nanmedian(r):.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"memory_truncation{args.tag}.npz"),
             k_values=np.array(K_VALUES), E_sk=E_sk, L=L, r=r, sd=sd,
             gauge_id=np.array(valid_ids, dtype=object), eps=EPS)
    import pandas as pd
    df = pd.DataFrame({"gauge_id": valid_ids, "L_s_days": L, "antecedent_r": r, "sd_fadd_full": sd})
    for j, k in enumerate(K_VALUES):
        df[f"E_k{k}"] = E_sk[:, j]
    df.to_csv(os.path.join(OUT_DIR, f"memory_truncation{args.tag}.csv"), index=False)
    print(f"Saved -> {os.path.join(OUT_DIR, f'memory_truncation{args.tag}.npz')} and .csv")


if __name__ == "__main__":
    main()
