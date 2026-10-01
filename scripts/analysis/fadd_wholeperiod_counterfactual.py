"""
F_add=0 counterfactual (Methods Eq. 12, Fig. 10): the ET contribution of the recurrent subsidy.
F_add is suppressed for the ENTIRE simulation including the spin-up (gate=1 every day, no reset),
so the storage trajectory diverges freely from the reference; delta ET = ET_with - ET_no is then
summed over all active-period days (2001-2020) and expressed as mm/yr, ensemble-averaged over the
25 recurrent checkpoints. The trained encoder and all parameters are held fixed.

Reuses the tested data/model/checkpoint machinery from memory_truncation and the model.forward
F_add gate, so the counterfactual is a real state-propagating rerun, not a descriptive
ET - F_add subtraction.

Usage (GPU via slurm):
  sbatch --job-name=moredo-cf-whole slurm/submit_analysis_gpu.sh fadd_wholeperiod_counterfactual.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch

from project_paths import REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
from memory_truncation import (load_data, build_model, load_checkpoint,
                               SEEDS, FOLDS, CONFIG_PATH, ACTIVE_START_YEAR)

RD = os.path.join(REPO, "results_lateral_experiments")
OUT_NPZ = os.path.join(RD, "figures", "fadd_wholeperiod_counterfactual.npz")


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------
def _date_seq(idx_daily, device):
    """Return the daily date sequence as an ordinal tensor on the given device."""
    return torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)


def reference_run(model, x_all, idx_daily, device):
    """Reference simulation: ET plus the reference S_AVAI / rel_moist trajectories."""
    with torch.no_grad():
        out = model(x_all.to(device), date_seq=_date_seq(idx_daily, device), mode="full",
                    compute_twsa=False, log_to_cpu=True, show_progress=False)
    et = out["fluxes"]["et"][:, :, 0].contiguous()      # [B, T] (cpu)
    s_avai = out["states"]["S_AVAI"].contiguous()       # [B, T, 1]
    rel = out["diagnos"]["rel_moist"].contiguous()      # [B, T, 1]
    pet = out["diagnos"]["pet"][:, :, 0].numpy()        # [B, T]
    # Stores are logged at end-of-day; the reset hook needs the START-of-day reference value,
    # i.e. the previous day's stored value (day 0 -> model init, which is zero).
    ref_savai_start = torch.zeros_like(s_avai); ref_savai_start[:, 1:, :] = s_avai[:, :-1, :]
    ref_rel_start = torch.zeros_like(rel); ref_rel_start[:, 1:, :] = rel[:, :-1, :]
    return et, ref_savai_start, ref_rel_start, pet


def counterfactual_et(model, x_all, idx_daily, device, gate, reset, ref_savai_start, ref_rel_start):
    """Run MOREDO with F_add scaled by the given suppression gate and return the simulated ET trajectory."""
    fadd_gate = torch.as_tensor(gate, dtype=torch.float32, device=device)
    reset_mask = torch.as_tensor(reset, dtype=torch.float32, device=device)
    with torch.no_grad():
        # supervised mode's return dict requires TWSA, so compute_twsa must stay on (the monthly
        # aggregation is cheap; we only read 'et' back).
        out = model(x_all.to(device), date_seq=_date_seq(idx_daily, device), mode="supervised",
                    compute_twsa=True, show_progress=False,
                    fadd_gate=fadd_gate, reset_mask=reset_mask,
                    ref_S_AVAI=ref_savai_start.to(device), ref_relmoist=ref_rel_start.to(device))
    return out["et"][:, :, 0].cpu()  # [B, T]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from libs.utils import load_config
    config = load_config(CONFIG_PATH)
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    x_all, idx_daily, valid_ids = load_data(config)
    B, T = x_all.shape[0], x_all.shape[1]
    print(f"  x_all {tuple(x_all.shape)}, {len(valid_ids)} basins, {T} days", flush=True)
    model = build_model(config, device)

    active = np.array([d.year >= ACTIVE_START_YEAR for d in idx_daily])
    n_years = len({d.year for d in idx_daily if d.year >= ACTIVE_START_YEAR})
    active_t = torch.tensor(active, dtype=torch.float32)          # [T]
    gate = np.ones((B, T), dtype=np.float32)                      # suppress F_add every day
    reset = np.zeros((B, T), dtype=np.float32)                    # no reset -> whole-period free run
    print(f"  active days: {int(active.sum())} over {n_years} years", flush=True)

    # sanity gate: gate=0 (no suppression), no reset must reproduce the reference ET exactly
    load_checkpoint(model, SEEDS[0], FOLDS[0], device)
    et_ref0, sa0, rl0, _ = reference_run(model, x_all, idx_daily, device)
    et_chk = counterfactual_et(model, x_all, idx_daily, device,
                               np.zeros((B, T), np.float32), reset, sa0, rl0)
    d0 = (et_chk - et_ref0).abs().max().item()
    print(f"[sanity] max |ET(gate=0) - ET_ref| = {d0:.3e}  ({'PASS' if d0 < 1e-4 else 'FAIL'})", flush=True)
    if d0 >= 1e-4:
        raise SystemExit("gate=0 did not reproduce the reference; aborting.")

    acc_d = torch.zeros(B)
    acc_ref = torch.zeros(B)
    n = 0
    for seed in SEEDS:
        for fold in FOLDS:
            load_checkpoint(model, seed, fold, device)
            et_ref, sa, rl, _ = reference_run(model, x_all, idx_daily, device)
            et_cf = counterfactual_et(model, x_all, idx_daily, device, gate, reset, sa, rl)
            acc_d += ((et_ref - et_cf) * active_t).sum(dim=1)
            acc_ref += (et_ref * active_t).sum(dim=1)
            n += 1
            print(f"  done seed{seed} fold{fold} ({n}/25)", flush=True)

    dET_annual = (acc_d / n).numpy() / n_years            # mm/yr
    et_ref_annual = (acc_ref / n).numpy() / n_years       # mm/yr
    with np.errstate(invalid="ignore", divide="ignore"):
        B_whole = np.where(et_ref_annual > 0, dET_annual / et_ref_annual, np.nan)

    np.savez(OUT_NPZ, gauge_id=np.array(valid_ids, dtype=object),
             dET_annual=dET_annual, et_ref_annual=et_ref_annual, B_whole=B_whole,
             n_years=n_years)
    x = dET_annual[np.isfinite(dET_annual)]
    print(f"\ndelta ET (whole-period, F_add=0) [mm/yr]: median={np.median(x):.1f}  "
          f"IQR=[{np.percentile(x,25):.1f}, {np.percentile(x,75):.1f}]  "
          f"p90={np.percentile(x,90):.1f}  max={x.max():.1f}", flush=True)
    print(f"relative ET contribution B_whole median = {np.nanmedian(B_whole)*100:.1f}%")
    print(f"Saved -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
