"""
Fig. 9 input: per catchment, the long-term observational water-balance residual eps_wb
(= P - Q - ET - dS, so R_in = -eps_wb), the 2001-2020 catchment-mean F_add of the recurrent
representation, the validity mask, coordinates and the Koppen-Geiger major class. Writes
fig10_nonclosure_v2_cache.pkl, read by fig9_relation_2panel.py (Fig. 9) and rin_surrogate_cv.py
(which computes the cross-validated R_in surrogate skill).

Loads the ~3.4 GB averaged pickle, run on SLURM:
  sbatch --job-name=moredo-fig9 slurm/submit_analysis_cpu.sh nonclosure_relation_cache.py
"""
import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_utils import compute_residual_vs_fadd
from run_analysis import load_results
from basin_attributes import build_basin_tables
from project_paths import RESULTS_DIR


def _compute_panel_inputs(results, df_coord, exp_recurrent):
    """Long-term residual, catchment-mean F_add and the per-catchment metadata used by Fig. 9,
    separated so the cache can be rebuilt without touching the figure code."""
    res = compute_residual_vs_fadd(results[exp_recurrent], df_coord)
    return {
        "v": res["valid"], "eps_wb": res["residual_corrected"], "fadd": res["mean_fadd"],
        "basin_lon": df_coord["basin_lon"].values, "basin_lat": df_coord["basin_lat"].values,
        "aridity_category": df_coord["aridity_category"].values,
        "kg_major_id": df_coord["kg_major_id"].values,
    }


def main():
    """Compute the Fig. 9 panel inputs from the recurrent model output and cache them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=RESULTS_DIR)
    args = parser.parse_args()
    cache_path = os.path.join(args.results_dir, "figures", "fig10_nonclosure_v2_cache.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print("Loading averaged full-output pickle (lstm_af only)...")
    results = load_results(args.results_dir, experiments=["lstm_af"])
    valid_ids = results["lstm_af"]["valid_ids"]

    print("Building basin coordinate tables...")
    df_coord, _ = build_basin_tables(valid_ids)

    panel = _compute_panel_inputs(results, df_coord, "lstm_af")
    import pickle
    with open(cache_path, "wb") as f:
        pickle.dump(panel, f)
    print(f"Cached panel inputs -> {cache_path}")


if __name__ == "__main__":
    main()
