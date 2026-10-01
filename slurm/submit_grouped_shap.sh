#!/bin/bash
# CPU-only analysis job: grouped-SHAP importance for F_add with 100x5-fold repeated CV.
# Runs on the work partition (no GPU); threads capped to avoid oversubscription on tiny data.
# Usage (from the repository root; create results_lateral_experiments/logs/ once beforehand):
#   sbatch slurm/submit_grouped_shap.sh
#SBATCH --job-name=grouped-shap-cv
#SBATCH --output=results_lateral_experiments/logs/%x-%j.out
#SBATCH --error=results_lateral_experiments/logs/%x-%j.err
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RF_NJOBS=8          # joblib workers for the RF (matches cpus-per-task)

# --- site settings. Defaults: sbatch is run from the repository root with the Python environment
#     activated; slurm/site_local.sh (optional, not part of the release) can override them ---
CODE_DIR="${SLURM_SUBMIT_DIR}"                                  # repository root
PYTHON_BIN="python"
WORK_DIR="${CODE_DIR}"                                          # the data loader keeps cache/ here
BASIN_LIST_DEFAULT="${CODE_DIR}/configs/basin_list_global_grdc.txt"
if [ -f "${SLURM_SUBMIT_DIR}/slurm/site_local.sh" ]; then source "${SLURM_SUBMIT_DIR}/slurm/site_local.sh"; fi
cd "${CODE_DIR}"
PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_DIR}" "${PYTHON_BIN}" scripts/analysis/fig_si_grouped_shap.py
