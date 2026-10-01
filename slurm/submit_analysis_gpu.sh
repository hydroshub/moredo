#!/bin/bash
# Generic GPU Slurm wrapper for the heavy analysis/inference scripts under
# scripts/analysis/ (memory_truncation.py, fig_si_stability.py, ...).
#
# Usage:
#   sbatch --job-name=moredo-<label> slurm/submit_analysis_gpu.sh <script.py> [args...]
#   (submit from the repository root; create results_lateral_experiments/logs/ once beforehand)
#   e.g.
#   sbatch --job-name=moredo-memtrunc  slurm/submit_analysis_gpu.sh memory_truncation.py --device cuda
#   sbatch --job-name=moredo-stability slurm/submit_analysis_gpu.sh fig_si_stability.py  --device cuda

#SBATCH --job-name=moredo-analysis
#SBATCH --output=results_lateral_experiments/logs/%x-%j.out
#SBATCH --error=results_lateral_experiments/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=128G

set -euo pipefail

# --- site settings. Defaults: sbatch is run from the repository root with the Python environment
#     activated; slurm/site_local.sh (optional, not part of the release) can override them ---
CODE_DIR="${SLURM_SUBMIT_DIR}"                                  # repository root
PYTHON_BIN="python"
WORK_DIR="${CODE_DIR}"                                          # the data loader keeps cache/ here
BASIN_LIST_DEFAULT="${CODE_DIR}/configs/basin_list_global_grdc.txt"
if [ -f "${SLURM_SUBMIT_DIR}/slurm/site_local.sh" ]; then source "${SLURM_SUBMIT_DIR}/slurm/site_local.sh"; fi
LOG_DIR="${CODE_DIR}/results_lateral_experiments/logs"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

SCRIPT="${1:?usage: sbatch slurm/submit_analysis_gpu.sh <script.py> [args...]}"
shift

echo "========================================================================"
echo "moredo analysis : ${SCRIPT}"
echo "job_id          : ${SLURM_JOB_ID}"
echo "node            : ${SLURMD_NODENAME:-unknown}"
echo "gpu             : ${CUDA_VISIBLE_DEVICES:-none}"
echo "args            : $*"
echo "========================================================================"

PYTHONPATH="${CODE_DIR}" "${PYTHON_BIN}" "${CODE_DIR}/scripts/analysis/${SCRIPT}" "$@"
