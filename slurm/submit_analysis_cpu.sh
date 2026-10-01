#!/bin/bash
# Generic CPU-only Slurm wrapper for the post-hoc analysis scripts under scripts/analysis/
# (random-forest surrogates and the cache producers that load the averaged pickles).
# Runs on the work partition (no GPU); BLAS threads pinned to 1 and the RF worker count
# (RF_NJOBS) set to cpus-per-task so joblib does not oversubscribe beyond the allocation.
#
# Usage:
#   sbatch --job-name=moredo-<label> slurm/submit_analysis_cpu.sh <script.py> [args...]
#   (submit from the repository root; create results_lateral_experiments/logs/ once beforehand)
#   e.g.
#   sbatch --job-name=moredo-fig6     slurm/submit_analysis_cpu.sh shap_fig6_cache.py
#   sbatch --job-name=moredo-grouped  slurm/submit_analysis_cpu.sh fig_si_grouped_shap.py

#SBATCH --job-name=moredo-analysis-cpu
#SBATCH --output=results_lateral_experiments/logs/%x-%j.out
#SBATCH --error=results_lateral_experiments/logs/%x-%j.err
#SBATCH --partition=work
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RF_NJOBS="${SLURM_CPUS_PER_TASK:-16}"   # joblib workers for the RF (matches cpus-per-task)

# --- site settings. Defaults: sbatch is run from the repository root with the Python environment
#     activated; slurm/site_local.sh (optional, not part of the release) can override them ---
CODE_DIR="${SLURM_SUBMIT_DIR}"                                  # repository root
PYTHON_BIN="python"
WORK_DIR="${CODE_DIR}"                                          # the data loader keeps cache/ here
BASIN_LIST_DEFAULT="${CODE_DIR}/configs/basin_list_global_grdc.txt"
if [ -f "${SLURM_SUBMIT_DIR}/slurm/site_local.sh" ]; then source "${SLURM_SUBMIT_DIR}/slurm/site_local.sh"; fi
LOG_DIR="${CODE_DIR}/results_lateral_experiments/logs"
mkdir -p "${LOG_DIR}"

SCRIPT="${1:?usage: sbatch slurm/submit_analysis_cpu.sh <script.py> [args...]}"
shift

echo "========================================================================"
echo "moredo cpu analysis : ${SCRIPT}"
echo "job_id              : ${SLURM_JOB_ID}"
echo "node                : ${SLURMD_NODENAME:-unknown}"
echo "cpus / RF_NJOBS     : ${SLURM_CPUS_PER_TASK:-?} / ${RF_NJOBS}"
echo "args                : $*"
echo "========================================================================"

cd "${CODE_DIR}"
PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_DIR}" "${PYTHON_BIN}" \
    "${CODE_DIR}/scripts/analysis/${SCRIPT}" "$@"
