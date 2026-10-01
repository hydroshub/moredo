#!/bin/bash

# Usage:
#   sbatch --job-name=moredo-avg-<exp_name> slurm/submit_build_averaged.sh <exp_name> <config_relpath> [basin_list_path] [results_root]
#   (submit from the repository root; create results_lateral_experiments/logs/ once beforehand)
#   e.g.
#   sbatch --job-name=moredo-avg-lstm_af slurm/submit_build_averaged.sh lstm_af configs/config_lateral_lstm_af.py
#
# Builds averaged_full_output.pickle for one experiment from its 25 trained
# best_model_seed{s}_fold{f}.pt checkpoints (see build_averaged_output.py).

#SBATCH --job-name=moredo-avg
#SBATCH --output=results_lateral_experiments/logs/%x-%j.out
#SBATCH --error=results_lateral_experiments/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G

set -euo pipefail

# --- site settings. Defaults: sbatch is run from the repository root with the Python environment
#     activated; slurm/site_local.sh (optional, not part of the release) can override them ---
CODE_DIR="${SLURM_SUBMIT_DIR}"                                  # repository root
PYTHON_BIN="python"
WORK_DIR="${CODE_DIR}"                                          # the data loader keeps cache/ here
BASIN_LIST_DEFAULT="${CODE_DIR}/configs/basin_list_global_grdc.txt"
if [ -f "${SLURM_SUBMIT_DIR}/slurm/site_local.sh" ]; then source "${SLURM_SUBMIT_DIR}/slurm/site_local.sh"; fi
# relative paths given on the command line are relative to the repository root
abspath() { case "$1" in /*) echo "$1" ;; *) echo "${CODE_DIR}/$1" ;; esac; }

EXP_NAME="${1:?usage: sbatch submit_build_averaged.sh <exp_name> <config_relpath> [basin_list_path] [results_root]}"
CONFIG_REL="${2:?usage: sbatch submit_build_averaged.sh <exp_name> <config_relpath> [basin_list_path] [results_root]}"
BASIN_LIST="$(abspath "${3:-${BASIN_LIST_DEFAULT}}")"
RESULTS_ROOT="$(abspath "${4:-results_lateral_experiments}")"

CONFIG_PATH="$(abspath "${CONFIG_REL}")"
MODEL_DIR="${RESULTS_ROOT}/${EXP_NAME}"
OUT_PATH="${MODEL_DIR}/averaged_full_output.pickle"
LOG_DIR="${CODE_DIR}/results_lateral_experiments/logs"

mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

echo "========================================================================"
echo "moredo build_averaged_output"
echo "job_id     : ${SLURM_JOB_ID}"
echo "node       : ${SLURMD_NODENAME:-unknown}"
echo "exp_name   : ${EXP_NAME}"
echo "config     : ${CONFIG_PATH}"
echo "basin_list : ${BASIN_LIST}"
echo "model_dir  : ${MODEL_DIR}"
echo "out        : ${OUT_PATH}"
echo "========================================================================"

PYTHONPATH="${CODE_DIR}" "${PYTHON_BIN}" "${CODE_DIR}/scripts/analysis/build_averaged_output.py" \
    --config     "$CONFIG_PATH" \
    --basin_list "$BASIN_LIST" \
    --model_dir  "$MODEL_DIR" \
    --out        "$OUT_PATH" \
    --device     cuda:0
