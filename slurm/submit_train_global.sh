#!/bin/bash

# Usage:
#   sbatch --job-name=moredo-<exp_name> slurm/submit_train_global.sh <exp_name> <config_relpath> [basin_list_path] [results_root]
#   (submit from the repository root; create results_lateral_experiments/logs/ once beforehand)
#   e.g. (default global GRDC basin list):
#   sbatch --job-name=moredo-mlp_af slurm/submit_train_global.sh mlp_af configs/config_lateral_mlp_af.py
#   e.g. (the large-catchment retrain of Fig. S7, on its own basin list):
#   sbatch --job-name=moredo-lstm_af_largebasins slurm/submit_train_global.sh lstm_af_largebasins \
#       configs/config_lateral_lstm_af.py configs/basin_list_largebasins_ge1000km2.txt
#
# Array decoding:
#   1  -> seed=100, fold=0
#   2  -> seed=100, fold=1
#   3  -> seed=100, fold=2
#   4  -> seed=100, fold=3
#   5  -> seed=100, fold=4
#   6  -> seed=200, fold=0
#   ...
#   25 -> seed=500, fold=4

#SBATCH --job-name=moredo-train
#SBATCH --output=results_lateral_experiments/logs/%x-%A_%a.out
#SBATCH --error=results_lateral_experiments/logs/%x-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --array=1-25

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

EXP_NAME="${1:?usage: sbatch --job-name=moredo-<exp> submit_train_global.sh <exp_name> <config_relpath> [basin_list_path] [results_root]}"
CONFIG_REL="${2:?usage: sbatch --job-name=moredo-<exp> submit_train_global.sh <exp_name> <config_relpath> [basin_list_path] [results_root]}"
BASIN_LIST="$(abspath "${3:-${BASIN_LIST_DEFAULT}}")"
RESULTS_ROOT="$(abspath "${4:-results_lateral_experiments}")"

CONFIG_PATH="$(abspath "${CONFIG_REL}")"
SAVE_DIR="${RESULTS_ROOT}/${EXP_NAME}"
LOG_DIR="${CODE_DIR}/results_lateral_experiments/logs"

SEEDS=(100 200 300 400 500)
NUM_FOLDS=5

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"
cd "${WORK_DIR}"

INDEX="${SLURM_ARRAY_TASK_ID}"
NUM_TASKS=$(( ${#SEEDS[@]} * NUM_FOLDS ))

if [ "${INDEX}" -lt 1 ] || [ "${INDEX}" -gt "${NUM_TASKS}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${INDEX} out of range 1-${NUM_TASKS}" >&2
    exit 1
fi

ZERO_BASED=$(( INDEX - 1 ))
SEED="${SEEDS[$(( ZERO_BASED / NUM_FOLDS ))]}"
FOLD=$(( ZERO_BASED % NUM_FOLDS ))
CV_SEED="${SEED}"   # spatial fold assignment follows training seed

echo "========================================================================"
echo "moredo train"
echo "job_id    : ${SLURM_JOB_ID}"
echo "array     : ${SLURM_ARRAY_TASK_ID}"
echo "node      : ${SLURMD_NODENAME:-unknown}"
echo "seed      : ${SEED}"
echo "fold      : ${FOLD}"
echo "cv_seed   : ${CV_SEED}"
echo "code_dir  : ${CODE_DIR}"
echo "work_dir  : ${WORK_DIR}"
echo "save_dir  : ${SAVE_DIR}"
echo "========================================================================"

PYTHONPATH="${CODE_DIR}" "${PYTHON_BIN}" "${CODE_DIR}/scripts/train.py" \
    --basin_list "$BASIN_LIST" \
    --config     "$CONFIG_PATH" \
    --fold_id    "$FOLD" \
    --seed       "$SEED" \
    --save_dir   "$SAVE_DIR" \
    --cv_seed    "$CV_SEED" \
    --device        cuda:0
