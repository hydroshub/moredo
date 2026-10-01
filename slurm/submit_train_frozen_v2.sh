#!/bin/bash

# Stabilised frozen-decoder ablation (v2). Same idea as submit_train_frozen.sh -- warm-start the
# MOREDO parameter networks from the no-subsidy baseline, freeze them, train ONLY the recurrent
# lat_alpha subsidy encoder -- but with a training setup that makes all 25 seed x fold runs
# converge instead of ~40% getting stuck in a bad local optimum:
#   --zero_init_subsidy        F_add starts ~0, so every fold begins at the baseline and only grows
#                              the subsidy where it reduces the loss.
#   --learning_rate 1e-3       gentler than the 1e-2 default (the cold-start encoder overshot).
#   --early_stop_patience 30   + more epochs, so the slow-but-correct descent is not cut off.
#
# Writes to results_lateral_experiments/lstm_af_frozen_v2/ (the v1 run is left untouched).
#
# Usage:
#   sbatch --job-name=moredo-lstm_af_frozen_v2 slurm/submit_train_frozen_v2.sh
#   (submit from the repository root; create results_lateral_experiments/logs/ once beforehand)

#SBATCH --job-name=moredo-train-frozen-v2
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

EXP_NAME="${1:-lstm_af_frozen_v2}"
CONFIG_REL="${2:-configs/config_lateral_lstm_af.py}"
BASIN_LIST="$(abspath "${3:-${BASIN_LIST_DEFAULT}}")"
RESULTS_ROOT="$(abspath "${4:-results_lateral_experiments}")"
BASELINE_DIR="$(abspath "${5:-results_lateral_experiments/baseline}")"

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
CV_SEED="${SEED}"

INIT_FROM="${BASELINE_DIR}/best_model_seed${SEED}_fold${FOLD}.pt"

echo "========================================================================"
echo "moredo train (frozen decoder, stabilised v2)"
echo "array=${SLURM_ARRAY_TASK_ID}  seed=${SEED}  fold=${FOLD}"
echo "init_from=${INIT_FROM}"
echo "save_dir=${SAVE_DIR}"
echo "========================================================================"

if [ ! -f "${INIT_FROM}" ]; then
    echo "ERROR: baseline checkpoint not found: ${INIT_FROM}" >&2
    exit 1
fi

PYTHONPATH="${CODE_DIR}" "${PYTHON_BIN}" "${CODE_DIR}/scripts/train.py" \
    --basin_list          "$BASIN_LIST" \
    --config              "$CONFIG_PATH" \
    --fold_id             "$FOLD" \
    --seed                "$SEED" \
    --save_dir            "$SAVE_DIR" \
    --cv_seed             "$CV_SEED" \
    --init_from           "$INIT_FROM" \
    --freeze_subsidy_only \
    --zero_init_subsidy \
    --learning_rate       1e-3 \
    --num_epochs          200 \
    --early_stop_patience 30 \
    --device              cuda:0
