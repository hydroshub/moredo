#!/bin/bash
# Full 25-instance ensemble: build averaged output (all 5 seeds) + memory_truncation for one
# LSTM(A,F) hidden-size variant. Usage (from the repository root; create
# results_lateral_experiments/logs/ once beforehand):
#   sbatch slurm/submit_hsweep_mem_full.sh <exp> <cfg_rel> <tag>
#   e.g. sbatch slurm/submit_hsweep_mem_full.sh lstm_af_h16 configs/config_lateral_lstm_af_h16.py h16
#SBATCH --output=results_lateral_experiments/logs/%x-%j.out
#SBATCH --error=results_lateral_experiments/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=128G
set -euo pipefail
# --- site settings. Defaults: sbatch is run from the repository root with the Python environment
#     activated; slurm/site_local.sh (optional, not part of the release) can override them ---
CODE_DIR="${SLURM_SUBMIT_DIR}"                                  # repository root
PYTHON_BIN="python"
WORK_DIR="${CODE_DIR}"                                          # the data loader keeps cache/ here
BASIN_LIST_DEFAULT="${CODE_DIR}/configs/basin_list_global_grdc.txt"
if [ -f "${SLURM_SUBMIT_DIR}/slurm/site_local.sh" ]; then source "${SLURM_SUBMIT_DIR}/slurm/site_local.sh"; fi
EXP="${1:?}"; CFG_REL="${2:?}"; TAG="${3:?}"
CODE="${CODE_DIR}"; PY="${PYTHON_BIN}"; BL="${BASIN_LIST_DEFAULT}"
cd "${WORK_DIR}"
CFG=${CODE}/${CFG_REL}; MODEL_DIR=${CODE}/results_lateral_experiments/${EXP}
echo "=== build_averaged (25 seeds) ${EXP} ==="
PYTHONPATH="${CODE}" "${PY}" "${CODE}/scripts/analysis/build_averaged_output.py" \
    --config "${CFG}" --basin_list "${BL}" --model_dir "${MODEL_DIR}" \
    --out "${MODEL_DIR}/averaged_full_output.pickle" --device cuda:0
echo "=== memory_truncation (25 seeds) ${EXP} tag ${TAG} ==="
PYTHONPATH="${CODE}" "${PY}" "${CODE}/scripts/analysis/memory_truncation.py" \
    --config "${CFG}" --model_dir "${MODEL_DIR}" \
    --seeds 100,200,300,400,500 --tag "${TAG}" --device cuda
