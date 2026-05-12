#!/bin/bash
#SBATCH --job-name=cff_eval
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cff_eval_%j.out
#SBATCH --error=cff_eval_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CHECKPOINT="${1:?Usage: sbatch submit_eval_izar.sh <checkpoint> [episodes] [seed1 seed2 ...]}"
EPISODES="${2:-50}"
shift 2 || true
SEEDS=("$@")

export PYTHONUNBUFFERED=1

if [[ -f "${SCRATCH}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

conda activate cff_rl

echo "JOB_ID=$SLURM_JOB_ID  HOST=$(hostname)"
echo "CHECKPOINT=$CHECKPOINT"
nvidia-smi || true

export PYGLET_HEADLESS=true

if [[ ${#SEEDS[@]} -gt 0 ]]; then
  python scripts/eval.py \
    --checkpoint "$CHECKPOINT" \
    --episodes "$EPISODES" \
    --seeds "${SEEDS[@]}"
else
  python scripts/eval.py \
    --checkpoint "$CHECKPOINT" \
    --episodes "$EPISODES"
fi