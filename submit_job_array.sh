#!/bin/bash
#SBATCH --job-name=cff_seeds
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cff_seeds_%j.out
#SBATCH --error=cff_seeds_%j.err

# Multi-seed PPO training: runs 3 seeds sequentially in a single job.
# (cs-503 QoS limits us to 1 concurrent job, so a SLURM job array doesn't help.)
#
# Usage (from repo root):
#   sbatch submit_job_array.sh <config_file> [wandb_api_key] [total_timesteps]
#
# Examples:
#   sbatch submit_job_array.sh configs/agent_a_fourrooms.yaml
#   sbatch submit_job_array.sh configs/agent_a_fourrooms.yaml "$WANDB_API_KEY" 1000000

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CONFIG="${1:-configs/agent_a_fourrooms.yaml}"
WANDB_KEY="${2:-}"
TOTAL_STEPS="${3:-1000000}"

SEEDS=(42 43 44)

export PYTHONUNBUFFERED=1
TRACK_FLAG=""
if [[ -n "$WANDB_KEY" ]]; then
  export WANDB_API_KEY="$WANDB_KEY"
  TRACK_FLAG="--track"
fi

# shellcheck disable=SC1091
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

CFG_STEM=$(basename "$CONFIG" .yaml)
echo "JOB_ID=$SLURM_JOB_ID  HOST=$(hostname)  STEPS=$TOTAL_STEPS  SEEDS=${SEEDS[*]}"
nvidia-smi || true

export PYGLET_HEADLESS=true
for SEED in "${SEEDS[@]}"; do
  EXP_NAME="${CFG_STEM}_seed${SEED}"
  echo "==== Starting seed=$SEED  exp=$EXP_NAME ===="
  python scripts/train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --exp-name "$EXP_NAME" \
    --total-timesteps "$TOTAL_STEPS" \
    $TRACK_FLAG
  echo "==== Finished seed=$SEED ===="
done
