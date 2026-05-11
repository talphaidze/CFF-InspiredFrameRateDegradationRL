#!/bin/bash
#SBATCH --job-name=cff_agent_a
#SBATCH --time=08:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cff_agent_a_%j.out
#SBATCH --error=cff_agent_a_%j.err

# Batch-submit Agent A (35 Hz baseline) PPO training on SCITAS (Izar).
#
# Usage (from repo root):
#   sbatch submit_job.sh [config_file] [wandb_api_key]
#
# Examples:
#   sbatch submit_job.sh                                                # TensorBoard only
#   sbatch submit_job.sh configs/agent_a_static.yaml                    # TensorBoard only
#   sbatch submit_job.sh configs/agent_a_static.yaml "$WANDB_API_KEY"   # wandb + TensorBoard

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CONFIG="${1:-configs/agent_a_static.yaml}"
WANDB_KEY="${2:-}"

export PYTHONUNBUFFERED=1
TRACK_FLAG=""
if [[ -n "$WANDB_KEY" ]]; then
  export WANDB_API_KEY="$WANDB_KEY"
  TRACK_FLAG="--track"
fi
# MiniWorld uses pyglet; on headless nodes it falls back to EGL/OSMesa.
# PyOpenGL / pyglet will pick a headless path automatically; no DISPLAY needed.

# Conda hook (batch shells often lack it)
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

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG=$CONFIG"
nvidia-smi || true

# MiniWorld uses pyglet; on Izar compute nodes (no X display, no Xvfb binary)
# we rely on pyglet's native EGL-based headless mode. The env var is
# redundant with the in-code setting in static_maze.py but harmless.
export PYGLET_HEADLESS=true
python scripts/train.py --config "$CONFIG" $TRACK_FLAG
