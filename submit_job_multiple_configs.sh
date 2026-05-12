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
#   sbatch submit_job_multiple_configs.sh [config_folder] [wandb_api_key]
#
# Examples:
#   sbatch submit_job_multiple_configs.sh configs/                                          # TensorBoard only
#   sbatch submit_job_multiple_configs.sh configs/ "$WANDB_API_KEY"                         # wandb + TensorBoard

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CONFIG_FOLDER="${1:-configs/}"
WANDB_KEY="${2:-}"

# Ensure folder ends with /
CONFIG_FOLDER="${CONFIG_FOLDER%/}/"

if [[ ! -d "$CONFIG_FOLDER" ]]; then
  echo "Error: Config folder '$CONFIG_FOLDER' does not exist"
  exit 1
fi

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
echo "CONFIG_FOLDER=$CONFIG_FOLDER"
nvidia-smi || true

# MiniWorld uses pyglet; on Izar compute nodes (no X display, no Xvfb binary)
# we rely on pyglet's native EGL-based headless mode. The env var is
# redundant with the in-code setting in static_maze.py but harmless.
export PYGLET_HEADLESS=true

# Loop through all YAML files in the config folder
for config_file in "$CONFIG_FOLDER"*.yaml; do
  if [[ ! -f "$config_file" ]]; then
    echo "Warning: No YAML files found in $CONFIG_FOLDER"
    exit 1
  fi
  
  echo "=========================================="
  echo "Training with config: $config_file"
  echo "=========================================="
  python scripts/train.py --config "$config_file" $TRACK_FLAG
done

echo "All configs trained successfully!"
