#!/bin/bash
#SBATCH --job-name=cff_heatmap
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cff_heatmap_%j.out
#SBATCH --error=cff_heatmap_%j.err

# Batch-submit spatial HF-usage heatmap plotting on SCITAS (Izar).
#
# Runs scripts/plot_hf_analyses.py for the two Agent C v2 checkpoints
# (vc=2e-5 and vc=1e-4) and saves the figures used on the project website.
#
# Usage (from repo root):
#   sbatch submit_job_heatmap.sh <vc2e5ckpt> <vc1e4ckpt> [episodes] [out_dir]
#
# Where:
#   <vc2e5ckpt>  Agent C v2 checkpoint trained with
#                agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002.yaml
#   <vc1e4ckpt>  Agent C v2 checkpoint trained with
#                agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc00001.yaml
#   [episodes]   Episodes per seed (optional, default: 500).
#   [out_dir]    Output directory for figures (optional, default: results/plots).
#
# Example:
#   sbatch submit_job_heatmap.sh runs/<vc2e5_run>/ckpt_vc2e-5.pt runs/<vc1e4_run>/ckpt_vc1e-4.pt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CKPT_VC2E5="${1:?ERROR: vc2e5ckpt must be provided as the 1st argument (Agent C v2 checkpoint trained with ...vc000002.yaml)}"
CKPT_VC1E4="${2:?ERROR: vc1e4ckpt must be provided as the 2nd argument (Agent C v2 checkpoint trained with ...vc00001.yaml)}"
EPISODES="${3:-500}"
OUT_DIR="${4:-results/plots}"

if [[ ! -f "$CKPT_VC2E5" ]]; then
  echo "ERROR: vc=2e-5 checkpoint '$CKPT_VC2E5' does not exist." >&2
  exit 1
fi
if [[ ! -f "$CKPT_VC1E4" ]]; then
  echo "ERROR: vc=1e-4 checkpoint '$CKPT_VC1E4' does not exist." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

# MiniWorld uses pyglet; on headless nodes it falls back to EGL/OSMesa.
# PyOpenGL / pyglet will pick a headless path automatically; no DISPLAY needed.

# Conda hook (batch shells often lack it)
# shellcheck disable=SC1091
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
conda activate cff_rl

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CKPT_VC2E5=$CKPT_VC2E5"
echo "CKPT_VC1E4=$CKPT_VC1E4"
echo "EPISODES=$EPISODES"
echo "OUT_DIR=$OUT_DIR"
nvidia-smi || true

# MiniWorld uses pyglet; on Izar compute nodes (no X display, no Xvfb binary)
# we rely on pyglet's native EGL-based headless mode. The env var is
# redundant with the in-code setting in static_maze.py but harmless.
export PYGLET_HEADLESS=true
python scripts/plot_hf_analyses.py \
  --ckpt-vc2e5 "$CKPT_VC2E5" \
  --ckpt-vc1e4 "$CKPT_VC1E4" \
  --episodes "$EPISODES" \
  --out "$OUT_DIR"
