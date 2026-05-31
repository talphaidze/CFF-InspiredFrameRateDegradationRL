#!/bin/bash
#SBATCH --job-name=cff_epvideo
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cff_epvideo_%j.out
#SBATCH --error=cff_epvideo_%j.err

# Batch-submit episode-video rendering on SCITAS (Izar).
#
# Runs scripts/record_episode_video.py for an Agent C v2 checkpoint and saves
# an .mp4 (top-down schematic + first-person observation + final HF heatmap).
#
# Usage (from repo root):
#   sbatch submit_job_epvideo.sh <vc1e4ckpt> [out_mp4] [seed]
#
# Where:
#   <vc1e4ckpt>  Agent C v2 checkpoint trained with
#                agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc00001.yaml
#   [out_mp4]    Output video path (optional, default: results/episode_video.mp4).
#   [seed]       Episode seed (optional; if omitted, the script auto-selects).
#
# Example:
#   sbatch submit_job_epvideo.sh runs/<vc1e4_run>/ckpt_vc1e-4.pt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CKPT_VC1E4="${1:?ERROR: vc1e4ckpt must be provided as the 1st argument (Agent C v2 checkpoint trained with ...vc00001.yaml)}"
OUT_MP4="${2:-results/episode_video.mp4}"
SEED="${3:-}"

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
echo "CKPT_VC1E4=$CKPT_VC1E4"
echo "OUT_MP4=$OUT_MP4"
echo "SEED=${SEED:-<auto>}"
nvidia-smi || true

# MiniWorld uses pyglet; on Izar compute nodes (no X display, no Xvfb binary)
# we rely on pyglet's native EGL-based headless mode. The env var is
# redundant with the in-code setting in static_maze.py but harmless.
export PYGLET_HEADLESS=true

SEED_FLAG=()
if [[ -n "$SEED" ]]; then
  SEED_FLAG=(--seed "$SEED")
fi

python scripts/record_episode_video.py \
  --ckpt "$CKPT_VC1E4" \
  --out "$OUT_MP4" \
  "${SEED_FLAG[@]}"
