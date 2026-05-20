#!/bin/bash
#SBATCH --job-name=cff_agent_a_eval
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Batch-submit PPO eval on SCITAS (Izar).
#
# Usage (from repo root):
#   sbatch submit_job_eval.sh <run_name> <env_id>
#
# Example:
#   sbatch submit_job_eval.sh agent_c2_fourroomshard_v2_lstm_proprio_turn10__42__1779140117 MiniWorld-FourRoomsHard-v0                                    

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

RUN_NAME="${1:?ERROR: run_name must be provided as the 1st argument (e.g. agent_c2_fourroomshard_v2_lstm_proprio_turn10__42__1779140117)}"
ENV_ID="${2:?ERROR: env_id must be provided as the 2nd argument (e.g. MiniWorld-FourRoomsHard-v0)}"

RUN_DIR="runs/$RUN_NAME"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: Run directory '$RUN_DIR' does not exist." >&2
  exit 1
fi

# ── Redirect all stdout/stderr into the run directory ────────────────────────
# SBATCH --output/--error are set to /dev/null above so SLURM does not create
# files in the working directory. We take over here so both files land in RUN_DIR.
exec > "$RUN_DIR/cff_agent_a_${SLURM_JOB_ID}.out" \
     2>"$RUN_DIR/cff_agent_a_${SLURM_JOB_ID}.err"

export PYTHONUNBUFFERED=1
TRACK_FLAG=""

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

# ── Find the latest checkpoint in the run directory ──────────────────────────
CKPT=$(ls -t "$RUN_DIR"/*.pt 2>/dev/null | head -1 || true)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: No .pt checkpoint found in $RUN_DIR" >&2
  exit 1
fi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "RUN_DIR=$RUN_DIR"
echo "ENV_ID=$ENV_ID"
echo "CKPT=$CKPT"
nvidia-smi || true

# MiniWorld uses pyglet; on Izar compute nodes (no X display, no Xvfb binary)
# we rely on pyglet's native EGL-based headless mode. The env var is
# redundant with the in-code setting in static_maze.py but harmless.
export PYGLET_HEADLESS=true
python scripts/eval.py \
  --checkpoint "$CKPT" \
  --seeds 1 2 3 4 5 --episodes 50 \
  --record-video --video-dir "$RUN_DIR/eval_videos"\
  --env-id "$ENV_ID"
