#!/bin/bash
#SBATCH --job-name=cff_eval
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cff_eval_mul_%j.out
#SBATCH --error=cff_eval_mul_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

TARGET="${1:?Usage: sbatch submit_eval_multiple.sh <checkpoint|runs_dir> [episodes] [seed1 seed2 ...] [--env-id ENV_ID] [--record-video]}"
EPISODES="${2:-50}"
shift 2 || true
SEEDS=()
RECORD_VIDEO=false
ENV_ID="MiniWorld-FourRooms-v0"

while (($#)); do
  case "$1" in
    --env-id)
      ENV_ID="${2:?--env-id requires a value}"
      shift 2
      ;;
    --record-video)
      RECORD_VIDEO=true
      shift
      ;;
    --)
      shift
      SEEDS+=("$@")
      break
      ;;
    *)
      SEEDS+=("$1")
      shift
      ;;
  esac
done

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
nvidia-smi || true

export PYGLET_HEADLESS=true

eval_args=(--episodes "$EPISODES")

if [[ -n "$ENV_ID" ]]; then
  eval_args+=(--env-id "$ENV_ID")
fi

if [[ ${#SEEDS[@]} -gt 0 ]]; then
  eval_args+=(--seeds "${SEEDS[@]}")
fi

if [[ "$RECORD_VIDEO" == true ]]; then
  eval_args+=(--record-video)
fi

if [[ -d "$TARGET" ]]; then
  mapfile -t CHECKPOINTS < <(find "$TARGET" -type f -name 'ckpt_000732.pt' | sort)
  if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No checkpoints named ckpt_000732.pt found under $TARGET"
    exit 1
  fi

  if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(1 2 3 4 5)
  fi

  echo "RUNS_ROOT=$TARGET"
  echo "EPISODES=$EPISODES"
  echo "ENV_ID=$ENV_ID"
  echo "SEEDS=${SEEDS[*]}"
  printf 'CHECKPOINTS_FOUND=%s\n' "${CHECKPOINTS[@]}"

  for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "=========================================="
    echo "Evaluating checkpoint: $checkpoint"
    echo "=========================================="
    python scripts/eval.py \
      --checkpoint "$checkpoint" \
      "${eval_args[@]}"
  done
else
  CHECKPOINT="$TARGET"
  echo "CHECKPOINT=$CHECKPOINT"
  echo "ENV_ID=$ENV_ID"

  python scripts/eval.py \
    --checkpoint "$CHECKPOINT" \
    "${eval_args[@]}"
fi