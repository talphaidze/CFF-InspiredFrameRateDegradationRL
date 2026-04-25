#!/usr/bin/env bash
# =============================================================================
# setup_env_izar.sh
# Run once on an Izar login node to install Miniforge + create the conda env
# for the CFF-RL project.
# Usage:  bash setup_env_izar.sh
# =============================================================================
set -euo pipefail

MINIFORGE_DIR="$HOME/miniforge3"
ENV_NAME="cff_rl"
PYTHON_VERSION="3.12"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Install Miniforge (if not already present)
# ---------------------------------------------------------------------------
if [ ! -f "${MINIFORGE_DIR}/bin/conda" ]; then
    echo "=== Downloading Miniforge ==="
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "${MINIFORGE_DIR}"
    rm /tmp/miniforge.sh
else
    echo "=== Miniforge already installed at ${MINIFORGE_DIR} ==="
fi

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. Create conda env
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "=== Conda env '${ENV_NAME}' already exists, skipping creation ==="
else
    echo "=== Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} ==="
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 3. PyTorch (CUDA 12.1 — matches Izar GPU drivers)
# ---------------------------------------------------------------------------
echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Project package (editable) — pulls remaining deps from pyproject.toml
# ---------------------------------------------------------------------------
echo "=== Installing project package from ${REPO_DIR} ==="
cd "${REPO_DIR}"
pip install -e .

echo ""
echo "=== Setup complete! ==="
echo "    Activate with:"
echo "      source ~/miniforge3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
