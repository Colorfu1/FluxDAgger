#!/bin/bash
# ============================================================
# FluxDAgger environment setup script.
# Default: creates conda env "fluxdagger" and installs deps + ROS .pth hooks.
#
# Use an existing env (skip conda create):
#   DAGGER_ENV_NAME=fluxdagger_doc_verify SKIP_CONDA_CREATE=1 bash setup_env.sh
#
# Pip index: defaults to official PyPI. For a China mirror:
#   DAGGER_DISABLE_PIP_MIRROR=0 bash setup_env.sh
# Override mirror URL:
#   DAGGER_DISABLE_PIP_MIRROR=0 DAGGER_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash setup_env.sh
#
# Requirements:
#   - Ubuntu 20.04 + ROS Noetic
#   - NVIDIA Driver >= 535 + CUDA 12.x
#   - conda (miniconda/anaconda)
#
# Usage:
#   bash setup_env.sh
#   # Or run it from any directory:
#   bash /path/to/FluxDAgger/setup_env.sh
# ============================================================
set -e

ENV_NAME="${DAGGER_ENV_NAME:-fluxdagger}"
PYTHON_VERSION="${DAGGER_PYTHON_VERSION:-3.10}"
SKIP_CONDA_CREATE="${SKIP_CONDA_CREATE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGGER_ROOT="${DAGGER_ROOT:-${SCRIPT_DIR}}"

DAGGER_DISABLE_PIP_MIRROR="${DAGGER_DISABLE_PIP_MIRROR:-1}"
DAGGER_PIP_INDEX_URL="${DAGGER_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_MIRROR_ARGS=()
if [[ "${DAGGER_DISABLE_PIP_MIRROR}" != "1" ]]; then
  PIP_MIRROR_ARGS=(-i "${DAGGER_PIP_INDEX_URL}")
fi

echo "=========================================="
echo " FluxDAgger environment setup: ${ENV_NAME}"
echo "=========================================="

eval "$(conda shell.bash hook)"

# ---- Step 1: Create conda environment (optional) ----
if [[ "${SKIP_CONDA_CREATE}" == "1" ]]; then
  echo "[1/5] Skipping conda create; activating existing env ${ENV_NAME}..."
  conda activate "${ENV_NAME}"
else
  echo "[1/5] Creating conda environment ${ENV_NAME} (Python ${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
  conda activate "${ENV_NAME}"
fi

echo "  Python: $(python --version)"
echo "  pip:    $(pip --version)"
if [[ "${DAGGER_DISABLE_PIP_MIRROR}" != "1" ]]; then
  echo "  pip -i : ${DAGGER_PIP_INDEX_URL}"
else
  echo "  pip -i : (disabled, using pip default — usually pypi.org)"
fi

# ---- Step 2: Upgrade pip (helps resolver with torch + heavy deps) ----
echo "[2/5] Upgrading pip / setuptools / wheel..."
pip install "${PIP_MIRROR_ARGS[@]}" -U pip setuptools wheel

# ---- Step 3: Install all Python deps (torch+cu121 via extra-index in requirements.txt) ----
echo "[3/5] Installing requirements.txt..."
pip install "${PIP_MIRROR_ARGS[@]}" -r "${DAGGER_ROOT}/requirements.txt"

# ---- Step 4: Configure ROS Noetic Python path ----
echo "[4/5] Configuring ROS Noetic compatibility..."
# conda Python does not load /usr/lib/python3/dist-packages where apt's python3-rospkg
# normally lives; requirements.txt installs rospkg/catkin_pkg via pip into this env.
# ROS Noetic Python packages live in /opt/ros/noetic/lib/python3/dist-packages.
# Add that path to the conda environment site-packages through a .pth file.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
ROS_PTH="${SITE_PACKAGES}/ros_noetic.pth"
echo "/opt/ros/noetic/lib/python3/dist-packages" > "${ROS_PTH}"
echo "  Wrote ROS path: ${ROS_PTH}"

# Also add the FluxDAgger catkin workspace Python path.
DAGGER_PTH="${SITE_PACKAGES}/dagger_catkin.pth"
echo "${DAGGER_ROOT}/devel/lib/python3/dist-packages" > "${DAGGER_PTH}"
echo "  Wrote FluxDAgger catkin path: ${DAGGER_PTH}"

# ---- Step 5: Verify installation ----
echo "[5/5] Verifying installation..."
python -c "
import sys
print(f'Python {sys.version}')

import torch
print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')

import torchvision
print(f'torchvision {torchvision.__version__}')

import numpy, cv2, pandas, pyarrow
print(f'numpy {numpy.__version__}, opencv {cv2.__version__}, pandas {pandas.__version__}, pyarrow {pyarrow.__version__}')

import transformers
print(f'transformers {transformers.__version__}')

import rospy
print(f'rospy OK: {rospy.__file__}')

print()
print('=== All checks passed! ===')
"

echo ""
echo "=========================================="
echo " Environment ${ENV_NAME} setup complete!"
echo ""
echo " Usage:"
echo "   conda activate ${ENV_NAME}"
echo "   source /opt/ros/noetic/setup.bash"
echo "   source ${DAGGER_ROOT}/devel/setup.bash"
echo "   roslaunch dagger dagger.launch ..."
echo "=========================================="
