#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-raysup}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install uv

uv pip install \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  --index-url "$PYTORCH_INDEX_URL"

uv pip install \
  einops \
  matplotlib \
  numpy \
  timm \
  plotly \
  tensorboard \
  hydra-core \
  rich \
  scikit-learn

cat <<'EOF'

Base environment installed.

Install NATTEN separately after PyTorch. Select the package matching your
PyTorch and CUDA versions from:
https://github.com/SHI-Labs/NATTEN/releases

For another CUDA version, change PYTORCH_INDEX_URL to the matching PyTorch
wheel index before running this script.
EOF
