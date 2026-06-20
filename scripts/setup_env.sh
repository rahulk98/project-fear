#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/opt/anaconda/etc/profile.d/conda.sh"
ENV_NAME="asym-rewards"
PY_VERSION="3.11"
REQ_FILE="env/requirements.txt"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda not found at $CONDA_SH" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Creating conda env $ENV_NAME (python=$PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
else
    echo ">>> Conda env $ENV_NAME already exists, reusing"
fi

conda activate "$ENV_NAME"

echo ">>> Upgrading pip"
pip install --upgrade pip

echo ">>> Installing pinned deps from $REQ_FILE"
pip install -r "$REQ_FILE"

echo ">>> Verifying CUDA visibility"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'OK: {torch.cuda.device_count()} GPUs, primary={torch.cuda.get_device_name(0)}')"

echo ">>> Verifying core imports"
python -c "import transformers, trl, peft, vllm, datasets, wandb, ollama, math_verify; print('imports OK')"

echo ">>> Done. Activate with: conda activate $ENV_NAME"
