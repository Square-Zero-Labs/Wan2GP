#!/bin/bash
set -euo pipefail

REQ_FILE="${1:-/opt/wan2gp_source/requirements.txt}"
PIP_LOG_DIR="/tmp/pip-logs"
mkdir -p "${PIP_LOG_DIR}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-8}"

run_pip_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2

  echo "${step_name}"
  if ! python3 -m pip "$@" --log "${log_file}"; then
    echo ""
    echo "❌ ${step_name} failed"
    echo "--- Last 200 lines from ${log_file} ---"
    tail -n 200 "${log_file}" || true
    echo "--- End pip log tail ---"
    exit 1
  fi
}

echo "=== Python Dependency Install Debug ==="
echo "[env] which python: $(command -v python 2>&1)"
echo "[env] which python3: $(command -v python3 2>&1)"
echo "[env] which pip: $(command -v pip 2>&1)"
echo "[env] which pip3: $(command -v pip3 2>&1)"
echo "[env] python: $(python3 --version 2>&1)"
echo "[env] pip: $(python3 -m pip --version 2>&1)"
echo "[env] working requirements: ${REQ_FILE}"
echo "[env] CUDA_HOME: ${CUDA_HOME}"
echo "[env] TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
if [ ! -f "${REQ_FILE}" ]; then
  echo "❌ Requirements file not found: ${REQ_FILE}"
  exit 1
fi

python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"[env] python tuple: {major}.{minor}")
if (major, minor) != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {major}.{minor}")
PY

echo "[step 1/5] Patching upstream requirements to skip torch/torchvision pins"
sed -i -e 's/^torch>=/#torch>=/' -e 's/^torchvision>=/#torchvision>=/' "${REQ_FILE}"

echo "[step 2/5] Installing Torch 2.10.0 CUDA 13.0 stack"
run_pip_step "[pip] torch stack" "${PIP_LOG_DIR}/01-torch.log" \
  install --no-cache-dir --upgrade --force-reinstall -v \
  torch==2.10.0+cu130 \
  torchvision==0.25.0+cu130 \
  torchaudio==2.10.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130

echo "[step 3/5] Verifying imported torch after install"
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

echo "[step 4/6] Installing app requirements"
run_pip_step "[pip] requirements.txt" "${PIP_LOG_DIR}/02-requirements.log" \
  install --no-cache-dir -v -r "${REQ_FILE}"

echo "[step 5/6] Installing SageAttention 2.2.0"
run_pip_step "[pip] setuptools for sage" "${PIP_LOG_DIR}/03-setuptools.log" \
  install --no-cache-dir -v --force-reinstall "setuptools<=75.8.2"
run_pip_step "[pip] sageattention from git tag v2.2.0" "${PIP_LOG_DIR}/04-sageattention.log" \
  install --no-cache-dir -v --no-build-isolation --force-reinstall \
  "git+https://github.com/thu-ml/SageAttention.git@v2.2.0"

echo "[step 6/6] Installing Runpod gradio override"
run_pip_step "[pip] gradio override" "${PIP_LOG_DIR}/05-gradio.log" \
  install --no-cache-dir -v gradio==5.35.0

echo "[done] Python dependencies installed successfully"
rm -rf /root/.cache/pip "${PIP_LOG_DIR}"
