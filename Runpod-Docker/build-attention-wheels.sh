#!/bin/bash
set -Eeuo pipefail

# Build release artifacts for Square-Zero-Labs/Wan2GP-Runpod-Wheels.
# Run this on a Linux x86_64 machine using the Torch 2.10 / CUDA 12.8 image.

SAGEATTENTION_COMMIT="${SAGEATTENTION_COMMIT:-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}"
WHEEL_OUTPUT_DIR="${WHEEL_OUTPUT_DIR:-/workspace/wan2gp-attention-wheels}"
BUILD_ROOT="${ATTENTION_BUILD_ROOT:-/workspace/wan2gp-attention-build}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
MAX_JOBS="${MAX_JOBS:-2}"
EXT_PARALLEL="${EXT_PARALLEL:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CC CXX CUDAHOSTCXX

if [ -z "$WHEEL_OUTPUT_DIR" ] || [ -z "$BUILD_ROOT" ] \
  || [ "$WHEEL_OUTPUT_DIR" = / ] || [ "$BUILD_ROOT" = / ] \
  || [ "$WHEEL_OUTPUT_DIR" = /workspace ] || [ "$BUILD_ROOT" = /workspace ]; then
  echo "ERROR: wheel output and build roots must be dedicated subdirectories" >&2
  exit 1
fi

if [ "$(uname -m)" != "x86_64" ]; then
  echo "ERROR: release wheels must be built on Linux x86_64, found $(uname -m)" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print(f"Build runtime: Python {sys.version.split()[0]}, Torch {torch.__version__}, CUDA {torch.version.cuda}")
PY

mkdir -p "$WHEEL_OUTPUT_DIR" "$BUILD_ROOT"
rm -f "$WHEEL_OUTPUT_DIR"/*.whl "$WHEEL_OUTPUT_DIR"/SHA256SUMS.txt "$WHEEL_OUTPUT_DIR"/BUILD-INFO.txt

"$PYTHON_BIN" -m pip install --no-cache-dir --upgrade \
  "setuptools<=75.8.2" wheel ninja packaging build

rm -rf "$BUILD_ROOT/SageAttention"
git clone --filter=blob:none https://github.com/thu-ml/SageAttention.git "$BUILD_ROOT/SageAttention"
git -C "$BUILD_ROOT/SageAttention" checkout --detach "$SAGEATTENTION_COMMIT"
(
  cd "$BUILD_ROOT/SageAttention"
  TORCH_CUDA_ARCH_LIST="8.6;12.0" \
    MAX_JOBS="$MAX_JOBS" \
    EXT_PARALLEL="$EXT_PARALLEL" \
    "$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation . -w "$WHEEL_OUTPUT_DIR"
)

(
  cd "$WHEEL_OUTPUT_DIR"
  sha256sum ./*.whl > SHA256SUMS.txt
  {
    echo "python=$($PYTHON_BIN -c 'import sys; print(sys.version.split()[0])')"
    echo "torch=$($PYTHON_BIN -c 'import torch; print(torch.__version__)')"
    echo "cuda=$($PYTHON_BIN -c 'import torch; print(torch.version.cuda)')"
    echo "cc=$($CC --version | sed -n '1p')"
    echo "cxx=$($CXX --version | sed -n '1p')"
    echo "nvcc=$(nvcc --version | sed -n '/release /p')"
    echo "sageattention_commit=$SAGEATTENTION_COMMIT"
    echo "sageattention_arches=8.6;12.0"
  } > BUILD-INFO.txt
)

"$PYTHON_BIN" "$SCRIPT_DIR/validate-attention-wheels.py" "$WHEEL_OUTPUT_DIR"
if [ "${SKIP_NATIVE_IMPORT_TEST:-0}" != "1" ]; then
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$WHEEL_OUTPUT_DIR"/*.whl
  "$PYTHON_BIN" - <<'PY'
import sageattention._fused
import sageattention._qattn_sm80
import sageattention._qattn_sm89

print("SageAttention native extension imports passed")
PY
fi

echo "Built release assets in $WHEEL_OUTPUT_DIR"
ls -lh "$WHEEL_OUTPUT_DIR"
