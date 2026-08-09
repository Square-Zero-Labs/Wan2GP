#!/usr/bin/env python3
"""Validate the image-owned runtime without requiring a GPU at build time."""

import os
from importlib.metadata import version
from importlib.util import find_spec

import onnxruntime
import torch
import torchcodec
import triton
import decord


EXPECTED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "torchcodec": "0.10.0",
    "triton": "3.6.0",
    "onnxruntime-gpu": "1.26.0",
    "gradio": "5.35.0",
    "decord2": "3.4.0",
    "hf-xet": "1.6.0",
    "sageattention": "2.2.0",
}
TRUE_ENV_VALUES = {"1", "ON", "YES", "TRUE"}


for package, expected in EXPECTED.items():
    actual = version(package).split("+", 1)[0]
    if actual != expected:
        raise RuntimeError(f"{package}: expected {expected}, found {actual}")

if torch.version.cuda != "12.8":
    raise RuntimeError(f"PyTorch must use CUDA 12.8, found {torch.version.cuda}")
if triton.__version__ != EXPECTED["triton"]:
    raise RuntimeError(f"Triton mismatch: {triton.__version__}")
if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    raise RuntimeError(f"ONNX Runtime CUDA provider missing: {onnxruntime.get_available_providers()}")
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").upper() in TRUE_ENV_VALUES:
    raise RuntimeError("HF_HUB_ENABLE_HF_TRANSFER must be disabled on RunPod")
if os.environ.get("HF_HUB_DISABLE_XET", "").upper() in TRUE_ENV_VALUES:
    raise RuntimeError("HF_HUB_DISABLE_XET must not disable the image-owned hf-xet runtime")
if os.environ.get("HF_XET_HIGH_PERFORMANCE", "").upper() not in TRUE_ENV_VALUES:
    raise RuntimeError("HF_XET_HIGH_PERFORMANCE must be enabled on RunPod")

import sageattention  # noqa: E402,F401
import hf_xet  # noqa: E402,F401

if find_spec("spas_sage_attn") is not None:
    raise RuntimeError("SpargeAttention must not be installed in the base image")

print(f"Python runtime validated: Torch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"TorchCodec {version('torchcodec')}; Triton {triton.__version__}")
print(f"HF Xet {version('hf-xet')}; high-performance mode enabled")
print(f"Decord2 {version('decord2')} API: {decord.__file__}")
print(f"ONNX providers: {onnxruntime.get_available_providers()}")
