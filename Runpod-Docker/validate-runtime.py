#!/usr/bin/env python3
"""Validate the image-owned runtime without requiring a GPU at build time."""

from importlib.metadata import version

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
    "sageattention": "2.2.0",
    "spas-sage-attn": "0.1.0",
}


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

import sageattention  # noqa: E402,F401
import spas_sage_attn  # noqa: E402,F401

print(f"Python runtime validated: Torch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"TorchCodec {version('torchcodec')}; Triton {triton.__version__}")
print(f"Decord2 {version('decord2')} API: {decord.__file__}")
print(f"ONNX providers: {onnxruntime.get_available_providers()}")
