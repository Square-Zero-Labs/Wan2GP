#!/usr/bin/env python3
"""Validate image-owned packages and source-coupled web dependencies."""

import os
import re
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

import onnxruntime
import torch
import torchcodec
import triton
import decord
import huggingface_hub
import transformers


EXPECTED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "torchcodec": "0.10.0",
    "triton": "3.6.0",
    "onnxruntime-gpu": "1.26.0",
    "decord2": "3.4.0",
    "hf-xet": "1.6.0",
    "sageattention": "2.2.0",
}
TRUE_ENV_VALUES = {"1", "ON", "YES", "TRUE"}


def source_root() -> Path:
    candidates = [
        Path(os.environ["WAN2GP_APP_DIR"]) if os.environ.get("WAN2GP_APP_DIR") else None,
        Path("/workspace/Wan2GP"),
        Path("/opt/wan2gp_source"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "requirements.txt").is_file() and (candidate / "wgp.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the active Wan2GP source tree")


def validate_source_requirement(root: Path, package: str) -> None:
    expected_name = canonicalize_name(package)
    for line_number, line in enumerate((root / "requirements.txt").read_text(encoding="utf-8").splitlines(), 1):
        candidate = re.split(r"\s+#", line.strip(), maxsplit=1)[0]
        if not candidate or candidate.startswith(("-", "http://", "https://", "git+")):
            continue
        try:
            requirement = Requirement(candidate)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != expected_name:
            continue
        installed = Version(version(requirement.name).split("+", 1)[0])
        if requirement.specifier and installed not in requirement.specifier:
            raise RuntimeError(
                f"{root}/requirements.txt:{line_number}: installed {requirement.name}=={installed} "
                f"does not satisfy {requirement}"
            )
        return
    raise RuntimeError(f"{root}/requirements.txt does not declare required package {package}")


def validate_gradio_frontend(root: Path) -> None:
    patch = root / "shared/gradio/gradio_frontend_patch.py"
    if not patch.is_file():
        return
    sys.path.insert(0, str(root))
    try:
        from shared.gradio import gradio_frontend_patch

        # install() validates every compiled asset and replacement signature.
        # This process exits immediately afterward, so its monkey patches are
        # intentionally confined to validation.
        gradio_frontend_patch.install()
    finally:
        sys.path.pop(0)


for package, expected in EXPECTED.items():
    actual = version(package).split("+", 1)[0]
    if actual != expected:
        raise RuntimeError(f"{package}: expected {expected}, found {actual}")

if torch.version.cuda != "12.8":
    raise RuntimeError(f"PyTorch must use CUDA 12.8, found {torch.version.cuda}")
if triton.__version__ != EXPECTED["triton"]:
    raise RuntimeError(f"Triton mismatch: {triton.__version__}")

active_source = source_root()
validate_source_requirement(active_source, "gradio")
validate_source_requirement(active_source, "huggingface-hub")
validate_source_requirement(active_source, "transformers")
validate_gradio_frontend(active_source)
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
print(f"Gradio {version('gradio')}; Transformers {transformers.__version__}; Hugging Face Hub {huggingface_hub.__version__}")
print(f"HF Xet {version('hf-xet')}; high-performance mode enabled")
print(f"Decord2 {version('decord2')} API: {decord.__file__}")
print(f"ONNX providers: {onnxruntime.get_available_providers()}")
