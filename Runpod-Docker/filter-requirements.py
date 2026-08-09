#!/usr/bin/env python3
"""Remove image-owned packages from an upstream Wan2GP requirements file.

The filtered file is safe to install with core-constraints.txt. Incompatible
managed requirements fail closed unless their package is explicitly allowed as
an intentional container override (currently Gradio and ONNX Runtime).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


LOCKED = {
    "torch": Version("2.10.0"),
    "torchvision": Version("0.25.0"),
    "torchaudio": Version("2.10.0"),
    "torchcodec": Version("0.10.0"),
    "triton": Version("3.6.0"),
    "onnxruntime-gpu": Version("1.26.0"),
    "gradio": Version("5.35.0"),
    "decord2": Version("3.4.0"),
    "hf-xet": Version("1.6.0"),
    "sageattention": Version("2.2.0"),
}
EXCLUDED = {"spas-sage-attn"}
REPLACEMENTS = {
    # The 2021 decord wheel falsely embeds a CPython 3.6 platform tag. Decord2
    # is the maintained API-compatible distribution with CPython 3.11 wheels.
    "decord": "decord2==3.4.0",
}
DEFAULT_ALLOWED_MISMATCHES = {"gradio", "onnxruntime-gpu"}
ORT_NIGHTLY_INDEX = "aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly"


def parse_requirement(line: str) -> Requirement | None:
    candidate = re.split(r"\s+#", line.strip(), maxsplit=1)[0]
    if not candidate or candidate.startswith(("-", "http://", "https://", "git+")):
        return None
    try:
        return Requirement(candidate)
    except InvalidRequirement:
        return None


def filter_requirements(source: Path, destination: Path, allow_mismatch: set[str]) -> None:
    output: list[str] = []
    removed: list[str] = []

    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if ORT_NIGHTLY_INDEX.lower() in line.lower():
            removed.append(f"line {line_number}: removed CUDA 13 ORT nightly index")
            continue

        requirement = parse_requirement(line)
        if requirement is None:
            output.append(line)
            continue

        name = canonicalize_name(requirement.name)
        if name in EXCLUDED:
            removed.append(f"line {line_number}: excluded optional package {requirement!s}")
            continue
        if name in REPLACEMENTS:
            removed.append(f"line {line_number}: {requirement!s} -> {REPLACEMENTS[name]}")
            continue
        if name not in LOCKED:
            output.append(line)
            continue

        locked_version = LOCKED[name]
        if requirement.specifier and locked_version not in requirement.specifier and name not in allow_mismatch:
            raise SystemExit(
                f"{source}:{line_number}: upstream requirement {requirement!s} rejects "
                f"the image-owned {name}=={locked_version}"
            )
        removed.append(f"line {line_number}: {requirement!s} -> image-owned {name}=={locked_version}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    for message in removed:
        print(f"[requirements] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--allow-mismatch",
        action="append",
        default=[],
        help="Managed package whose upstream spec may intentionally differ",
    )
    args = parser.parse_args()
    allowed = DEFAULT_ALLOWED_MISMATCHES | {canonicalize_name(name) for name in args.allow_mismatch}
    filter_requirements(args.source, args.destination, allowed)


if __name__ == "__main__":
    main()
