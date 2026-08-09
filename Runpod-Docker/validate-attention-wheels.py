#!/usr/bin/env python3
"""Validate release wheel metadata and embedded CUDA architectures."""

from __future__ import annotations

import argparse
import email
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


EXPECTED = {
    "sageattention": {
        "glob": "sageattention-*.whl",
        "version": "2.2.0",
        "arches": {"86", "120"},
    },
}


def normalized_arches(cuobjdump_output: str) -> set[str]:
    return {value.removesuffix("a") for value in re.findall(r"sm_([0-9]+a?)", cuobjdump_output)}


def validate_wheel(directory: Path, package: str, expected: dict[str, object]) -> None:
    matches = list(directory.glob(str(expected["glob"])))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {expected['glob']} wheel, found: {matches}")
    wheel = matches[0]
    if "cp311-cp311-linux_x86_64" not in wheel.name:
        raise RuntimeError(f"unexpected release tag: {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"{wheel.name}: invalid METADATA entries: {metadata_names}")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if metadata["Name"].lower().replace("-", "_") != package:
            raise RuntimeError(f"{wheel.name}: unexpected package name {metadata['Name']}")
        if metadata["Version"].split("+", 1)[0] != expected["version"]:
            raise RuntimeError(f"{wheel.name}: unexpected version {metadata['Version']}")

        with tempfile.TemporaryDirectory(prefix=f"{package}-wheel-") as temp_dir:
            archive.extractall(temp_dir)
            shared_objects = list(Path(temp_dir).rglob("*.so"))
            if not shared_objects:
                raise RuntimeError(f"{wheel.name}: no native extension modules")
            output = "\n".join(
                subprocess.check_output(["cuobjdump", "--list-elf", str(shared_object)], text=True)
                for shared_object in shared_objects
            )

    arches = normalized_arches(output)
    missing = set(expected["arches"]) - arches
    if missing:
        raise RuntimeError(f"{wheel.name}: missing CUDA architectures {sorted(missing)}; found {sorted(arches)}")
    print(f"{wheel.name}: metadata and CUDA architectures validated ({sorted(arches)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_directory", type=Path)
    args = parser.parse_args()
    for package, expected in EXPECTED.items():
        validate_wheel(args.wheel_directory, package, expected)


if __name__ == "__main__":
    main()
