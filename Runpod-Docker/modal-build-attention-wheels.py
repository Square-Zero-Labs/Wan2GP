#!/usr/bin/env python3
"""Build Wan2GP attention wheels on Modal and persist verified artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "wan2gp-attention-wheel-builder"
VOLUME_NAME = "wan2gp-runpod-wheels"
VOLUME_MOUNT = Path("/modal-wheels")
RELEASE_TAG = "pytorch2.10.0-cu128-py311-sm86-sm120-sage-v1"
SAGEATTENTION_COMMIT = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"

LOCAL_DIR = Path(__file__).resolve().parent
REMOTE_DIR = Path("/repo/Runpod-Docker")

build_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("build-essential", "ca-certificates", "git")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file(
        LOCAL_DIR / "build-attention-wheels.sh",
        str(REMOTE_DIR / "build-attention-wheels.sh"),
    )
    .add_local_file(
        LOCAL_DIR / "validate-attention-wheels.py",
        str(REMOTE_DIR / "validate-attention-wheels.py"),
    )
)

app = modal.App(APP_NAME)
wheel_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.whl"))
    if [path.name.split("-", 1)[0] for path in paths] != ["sageattention"]:
        raise RuntimeError(f"unexpected wheel outputs: {[path.name for path in paths]}")
    return paths


def validate_native_imports(paths: list[Path]) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", *paths],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sageattention._fused",
                    "import sageattention._qattn_sm80",
                    "import sageattention._qattn_sm89",
                    'print("SageAttention native extension imports passed")',
                ]
            ),
        ],
        check=True,
    )


@app.function(
    image=build_image,
    cpu=16,
    memory=65536,
    nonpreemptible=True,
    timeout=4 * 60 * 60,
    volumes={VOLUME_MOUNT: wheel_volume},
)
def build_and_store() -> dict[str, object]:
    if platform.machine() != "x86_64":
        raise RuntimeError(f"release build requires x86_64, found {platform.machine()}")

    release_dir = VOLUME_MOUNT / "releases" / RELEASE_TAG
    staging_dir = VOLUME_MOUNT / "staging" / RELEASE_TAG
    if release_dir.exists():
        raise RuntimeError(f"immutable release directory already exists: {release_dir}")

    if staging_dir.exists():
        print(f"Reusing persisted wheel checkpoint: {staging_dir}")
        subprocess.run(
            [sys.executable, str(REMOTE_DIR / "validate-attention-wheels.py"), str(staging_dir)],
            check=True,
        )
        paths = wheel_paths(staging_dir)
        manifest = json.loads((staging_dir / "MODAL-BUILD.json").read_text(encoding="utf-8"))
        manifest_entries = manifest["wheels"]
        actual_hashes = {path.name: sha256(path) for path in paths}
        expected_hashes = {entry["filename"]: entry["sha256"] for entry in manifest_entries}
        if actual_hashes != expected_hashes:
            raise RuntimeError("persisted wheel checkpoint does not match MODAL-BUILD.json")
    else:
        job_root = Path(tempfile.mkdtemp(prefix="wan2gp-attention-build-"))
        dist_dir = job_root / "dist"
        source_dir = job_root / "source"
        dist_dir.mkdir()

        environment = os.environ.copy()
        environment.update(
            {
                "ATTENTION_BUILD_ROOT": str(source_dir),
                "CC": "gcc",
                "CUDAHOSTCXX": "g++",
                "CXX": "g++",
                "EXT_PARALLEL": "1",
                "MAX_JOBS": "2",
                "PIP_VERBOSE": "1",
                "PYTHON_BIN": "python",
                "SAGEATTENTION_COMMIT": SAGEATTENTION_COMMIT,
                "SKIP_NATIVE_IMPORT_TEST": "1",
                "WHEEL_OUTPUT_DIR": str(dist_dir),
            }
        )

        subprocess.run(
            ["bash", str(REMOTE_DIR / "build-attention-wheels.sh")],
            check=True,
            env=environment,
        )

        paths = wheel_paths(dist_dir)
        manifest_entries = [
            {"filename": path.name, "sha256": sha256(path), "size": path.stat().st_size}
            for path in paths
        ]
        manifest = {
            "release_tag": RELEASE_TAG,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "modal_region": os.environ.get("MODAL_REGION"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "python": platform.python_version(),
            "sageattention_commit": SAGEATTENTION_COMMIT,
            "wheels": manifest_entries,
        }
        (dist_dir / "MODAL-BUILD.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        staging_dir.mkdir(parents=True, exist_ok=False)
        for artifact in dist_dir.iterdir():
            if artifact.is_file():
                shutil.copy2(artifact, staging_dir / artifact.name)
        wheel_volume.commit()
        print(f"Persisted wheel checkpoint before runtime validation: {staging_dir}")
        paths = wheel_paths(staging_dir)

    validate_native_imports(paths)
    manifest["validated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["validation"] = {"metadata": "passed", "cuda_architectures": "passed", "native_imports": "passed"}
    (staging_dir / "MODAL-BUILD.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    wheel_volume.commit()

    release_dir.mkdir(parents=True, exist_ok=False)
    for artifact in staging_dir.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, release_dir / artifact.name)
    wheel_volume.commit()

    return {
        "volume": VOLUME_NAME,
        "path": f"releases/{RELEASE_TAG}",
        "artifacts": sorted(path.name for path in release_dir.iterdir()),
        "wheels": manifest_entries,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(build_and_store.remote(), indent=2, sort_keys=True))
