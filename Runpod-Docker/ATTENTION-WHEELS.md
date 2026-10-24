# SageAttention wheel

The production container consumes one project-owned release artifact because no public publisher currently offers a provenance-complete Linux CPython 3.11 SageAttention wheel for PyTorch 2.10/CUDA 12.8 with both A40 and RTX 5090 targets.

Release repository: `Square-Zero-Labs/Wan2GP-Runpod-Wheels`

GitHub release tag: `8-8-2026`

Modal build/checkpoint tag: `pytorch2.10.0-cu128-py311-sm86-sm120-sage-v1`

Pinned SageAttention revision: `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`

## Build and retrieve with Modal

With `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` exported, run:

```bash
modal run Runpod-Docker/modal-build-attention-wheels.py
```

The Modal app uses the CUDA 12.8.1 development image and a non-preemptible, CPU-only x86_64 builder. It checkpoints validated artifacts in the `wan2gp-runpod-wheels` Volume before copying them to the immutable release directory.

Download each artifact explicitly because the Modal CLI can treat a Volume prefix as a single file when given a directory path:

```bash
release_tag='pytorch2.10.0-cu128-py311-sm86-sm120-sage-v1'
release_path="releases/$release_tag"

modal volume ls wan2gp-runpod-wheels "$release_path"
mkdir -p wheel-dist

for artifact in \
  sageattention-2.2.0-cp311-cp311-linux_x86_64.whl \
  SHA256SUMS.txt \
  BUILD-INFO.txt \
  MODAL-BUILD.json
do
  modal volume get wan2gp-runpod-wheels \
    "$release_path/$artifact" \
    "wheel-dist/$artifact"
done

(cd wheel-dist && sha256sum --check --strict SHA256SUMS.txt)
```

## Build on Linux x86_64

Use an x86_64 runner. The build does not require an attached GPU because the target architectures are explicit.

```bash
mkdir -p wheel-dist

docker run --rm --platform linux/amd64 \
  --volume "$PWD/Runpod-Docker:/repo/Runpod-Docker:ro" \
  --volume "$PWD/wheel-dist:/workspace/wan2gp-attention-wheels" \
  runpod/base:1.1.0-cuda1281-ubuntu2404 \
  bash -lc '
    python3.11 -m venv /tmp/attention-venv
    . /tmp/attention-venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    PYTHON_BIN=python /repo/Runpod-Docker/build-attention-wheels.sh
  '
```

Expected assets:

- `sageattention-2.2.0-cp311-cp311-linux_x86_64.whl`
- `SHA256SUMS.txt`
- `BUILD-INFO.txt`
- `MODAL-BUILD.json` for Modal builds

The build validates wheel metadata, inspects native objects with `cuobjdump`, and imports the required extension modules. The wheel must contain both `sm86` and `sm120` code.

## Published release

The verified SageAttention artifact is available in the `8-8-2026` pre-release:

```text
9fa280f209779caefecf70dcbc368eb1514348e89cd6fcc3e92d50e37965ed65  sageattention-2.2.0-cp311-cp311-linux_x86_64.whl
```

An optional Wan2GP repository-variable override can be set with:

```bash
sage_sha="$(awk '/sageattention-/{print $1}' wheel-dist/SHA256SUMS.txt)"

gh variable set SAGEATTENTION_WHEEL_SHA256 \
  --body "$sage_sha" \
  --repo Square-Zero-Labs/Wan2GP
```

The Docker build downloads the wheel from the pinned release tag, verifies its hash, installs it with `--no-deps`, and checks the exact package version.

## GPU qualification

After the candidate image is published, run this on one A40 and one RTX 5090:

```bash
python /opt/wan2gp-container/validate-gpu.py
```

Then run one small Sage2 generation and one short FlashVSR upscale with its bundled Triton sparse-attention backend. Do not promote the container tag if either GPU fails.
