# SageAttention and SpargeAttention wheels

The production container consumes two project-owned release artifacts because no public publisher currently offers a provenance-complete Linux CPython 3.11 pair for PyTorch 2.10/CUDA 12.8 and both A40 and RTX 5090.

Release repository: `Square-Zero-Labs/Wan2GP-Runpod-Wheels`

GitHub release tag: `8-8-2026`

Modal build/checkpoint tag: `pytorch2.10.0-cu128-py311-sm86-sm120-v1`

Pinned source revisions:

- SageAttention: `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`
- SpargeAttention fork: `067d80cb6b76345c7b8be40e86c7d19a3cf7c4eb`

## Build and retrieve with Modal

With `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` exported, run:

```bash
modal run Runpod-Docker/modal-build-attention-wheels.py
```

The Modal app uses the CUDA 12.8.1 development image and a non-preemptible,
CPU-only x86_64 builder. After wheel metadata and embedded CUDA architectures
pass inspection, it checkpoints the artifacts to the `wan2gp-runpod-wheels`
Volume under `staging/<release-tag>`. Native import validation then runs from
that checkpoint. Only passing artifacts are copied to the immutable
`releases/<release-tag>` path, and a retry reuses the checkpoint instead of
recompiling.

List the validated release, then download each artifact explicitly. The Modal
CLI can treat this Volume prefix as one file and discard its filename when the
whole release path is passed to `volume get`, so do not use the directory form
for this release.

```bash
release_tag='pytorch2.10.0-cu128-py311-sm86-sm120-v1'
release_path="releases/$release_tag"

modal volume ls wan2gp-runpod-wheels "$release_path"
mkdir -p wheel-dist

for artifact in \
  sageattention-2.2.0-cp311-cp311-linux_x86_64.whl \
  spas_sage_attn-0.1.0-cp311-cp311-linux_x86_64.whl \
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

The Volume release includes `MODAL-BUILD.json`, with pinned revisions, hashes,
sizes, and validation status, in addition to the normal release assets.

## Build on Linux x86_64

Use an x86_64 runner; CUDA compilation under ARM emulation is not a release build. The build does not require an attached GPU because all target architectures are explicit.

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
- `spas_sage_attn-0.1.0-cp311-cp311-linux_x86_64.whl`
- `SHA256SUMS.txt`
- `BUILD-INFO.txt`
- `MODAL-BUILD.json` (Modal builds only)

The build script validates metadata with `validate-attention-wheels.py`, inspects the native objects with `cuobjdump`, and imports all required extension modules before returning success. Sage must contain `sm86` and `sm120` code; Sparge deliberately uses its `sm80` kernel for the A40 path and must also contain `sm120`. Review `BUILD-INFO.txt` and `SHA256SUMS.txt` before publishing.

## Published release

The verified artifacts are published as the `8-8-2026` pre-release. Upload the
missing `BUILD-INFO.txt` file so the published assets match the release notes:

```bash
gh release upload 8-8-2026 wheel-dist/BUILD-INFO.txt \
  --repo Square-Zero-Labs/Wan2GP-Runpod-Wheels \
  --clobber
```

Published wheel hashes:

```text
9fa280f209779caefecf70dcbc368eb1514348e89cd6fcc3e92d50e37965ed65  sageattention-2.2.0-cp311-cp311-linux_x86_64.whl
45121bcd834eeabc58032287e60ef48837ec9afebe39940e35796bbb9ec645a5  spas_sage_attn-0.1.0-cp311-cp311-linux_x86_64.whl
```

Optional Wan2GP repository-variable overrides can be set with:

```bash
sage_sha="$(awk '/sageattention-/{print $1}' wheel-dist/SHA256SUMS.txt)"
sparge_sha="$(awk '/spas_sage_attn-/{print $1}' wheel-dist/SHA256SUMS.txt)"

gh variable set SAGEATTENTION_WHEEL_SHA256 \
  --body "$sage_sha" \
  --repo Square-Zero-Labs/Wan2GP
gh variable set SPARGEATTN_WHEEL_SHA256 \
  --body "$sparge_sha" \
  --repo Square-Zero-Labs/Wan2GP
```

The Docker build downloads from the pinned release tag, verifies both hashes,
installs with `--no-deps`, and checks exact package versions.

## GPU qualification

After the `:overhaul-docker` image is published, run this on one A40 and one RTX 5090:

```bash
python /opt/wan2gp-container/validate-gpu.py
```

Then run one small Sage2 generation and one short FlashVSR upscale. Do not overwrite the wheel release or promote the container tag if either GPU fails; publish a new `-v2` release instead.

## Public-wheel search note

The broad search conducted on 2026-08-09 found a CPython 3.11 Linux Sage wheel that imports against PyTorch 2.10/CUDA 12.8, but its publisher supplied no source revision or build metadata. No matching public Linux Sparge wheel was found. Details and the diagnostic wheel checksum are retained in `notes/version-update-plan.log`; that artifact is not part of the production supply chain.
