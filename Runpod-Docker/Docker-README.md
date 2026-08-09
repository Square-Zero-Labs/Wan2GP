# Wan2GP RunPod image

This image runs Wan2GP on both NVIDIA A40 (`sm86`) and RTX 5090 (`sm120`) with a single CUDA 12.8 stack. It preserves RunPod SSH, web terminal, nginx, and Jupyter services while keeping build tools and nvcc out of the runtime image.

## Runtime stack

- Ubuntu 24.04 RunPod service base
- Python 3.11
- PyTorch 2.10.0 + CUDA 12.8
- torchvision 0.25.0, torchaudio 2.10.0, TorchCodec 0.10.0
- Triton 3.6.0
- ONNX Runtime GPU 1.26.0
- Gradio 5.35.0
- Decord2 3.4.0 (the maintained `decord` API for Python 3.11)
- SageAttention 2.2.0 and SpargeAttention 0.1.0, installed from checksum-verified project wheels
- Ubuntu FFmpeg and Jupyter Lab

The base image intentionally omits the larger optional native-kernel stack: FlashAttention, Nunchaku, LightX2V, and precompiled GGUF kernels. Wan2GP's ordinary features and its Python GGUF support remain installed.

## Image tags

- `ghcr.io/square-zero-labs/wan2gp:overhaul-docker` — modernization candidate
- `ghcr.io/square-zero-labs/wan2gp:docker` — deployment branch after acceptance
- `ghcr.io/square-zero-labs/wan2gp:sha-<commit>` — immutable rollback image

Do not promote the candidate until the A40 and RTX 5090 acceptance tests pass.

## RunPod configuration

- Container disk: 50 GB or more
- Persistent volume: 75 GB or more
- Volume mount: `/workspace`
- HTTP ports: `7862,8888`
- Host driver: R570 or newer. CUDA 12.8 does not require an R580 driver.

Wan2GP listens only on `127.0.0.1:7860`. Nginx exposes it with Basic Auth on port 7862; do not expose 7860 directly.

Default login:

- Username: `admin`
- Password: `gpuPoor2025`

Set `WAN2GP_USERNAME` and `WAN2GP_PASSWORD` in the RunPod template to override both values.

Jupyter listens on port 8888. The container generates a random token unless `JUPYTER_PASSWORD` is set. Retrieve the active token from a web terminal or SSH session:

```bash
jupyter server list
```

The default Jupyter interpreter and the `Wan2GP (Python 3.11)` kernel use the same environment as Wan2GP.

## Operations

Application and Supervisor logs are persistent:

```bash
tail -f /workspace/wan2gp.log
supervisorctl -c /etc/supervisor/wan2gp.conf status
```

Restart only the Wan2GP process and wait for its health check:

```bash
restart-wan2gp.sh
```

Apply a compatible upstream live update:

```bash
update-wan2gp.sh
```

The updater stashes tracked local edits, fast-forwards upstream `main`, filters image-owned dependencies, validates the environment, and then restarts Wan2GP. On failure it restores the previous source commit and dependency snapshot. Untracked models, outputs, and configuration are not touched. Its compatible dependency manifest is persisted in `/workspace/.wan2gp-state` and reconciled after pod recreation.

Core Torch, CUDA, Triton, ONNX, Sage, and Sparge versions never change during a live update; updating those requires a new container image.

## Build

The default build is pinned to the verified `8-8-2026` attention-wheel release
and its two SHA-256 values:

```bash
docker build \
  --platform linux/amd64 \
  --file Runpod-Docker/Dockerfile \
  --tag wan2gp:local \
  .
```

When advancing to a different wheel release, override the release tag, URLs, and
hashes together. The GitHub workflow accepts optional repository variables named
`SAGEATTENTION_WHEEL_SHA256` and `SPARGEATTN_WHEEL_SHA256`; otherwise it uses the
verified release hashes committed in the workflow.

## Acceptance tests

Run these on both an A40 and RTX 5090 candidate pod:

```bash
python /opt/wan2gp-container/validate-runtime.py
python /opt/wan2gp-container/validate-gpu.py
curl -I http://127.0.0.1:7862/
curl -u admin:gpuPoor2025 -I http://127.0.0.1:7862/
jupyter server list
restart-wan2gp.sh
```

The unauthenticated request must return `401`; the authenticated request must reach Gradio. Also run one small Sage2 generation and a short FlashVSR/Sparge upscale on each GPU before merging `overhaul-docker` into `docker`.

## Persistent paths

- `/workspace/Wan2GP` — live application checkout
- `/workspace/.wan2gp-state` — live-update state and history
- `/workspace/wan2gp.log` — application log
- `/opt/wan2gp_source` — immutable image seed used on a fresh volume
- `/opt/wan2gp-venv` — image-owned Python environment
