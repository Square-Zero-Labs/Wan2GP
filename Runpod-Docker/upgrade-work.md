# Wan2GP RunPod Upgrade Work (Paused)

## Context
- Goal: run Wan2GP on RunPod using `Runpod-Docker/` with GPU pose extraction.
- Constraint: avoid CPU fallback for pose extraction.
- Environment: RunPod CUDA 13 base image (`runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404`) with Python 3.11 + Torch 2.10.0 (installed in-image).

## Primary Failure
- Pose extraction failed on GPU with ONNX Runtime CUDA MatMul:
  - `CUBLAS_STATUS_INVALID_VALUE`
  - Node: `MatMul_282`
  - Repro observed around frame ~171.

## What We Tested
1. Kept GPU ORT on CUDA 13 nightly and moved pin forward:
   - from `onnxruntime-gpu==1.25.0.dev20260210001`
   - to newer `1.25.0.dev20260310001`.
2. Disabled TF32 paths in runtime attempts.
3. Tried provider adjustments (forcing CUDA EP behavior, reducing TRT interaction).
4. Tried deeper ORT rollback:
   - `1.24.0.dev20251107001` (failed to load CUDA provider on this base).

## Secondary Failure Found
- On 1.24 nightly rollback, ORT could not load CUDA provider:
  - missing `libcublasLt.so.12`
  - indicates ABI/runtime mismatch for that wheel vs current CUDA 13 base stack.
  - Effect: ORT fell back to CPU providers.

## Additional Issue Observed
- Jupyter Lab also failed to load reliably on this setup during upgrade testing.
- This should be investigated separately before resuming the CUDA 13 upgrade path.

## Current Decision
- Pause this upgrade path for now.
- Revert temporary runtime patching and script complexity.
- Keep RunPod scripts/Docker setup clean and close to repo baseline.

## Why Pausing Is Reasonable
- Current CUDA 13 + ORT nightly combinations showed instability on this workload (`MatMul_282`).
- Older ORT nightly fallback introduced provider library mismatch.
- Time spent chasing provider/runtime edge cases is high relative to value right now.

## Revisit Plan (Later)
When RunPod base and upstream compatibility improve, retry with:
1. A RunPod base that natively aligns Python/Torch/CUDA with Wan2GP needs.
2. ORT CUDA build known-good for that exact CUDA runtime.
3. Minimal matrix test:
   - fixed input clip
   - same failing frame window (~150-200)
   - 2-3 adjacent ORT builds only
   - record provider list + exact error/log.

## Key Error Signatures (for future search)
- `Non-zero status code returned while running MatMul node. Name:'MatMul_282'`
- `CUBLAS failure 7: CUBLAS_STATUS_INVALID_VALUE`
- `Failed to load ... libonnxruntime_providers_cuda.so ... libcublasLt.so.12`
