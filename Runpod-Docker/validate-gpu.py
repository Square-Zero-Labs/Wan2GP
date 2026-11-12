#!/usr/bin/env python3
"""Exercise the actual Wan2GP SageAttention path on a running pod."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


app_dir = Path(os.environ.get("WAN2GP_APP_DIR", "/workspace/Wan2GP"))
if not (app_dir / "shared").is_dir():
    image_source_dir = Path("/opt/wan2gp_source")
    if (image_source_dir / "shared").is_dir():
        app_dir = image_source_dir
    else:
        raise SystemExit(
            "Wan2GP shared modules were not found under "
            f"{app_dir} or {image_source_dir}"
        )
sys.path.insert(0, str(app_dir))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

device = torch.device("cuda")
capability = torch.cuda.get_device_capability(device)
if capability not in {(8, 6), (12, 0)}:
    raise SystemExit(f"unsupported validation GPU capability: {capability}")

torch.manual_seed(7)
dtype = torch.bfloat16
q, k, v = [torch.randn((1, 256, 4, 64), device=device, dtype=dtype) for _ in range(3)]

from shared.sage2_core import sageattn  # noqa: E402

sage_output = sageattn([q.clone(), k.clone(), v.clone()], tensor_layout="NHD")
reference = F.scaled_dot_product_attention(
    q.transpose(1, 2),
    k.transpose(1, 2),
    v.transpose(1, 2),
).transpose(1, 2)
if not torch.isfinite(sage_output).all():
    raise RuntimeError("SageAttention returned non-finite values")
torch.testing.assert_close(sage_output.float(), reference.float(), rtol=0.25, atol=0.25)

print(f"GPU: {torch.cuda.get_device_name(device)} (sm{capability[0]}{capability[1]})")
print(f"Sage2 max absolute error vs SDPA: {(sage_output.float() - reference.float()).abs().max().item():.6f}")
print("A40/RTX 5090 SageAttention validation passed")
