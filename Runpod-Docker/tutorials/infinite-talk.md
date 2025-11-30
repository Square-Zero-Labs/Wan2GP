# InfiniteTalk Tutorial

Use this quick reference to capture the key settings and workflow from [the companion YouTube tutorial](https://youtu.be/kj9_3kGSa4w).

## Install LoRA via `curl`

### FusionX

```bash
curl -L --create-dirs \
  -o /workspace/Wan2GP/loras/wan_i2v/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
  https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

## Settings

- Inference Steps 10
- CFG 1
- Shift Scale 3
- Audio Guidance 7
- Unipc Sampler
- FusionX LoRA at 1.0x
- Enable TeaCache
