# Wan Animate Tutorial

Use this quick reference to capture the key settings and workflow from [the companion YouTube tutorial](https://youtu.be/slNId3GoGrU).

## Summary Tips

- Keep the driving video, reference image, and output at matching dimensions.
- Use the FusionX or LightX2V LoRA for best quality and to reduce inference steps.
- LightX2V keeps facial movement more faithful to the original.
- Disable Sage Attention and switch to SDPA for better output.
- When animating with a driving clip, use the targeted person option rather than whole motion.
- For replacement videos, choose source and target characters with similar size and lighting.

## Install LoRAs via `curl`

### LightX2V

```bash
curl -L --create-dirs \
  -o /workspace/Wan2GP/loras/wan_i2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors?download=true
```

### FusionX

```bash
curl -L --create-dirs \
  -o /workspace/Wan2GP/loras/wan_i2v/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
  https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan2.1_I2V_14B_FusionX_LoRA.safetensors
```

## Prompt Reference

- Tutorial prompt: `视频中的人在做动作`
- Translation: “The person in the video is performing the action.”
- Source configuration: [Wan 2.2 repository `wan_animate_14B.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_animate_14B.py)

## Settings

- Inference Steps 10-12
- CFG 1
- Shift 7
- Enable TeaCache
- Lightx2v or FusionX LoRA at 1.0x
