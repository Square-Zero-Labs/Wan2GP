# Wan2GP - AI Video Generation Template

## What is Wan2GP?

WAN2GP (aka "Wan 2.1 for the GPU Poor") is a free, open-source tool that lets you generate videos using advanced AI models—even on modest GPUs. Wan2GP supports multiple state-of-the-art models including:

- **Text-to-Video**: Create videos from text prompts
- **Image-to-Video**: Animate static images
- **Multitalk**: Animate multiple characters in the same scene with different voices

## What This Template Provides

This RunPod template is an extenstion of the official Runpod Pytorch 2.8.0 template. It gives you a fully configured environment with:

- ✅ **Wan2GP Application** - Ready to use on port 7860
- ✅ **Jupyter Lab** - Development environment on port 8888
- ✅ **All Dependencies** - PyTorch, FFmpeg, and AI libraries pre-installed
- ✅ **Persistent Storage** - Your models and outputs saved to `/workspace`

## Quick Start

### 1. Launch Your Pod

- **CUDA Version**: Make sure you use a machine that has CUDA 12.8 installed (use additional filters when selecting machine)
- **Recommended**: This template was tested with an A40

### 2. Wait for Startup (Important!)

- ⏱️ **The application takes a few minutes to fully start** after your pod boots. This is normal.

### 3. Access Your Applications

#### Wan2GP Interface

1. Wait for startup to complete
2. Connect to port **7860**
3. Start generating videos!

#### Jupyter Lab (Optional)

1. Connect to port **8888**
2. Get the access token: Connect via SSH or the web terminal and run:
   ```bash
   ps aux | grep jupyter
   ```
3. Look for `--ServerApp.token=XXXXXX` in the output
4. Use that token to log into Jupyter Lab

## First Run

- Models download automatically on first use
- Models are cached in `/workspace` for future use

## Troubleshooting

### Application Not Loading?

```bash
# Check if services are running
tail -f /workspace/wan2gp.log

# Restart if needed (rare)
cd /workspace/Wan2GP
python3 wgp.py --server-name 0.0.0.0
```

### Out of Space?

- Increase your **Volume Storage** (not Container Disk)
- Clean up old outputs in `/workspace/outputs`
- Remove unused models from `/workspace/models`
- Remove old logs at `/worskpace/wan2gp.log`

## Support

- **Template Issues**: [GitHub Issues](https://github.com/Square-Zero-Labs/Wan2GP/issues)
- **Model Questions**: Check the Wan2GP documentation
- **RunPod Platform**: RunPod support

---

**You can review the complete code for this template at [Template Code](https://github.com/Square-Zero-Labs/Wan2GP/tree/docker). No hidden Dockerfiles!**

---

**🎬 Go forth and create amazing videos. Just wait for startup and connect to port 7860!**
