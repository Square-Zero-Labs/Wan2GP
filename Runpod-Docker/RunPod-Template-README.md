# Wan2GP - AI Image and Video Generation Template

#### Last Updated on 8/9/2026 to WanGP v12.434 with PyTorch 2.10 / CUDA 12.8 support

### _This version targets both A40 and RTX 5090 GPUs_

- Python 3.11, PyTorch 2.10, and CUDA 12.8 are installed in the container.
- Prebuilt SageAttention 2.2 and SpargeAttention 0.1 wheels include the native targets needed by both GPUs.

## What is Wan2GP?

WAN2GP (aka "Wan for the GPU Poor") is a free, open-source tool that lets you generate videos using advanced AI models—even on modest GPUs.

## What This Template Provides

This template uses a lean RunPod Ubuntu 24.04 service base with a pinned CUDA 12.8 / PyTorch 2.10 environment. It gives you a fully configured environment with:

- ✅ **Wan2GP Application** - Ready to use on port 7862 (password protected)
- ✅ **Jupyter Lab** - Development environment on port 8888
- ✅ **All Dependencies** - PyTorch, Triton, FFmpeg, ONNX Runtime GPU, SageAttention, SpargeAttention, and required python libraries pre-installed
- ✅ **Storage** - Your models and outputs saved to `/workspace`

## Quick Start

### 1. Launch Your Pod/Selecting GPU

- **NVIDIA Driver**: Select a machine with an **R570 or newer** NVIDIA driver. CUDA 12.8 is supplied by the container, so the host does not need a separate CUDA 12.8 toolkit.
- **Supported GPUs**: A40 and RTX 5090

### 2. Wait for Startup (Important!)

- ⏱️ **The application takes a few minutes to fully start**

### 3. Access Your Applications

#### Wan2GP Interface

1. Wait for startup to complete
2. Connect to port **7862** (authenticated proxy)
3. **Login when prompted:**
   - Username: `admin`
   - Password: `gpuPoor2025`
4. Start generating videos! Note that on the first run of a model (when you hit "generate"), the model is downloaded which can take a few additional minutes.

#### Custom Authentication (Optional)

To use your own login credentials, set environment variables in the template:

```
WAN2GP_USERNAME=your_username
WAN2GP_PASSWORD=your_secure_password
```

#### Jupyter Lab (Optional)

1. Connect to port **8888**
2. Get the access token: Connect via SSH or the web terminal and run:
   ```bash
   jupyter server list
   ```

## Troubleshooting

### Can't Login

- Try the default credentials: `admin` / `gpuPoor2025`
- Check if you set custom `WAN2GP_PASSWORD` environment variable
- Wait 30 seconds after pod start for nginx to initialize

### Check logs for issues

```bash
# Check if services are running
tail -f /workspace/wan2gp.log
```

### Quick Restart Command

- Restart the Wan2GP app without updating code/dependencies:
  - `restart-wan2gp.sh`

### Out of Space?

- Increase your **Volume Storage** (not Container Disk)
- Clean up old outputs in `/workspace/outputs`
- Remove unused models from `/workspace/models`
- Remove old logs at `/worskpace/wan2gp.log`

### Running on RTX 5090

SageAttention 2.2 and SpargeAttention 0.1 are installed from wheels built for the PyTorch 2.10 / CUDA 12.8 stack with RTX 5090 / Blackwell support. If an attention mode fails on a specific model, switch to Scale Dot Product Attention.

1. Open the Wan2GP UI and go to the Configuration tab.
2. Find the Attention Type setting.
3. From its dropdown menu, choose Scale Dot Product Attention.
4. Click Apply Changes at the bottom.

## ⚠️ Advanced: Live-Updating the Application ⚠️

If you want to update the Wan2GP application to the latest version of Wan2GP without waiting for a new version of the template, you can use the built-in update script.

> 🛑 **DANGER: This is an advanced feature and can break your pod.**
>
> - **Automatic Validation and Rollback:** The script validates a new version before restarting Wan2GP and restores the previous source and dependencies if the update fails. Model compatibility problems can still appear after validation.
> - **Untested Code:** You are pulling the latest code from the Wan2GP repository, which has not been tested in this specific environment. It may have new dependencies or bugs that cause the application to fail.
> - **Restarting Pod:** Compatible live-update dependencies are recorded under `/workspace` and reconciled when the pod restarts. Core CUDA, PyTorch, Triton, ONNX, and attention packages remain pinned to the container image.

To run the live update:

1.  Connect to your pod using the web terminal or SSH.
2.  Run the following command:
    ```bash
    update-wan2gp.sh
    ```
3.  The script will stop the application, download the latest code, update compatible dependencies, validate the result, and restart the service. If validation fails, it restores the previous version.
4.  You can monitor the progress in the terminal and check the application log once it's complete: `tail -f /workspace/wan2gp.log`

## Walkthroughs and Tutorials

See [Fuzz Puppy on YouTube](https://www.youtube.com/@fuzz_puppy)

---

**You can review the complete code for this template at [Template Code](https://github.com/Square-Zero-Labs/Wan2GP/tree/docker). No hidden Dockerfiles!**

---

**🎬 Go forth and create amazing videos. Just wait for startup and connect to port 7862!**
