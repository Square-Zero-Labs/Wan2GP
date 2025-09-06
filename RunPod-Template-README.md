# Wan2GP - AI Video Generation Template

#### Last Updated on 9/6/2025 to v8.4 - Take me to Outer Space

### _This template has been tested with an A40_

- To run on RTX 5090, you must deactivate SageAttention. See instructions below.

## What is Wan2GP?

WAN2GP (aka "Wan for the GPU Poor") is a free, open-source tool that lets you generate videos using advanced AI models—even on modest GPUs.

## What This Template Provides

This RunPod template is an extenstion of the official Runpod Pytorch 2.8.0 template. It gives you a fully configured environment with:

- ✅ **Wan2GP Application** - Ready to use on port 7862 (password protected)
- ✅ **Jupyter Lab** - Development environment on port 8888
- ✅ **All Dependencies** - PyTorch, FFmpeg, and required python libraries pre-installed
- ✅ **Storage** - Your models and outputs saved to `/workspace`

## Quick Start

### 1. Launch Your Pod/Selecting GPU

- **CUDA Version**: Make sure you use a machine that has **CUDA 12.8** installed (use additional filters when selecting machine)
- **Recommended GPU**: This template was tested with an A40

### 2. Wait for Startup (Important!)

- ⏱️ **The application takes a few minutes to fully start** after your pod boots. This is normal.

### 3. Access Your Applications

#### Wan2GP Interface

1. Wait for startup to complete
2. Connect to port **7862** (authenticated proxy)
3. **Login when prompted:**
   - Username: `admin`
   - Password: `gpuPoor2025`
4. Start generating videos! Note that on the first run of a model (when you hit "generate"), the model is downloaded which can take a few additional minutes. The next time you generate with the same model, the model is already there and the generation can start right away.

#### Custom Authentication (Optional)

To use your own login credentials, set environment variables in your RunPod template:

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
3. Look for the token in the output
4. Use just the token to log into Jupyter Lab

## Troubleshooting

### Can't Login

- Try the default credentials: `admin` / `gpuPoor2025`
- Check if you set custom `WAN2GP_PASSWORD` environment variable
- Wait 30 seconds after pod start for nginx to initialize

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

### Running on RTX 5090

You must deactivate SageAttention before generating videos.

1. Open the Wan2GP UI and go to the Configuration tab.
2. Find the Attention Type setting.
3. From its dropdown menu, choose Scale Dot Product Attention.
4. Click Apply Changes at the bottom.

## ⚠️ Advanced: Live-Updating the Application ⚠️

If you want to update the Wan2GP application to the latest version of Wan2GP without waiting for a new version of the template, you can use the built-in update script.

> 🛑 **DANGER: This is an advanced feature and can break your pod.**
>
> - **No Automatic Rollback:** If the update fails or introduces bugs, there is no automatic way to go back. Your pod may become unusable.
> - **Untested Code:** You are pulling the latest code from the Wan2GP repository, which has not been tested in this specific RunPod environment. It may have new dependencies or bugs that cause the application to fail.
> - **Restarting Pod:** If you stop and restart your pod after updating, you should run the update script again on restart. Otherwise, you could have mismatched python dependency versions.

To run the live update:

1.  Connect to your pod using the web terminal or SSH.
2.  Run the following command:
    ```bash
    update-wan2gp.sh
    ```
3.  The script will stop the application, download the latest code, update dependencies, and restart the service. This may take a minute.
4.  You can monitor the progress in the terminal and check the application log once it's complete: `tail -f /workspace/wan2gp.log`

## Walkthroughs and Tutorials

See [Fuzz Puppy on YouTube](https://www.youtube.com/@fuzz_puppy)

---

**You can review the complete code for this template at [Template Code](https://github.com/Square-Zero-Labs/Wan2GP/tree/docker). No hidden Dockerfiles!**

---

**🎬 Go forth and create amazing videos. Just wait for startup and connect to port 7862!**
