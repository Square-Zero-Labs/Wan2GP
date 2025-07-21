## Docker Deployment (Automated)

### Using Pre-built Docker Image from GitHub Container Registry

The easiest way to deploy Wan2GP on RunPod is using our pre-built Docker image:

**Container Image**: `ghcr.io/square-zero-labs/wan2gp:latest`

#### RunPod Template Configuration:

- **Container Image**: `ghcr.io/square-zero-labs/wan2gp:latest`
- **Container Disk**: 60 GB
- **Expose Port**: 7860
- **Volume Mount**: `/workspace` (optional, for persistent storage)

#### Deploy Steps:

1. Choose GPU (move slider to desired number of GPUs)
2. Select On-Demand or Spot
3. The container will automatically start Wan2GP on port 7860

No manual setup required! The Docker image includes all dependencies and automatically runs the application.

> **Note**: This image uses RunPod's official PyTorch 2.8.0 base with CUDA 12.8.1 support and Python 3.11, providing the latest PyTorch features and excellent GPU performance.

---

## Manual Setup (Original Method)

### Select Template

RunPod PyTorch 2.8.0
Overrides:
Increase Container disk to 60 GB
Expose port 7860

### Deploy

Move GPU slider to one GPU
Choose On-Demand or Spot

### On Machine

```bash
cd workspace
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
git checkout 597d26b7e0e53550f57a9973c5d6a1937b2e1a7b
pip install -r requirements.txt
apt-get update && apt-get install -y ffmpeg
pip install gradio==5.35.0
pip install sageattention==1.0.6
apt update && apt install -y tmux
tmux new -s wan
python wgp.py --server-name 0.0.0.0
```
