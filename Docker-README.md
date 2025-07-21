# Wan2GP Docker Setup

This repository includes automated Docker image building and deployment for RunPod.

## Quick Start (RunPod)

Use our pre-built Docker image directly on RunPod:

**Container Image**: `ghcr.io/square-zero-labs/wan2gp:latest`

The image will automatically:

- Set up the complete Wan2GP environment
- Install all dependencies (PyTorch, CUDA, FFmpeg, etc.)
- Start the Gradio interface on port 7860
- Use `/workspace` as the working directory

## GitHub Container Registry

The Docker image is automatically built and pushed to GitHub Container Registry (GHCR) when code is pushed to the `main` or `docker` branches.

### Available Tags

- `latest` - Latest build from main branch
- `docker` - Latest build from docker branch
- `v*` - Specific version tags
- `main` - Latest from main branch

## Building Locally

### Prerequisites

- Docker installed
- Git

### Build Command

```bash
# Clone the repository
git clone https://github.com/Square-Zero-Labs/Wan2GP.git
cd Wan2GP

# Build the Docker image
docker build -t wan2gp:local .
```

### Running Locally

```bash
# Run with GPU support (requires nvidia-docker)
docker run --gpus all -p 7860:7860 wan2gp:local

# Run without GPU (CPU only)
docker run -p 7860:7860 wan2gp:local
```

## RunPod Template Configuration

### Custom Template Settings

Create a new RunPod template with these settings:

- **Template Name**: Wan2GP
- **Container Image**: `ghcr.io/square-zero-labs/wan2gp:latest`
- **Container Disk**: 60 GB
- **Expose HTTP Ports**: 7860
- **Volume Mount Path**: `/workspace` (optional)

### Environment Variables (Optional)

You can customize the deployment with these environment variables:

- `SERVER_NAME`: Default is `0.0.0.0`
- `SERVER_PORT`: Default is `7860`

## Image Details

### Base Image

- RunPod PyTorch Base: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- PyTorch 2.8.0 with CUDA 12.8.1 and Python 3.11
- Ubuntu 22.04 LTS with full RunPod infrastructure compatibility
- Includes Jupyter Lab, code-server, runpodctl, and other RunPod tools

### Included Software

- Python 3.11 with pip
- PyTorch 2.8.0 with CUDA 12.8.1 support
- FFmpeg for video processing
- tmux for session management
- All Python dependencies from requirements.txt
- Gradio 5.35.0
- SageAttention 1.0.6

### Working Directory

- `/workspace` - Main application directory
- Port 7860 exposed for Gradio interface

## Automatic Builds

The GitHub Actions workflow (`.github/workflows/docker-build.yml`) automatically:

1. **Triggers on**:

   - Push to `main` branch
   - Push to `docker` branch
   - Git tags starting with `v`
   - Pull requests to `main`

2. **Build Process**:

   - Checks out the repository
   - Sets up Docker Buildx
   - Logs into GitHub Container Registry
   - Builds the Docker image for linux/amd64
   - Pushes to GHCR (except for PRs)
   - Uses GitHub Actions cache for faster builds

3. **Registry**: GitHub Container Registry (ghcr.io)

## Customization

### Modifying the Dockerfile

The `Dockerfile` includes:

- System dependencies installation
- Repository cloning at specific commit
- Python package installation
- Port exposure and startup command

To customize:

1. Edit the `Dockerfile`
2. Commit and push to trigger automatic rebuild
3. New image will be available at `ghcr.io/square-zero-labs/wan2gp:latest`

### Using Different Base Images

To use a different base image (e.g., different CUDA version):

1. Modify the `FROM` line in `Dockerfile`
2. Update package versions as needed
3. Test locally before pushing

## Troubleshooting

### Build Issues

- Check that all files referenced in `COPY` commands exist
- Verify requirements.txt is compatible with the base image
- Check Docker build logs in GitHub Actions

### Runtime Issues

- Ensure GPU drivers are compatible on RunPod
- Check that port 7860 is properly exposed
- Verify volume mounts if using persistent storage

### Registry Access

- Image is public and should not require authentication
- For private repositories, ensure proper GitHub permissions

## Development Workflow

1. Make changes to code or Dockerfile
2. Test locally: `docker build -t test .`
3. Push to `docker` branch for testing
4. Merge to `main` for production release
5. Tag with `v*` for versioned releases

## Security

- Images are built from source code in this repository
- Base images are official NVIDIA CUDA images
- No secrets or credentials are included in images
- GitHub Actions uses minimal required permissions
