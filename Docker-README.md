# Wan2GP Docker Setup

This repository includes automated Docker image building and deployment for RunPod.

## Quick Start (RunPod)

Use our pre-built Docker image directly on RunPod:

**Container Image**: `ghcr.io/square-zero-labs/wan2gp:latest`

The image will automatically:

- Set up the complete Wan2GP environment
- Install all dependencies (PyTorch, CUDA, FFmpeg, etc.)
- Start the Gradio interface on port 7860
- Start Jupyter Lab on port 8888 (get token with `ps aux | grep jupyter`)
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
docker run --gpus all -p 7860:7860 -p 8888:8888 wan2gp:local

# Run without GPU (CPU only)
docker run -p 7860:7860 -p 8888:8888 wan2gp:local
```

## RunPod Template Configuration

### Custom Template Settings

Create a new RunPod template with these settings:

- **Template Name**: Wan2GP
- **Container Image**: `ghcr.io/square-zero-labs/wan2gp:latest`
- **Container Disk**: 50 GB (for OS and applications)
- **Volume Storage**: 75 GB minimum (for models and outputs)
- **Expose HTTP Ports**: `7860,8888`
- **Volume Mount Path**: `/workspace` (recommended for persistence)

### What You Get

- **Wan2GP Application**: Available on port 7860
- **Jupyter Lab**: Available on port 8888 (get token with `ps aux | grep jupyter`)
- **SSH Access**: Standard RunPod SSH functionality
- **File Persistence**: Files saved to `/workspace` persist across restarts

### Environment Variables (Optional)

You can customize the deployment with these environment variables:

- `SERVER_NAME`: Default is `0.0.0.0`
- `SERVER_PORT`: Default is `7860`

## Checking Logs and Status

### View Wan2GP Application Logs

```bash
# Tail the main application logs
tail -f /workspace/wan2gp.log

# View all logs at once
cat /workspace/wan2gp.log
```

### Check Running Services

```bash
# Check if Wan2GP is running
ps aux | grep wgp.py

# Check if Jupyter Lab is running
ps aux | grep jupyter

# Check all Python processes
ps aux | grep python
```

### Debug Startup Issues

```bash
# Check container startup messages
docker logs <container_id>

# Or within the container, check the application directory
ls -la /workspace/Wan2GP/

# Check if application files were restored properly
ls -la /opt/wan2gp_source/
```

### Network and Ports

```bash
# Check what's listening on our ports
netstat -tlnp | grep :7860
netstat -tlnp | grep :8888

# Test if services are responding
curl -s http://localhost:7860 | head
curl -s http://localhost:8888 | head
```

## Image Details

### Base Image

- **RunPod PyTorch Base**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **PyTorch**: 2.8.0 with CUDA 12.8.1 and Python 3.11
- **OS**: Ubuntu 22.04 LTS with full RunPod infrastructure compatibility
- **Includes**: Jupyter Lab, code-server, runpodctl, and other RunPod tools

### Included Software

- **Python**: 3.11 with pip
- **PyTorch**: 2.8.0 with CUDA 12.8.1 support (pre-installed, not reinstalled)
- **FFmpeg**: For video processing
- **tmux**: For session management
- **rsync**: For reliable file operations
- **build-essential**: For compiling native extensions
- All Python dependencies from requirements.txt
- **Gradio**: 5.35.0 (upgraded from requirements.txt version)
- **SageAttention**: 1.0.6

### Working Directory Structure

- `/workspace/Wan2GP/` - Main application directory (persistent if volume mounted)
- `/opt/wan2gp_source/` - Backup copy of application (built into image)
- `/workspace/wan2gp.log` - Application logs
- Ports 7860 (Gradio) and 8888 (Jupyter Lab) exposed

### Startup Process

1. **File Restoration**: If `/workspace/Wan2GP/wgp.py` doesn't exist, copies from `/opt/wan2gp_source/`
2. **Wan2GP Launch**: Starts `python3 wgp.py --server-name 0.0.0.0` in background
3. **RunPod Services**: Chains to `/start.sh` to start Jupyter Lab and other RunPod services
4. **Result**: Both services running simultaneously (Jupyter with auto-generated token)

## Automatic Builds

The GitHub Actions workflow (`.github/workflows/docker-build.yml`) automatically:

1. **Triggers on**:

   - Push to `main` branch
   - Push to `docker` branch
   - Git tags starting with `v`

2. **Build Process**:

   - Checks out the repository
   - Frees up runner disk space (prevents build failures)
   - Sets up Docker Buildx with container driver
   - Logs into GitHub Container Registry
   - Builds the Docker image for linux/amd64
   - Pushes to GHCR
   - Uses GitHub Actions cache for faster builds

3. **Registry**: GitHub Container Registry (ghcr.io)

4. **Space Optimization**:
   - Comments out torch/torchvision in requirements.txt (uses pre-installed versions)
   - Aggressive cleanup during GitHub Actions build process

## Customization

### Modifying the Dockerfile

The `Dockerfile` includes:

- System dependencies installation
- Repository cloning at specific commit (`597d26b7e0e53550f57a9973c5d6a1937b2e1a7b`)
- Python package installation with torch/torchvision skip
- Startup script setup and permissions
- Port exposure

To customize:

1. Edit the `Dockerfile`
2. Commit and push to `docker` branch for testing
3. Merge to `main` for production
4. New image will be available at `ghcr.io/square-zero-labs/wan2gp:latest`

### Using Different Base Images

To use a different base image (e.g., different CUDA version):

1. Modify the `FROM` line in `Dockerfile`
2. Update the `sed` command if torch/torchvision versions change
3. Update package versions in requirements.txt as needed
4. Test locally before pushing

### Modifying Startup Behavior

Edit `start-wan2gp.sh` to customize:

- Application startup parameters
- Environment setup
- Service integration
- Logging configuration

## Troubleshooting

### Build Issues

**Disk Space Errors**:

- GitHub Actions includes disk cleanup step
- Locally: `docker system prune -af` to free space

**Package Installation Failures**:

- Check that torch/torchvision are properly commented out in requirements.txt
- Verify base image compatibility

**Git Clone Issues**:

- Ensure the commit hash in Dockerfile exists
- Check repository access permissions

### Runtime Issues

**Application Won't Start**:

```bash
# Check application logs
cat /workspace/wan2gp.log

# Check if files were restored
ls -la /workspace/Wan2GP/

# Manually test the application
cd /workspace/Wan2GP && python3 wgp.py --help
```

**Port Access Issues**:

```bash
# Check if ports are exposed
docker port <container_id>

# Check if services are binding correctly
netstat -tlnp | grep -E ':(7860|8888)'
```

**Jupyter Lab Access**:

```bash
# Get the Jupyter Lab token from the running process
ps aux | grep jupyter

# Look for --ServerApp.token=XXXXXX in the output
# Use that token to log into Jupyter Lab on port 8888
# Example: if you see --ServerApp.token=oj77savqc51ysev68yfq
# then use "oj77savqc51ysev68yfq" as your login token
```

**Volume Mount Issues**:

- Ensure RunPod template has `/workspace` volume mount configured
- Check that files persist after container restart

### Registry Access

- Image is public and should not require authentication
- For private repositories, ensure proper GitHub permissions
- Use `docker pull ghcr.io/square-zero-labs/wan2gp:latest` to test access

### Performance Issues

**GPU Not Detected**:

```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.device_count())"
```

**Memory Issues**:

```bash
# Check system resources
free -h
nvidia-smi
```

## Development Workflow

1. **Local Development**:

   ```bash
   # Test changes locally
   docker build -t wan2gp:test .
   docker run --rm -p 7860:7860 -p 8888:8888 wan2gp:test
   ```

2. **Testing**:

   - Push to `docker` branch for CI testing
   - Check GitHub Actions logs for build success
   - Test deployed image on RunPod

3. **Production Release**:

   - Merge to `main` for production release
   - Tag with `v*` for versioned releases
   - Monitor logs for any deployment issues

4. **Debugging Builds**:

   ```bash
   # Check GitHub Actions logs
   # View build process in repository Actions tab

   # Test locally if build fails
   docker build --progress=plain -t debug .
   ```

## Security

- **Images**: Built from source code in this repository
- **Base Images**: Official RunPod/NVIDIA CUDA images
- **Secrets**: No secrets or credentials included in images
- **Permissions**: GitHub Actions uses minimal required permissions
- **Network**: Only necessary ports (7860, 8888) exposed

## Support

For issues related to:

- **Docker Setup**: Check this README and GitHub Issues
- **Wan2GP Application**: See main project documentation
- **RunPod Platform**: Contact RunPod support
- **Build Failures**: Check GitHub Actions logs and this troubleshooting section

## Log Files

All service logs are saved to the persistent `/workspace` directory:

- **Wan2GP Application**: `/workspace/wan2gp.log`
- **Jupyter Lab**: Check with `ps aux | grep jupyter` for token
- **Container Startup**: Visible in Docker logs

### Viewing Logs

```bash
# View Wan2GP logs
tail -f /workspace/wan2gp.log

# View last 100 lines
tail -n 100 /workspace/wan2gp.log

# Monitor for errors
grep -i error /workspace/wan2gp.log
```
