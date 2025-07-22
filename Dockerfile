# Use the official RunPod base image, which includes a pre-configured environment.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables for non-interactive installs.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1. Install system dependencies.
# We add `supervisor` to manage our custom process and `rsync` for robust file copying.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tmux \
    build-essential \
    supervisor \
    rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone the application source code to a "safe" location that won't be volume-mounted.
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /opt/wan2gp_source \
    && cd /opt/wan2gp_source \
    && git checkout 597d26b7e0e53550f57a9973c5d6a1937b2e1a7b

# 3. Install Python dependencies using a multi-stage approach that balances
# robustness against maintainability.

# First, comment out the large, pre-installed packages to prevent re-downloading.
RUN sed -i -e 's/^torch>=/#torch>=/' -e 's/^torchvision>=/#torchvision>=/' /opt/wan2gp_source/requirements.txt

# Step 1: Install the known "heavy hitters" in their own layer.
# This isolates the largest packages and allows the builder to reclaim
# temporary space before installing the rest of the requirements.
RUN python3 -m pip install --no-cache-dir \
    "opencv-python>=4.9.0.80" \
    "onnxruntime-gpu" \
    "rembg[gpu]==2.0.65" \
    "pyannote.audio"

# Step 2: Install the rest of the packages from the requirements file.
# pip will intelligently skip the packages that were already installed in the step above.
# This keeps the process dynamic for most packages.
RUN python3 -m pip install --no-cache-dir -r /opt/wan2gp_source/requirements.txt

# Step 3: Install final specific version overrides.
RUN python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6

# Final cleanup of any remaining cache.
RUN rm -rf /root/.cache/pip

# 4. Set up supervisor to run our application.
# The base image's start.sh will launch supervisord, which will then run our script.
# This is the correct way to add a service to the RunPod environment.

# Copy the supervisor configuration file.
COPY <<'EOF' /etc/supervisor/conf.d/wan2gp.conf
[program:wan2gp]
command=/usr/local/bin/start-wan2gp.sh
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/wan2gp.log
stderr_logfile=/var/log/supervisor/wan2gp_err.log
EOF

# Copy our application startup script.
COPY <<'EOF' /usr/local/bin/start-wan2gp.sh
#!/bin/bash
set -e
echo "--- Wan2GP Supervisor Startup Script ---"

# Restore application files if workspace is mounted and empty. This handles the volume mount issue.
if [ ! -f "/workspace/Wan2GP/wgp.py" ]; then
    echo "--> wgp.py not found. Restoring from backup..."
    mkdir -p /workspace/Wan2GP
    rsync -a --ignore-existing /opt/wan2gp_source/ /workspace/Wan2GP/
    echo "--> Restore complete."
fi

# Navigate to the correct directory and launch the application.
cd /workspace/Wan2GP
echo "--> Starting Wan2GP application..."
exec python3 wgp.py --server-name 0.0.0.0
EOF

# Make our startup script executable.
RUN chmod +x /usr/local/bin/start-wan2gp.sh

# 5. Expose ports. We do NOT provide a CMD, so the base image's CMD runs.
EXPOSE 7860 8888 