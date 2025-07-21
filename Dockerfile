# Use actual RunPod PyTorch 2.8.0 base image as specified
# PyTorch 2.8.0 with CUDA 12.8.1 and Python 3.11
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SHELL=/bin/bash

# Explicitly set shell for RUN commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Additional system dependencies for Wan2GP  
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory (RunPod convention)
WORKDIR /workspace

# Clone Wan2GP repository to subdirectory (matches manual setup: cd workspace; git clone ...)
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git Wan2GP

# Switch to the cloned directory  
WORKDIR /workspace/Wan2GP

# Checkout specific commit
RUN git checkout 597d26b7e0e53550f57a9973c5d6a1937b2e1a7b

# Install Python dependencies with aggressive cleanup to manage disk space
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/pip && \
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    python3 -c "import gc; gc.collect()"

# Install specific versions mentioned in deploy instructions (will upgrade gradio from 5.23.0 to 5.35.0)
RUN python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6

# Final cleanup to minimize image size
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    python3 -c "import gc; gc.collect()"

# Expose ports for Gradio interface and Jupyter Lab
EXPOSE 7860 8888

# Create a startup script that runs RunPod services first, then Wan2GP
RUN echo '#!/bin/bash\n\
echo "=== RunPod Wan2GP Startup ==="\n\
\n\
# Start RunPod services (Jupyter, SSH, etc.) in background\n\
echo "Starting RunPod services..."\n\
if [ -f "/start.sh" ]; then\n\
    echo "✅ Found RunPod start.sh, starting services..."\n\
    nohup /start.sh > /tmp/runpod-services.log 2>&1 &\n\
    echo "RunPod services started (check /tmp/runpod-services.log for details)"\n\
else\n\
    echo "⚠️ RunPod start.sh not found, services may not be available"\n\
fi\n\
\n\
# Give services a moment to start\n\
sleep 5\n\
\n\
echo "=== Wan2GP Debug Info ==="\n\
echo "Current working directory: $(pwd)"\n\
echo "Contents of /workspace:"\n\
ls -la /workspace/\n\
echo "Contents of /workspace/Wan2GP (if exists):"\n\
ls -la /workspace/Wan2GP/ 2>/dev/null || echo "Directory /workspace/Wan2GP does not exist"\n\
echo "Looking for wgp.py anywhere in /workspace:"\n\
find /workspace -name "wgp.py" -type f 2>/dev/null || echo "wgp.py not found in /workspace"\n\
echo "================================"\n\
\n\
# Check if wgp.py exists in current directory\n\
if [ -f "wgp.py" ]; then\n\
    echo "✅ Found wgp.py in current directory, starting application..."\n\
    python3 wgp.py --server-name 0.0.0.0\n\
elif [ -f "/workspace/Wan2GP/wgp.py" ]; then\n\
    echo "✅ Found wgp.py in /workspace/Wan2GP, changing directory..."\n\
    cd /workspace/Wan2GP\n\
    python3 wgp.py --server-name 0.0.0.0\n\
else\n\
    echo "❌ wgp.py not found! Container will sleep for debugging..."\n\
    echo "You can connect via Jupyter Lab or SSH (RunPod services should be running)"\n\
    sleep 3600\n\
fi' > /usr/local/bin/start-wan2gp.sh && chmod +x /usr/local/bin/start-wan2gp.sh

# Set the default command to run our debug startup script
CMD ["/usr/local/bin/start-wan2gp.sh"] 