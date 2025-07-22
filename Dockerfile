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

# Install system dependencies and perform cleanup in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tmux \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory (RunPod convention)
WORKDIR /workspace

# Clone Wan2GP repository to subdirectory (matches manual setup: cd workspace; git clone ...)
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git Wan2GP

# Switch to the cloned directory
WORKDIR /workspace/Wan2GP

# Checkout specific commit
RUN git checkout 597d26b7e0e53550f57a9973c5d6a1937b2e1a7b

# Install Python dependencies with aggressive cleanup to avoid running out of space.
# We install from requirements.txt (without --upgrade), then handle specific version
# overrides, and finally clean up all temporary files in a single layer.
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache/pip \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Expose ports for Gradio interface and Jupyter Lab
EXPOSE 7860 8888

# Copy our custom startup script into the container
COPY start-wan2gp.sh /usr/local/bin/start-wan2gp.sh
RUN chmod +x /usr/local/bin/start-wan2gp.sh

# Set the default command to run our startup script
CMD ["/usr/local/bin/start-wan2gp.sh"] 