# Use actual RunPod PyTorch 2.8.0 base image as specified
# PyTorch 2.8.0 with CUDA 12.8.1 and Python 3.11
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Additional system dependencies for Wan2GP
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory (RunPod convention)
WORKDIR /workspace

# Clone Wan2GP repository at specific commit as specified in deploy instructions
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git . \
    && git checkout 597d26b7e0e53550f57a9973c5d6a1937b2e1a7b

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install specific versions mentioned in deploy instructions
RUN python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6

# Expose port 7860 for Gradio interface
EXPOSE 7860

# Set the default command to run the application
CMD ["python3", "wgp.py", "--server-name", "0.0.0.0"] 