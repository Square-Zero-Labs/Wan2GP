# Use the official RunPod base image, which includes a pre-configured environment.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables for non-interactive installs.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tmux \
    build-essential \
    rsync \
    apache2-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the application source code to a "safe" location that won't be volume-mounted.
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /opt/wan2gp_source \
    && cd /opt/wan2gp_source \
    && git checkout 6850d60caa0edddeed922d56e8297bd955eb5dfb

# Install Python dependencies from the source code.
# The GitHub Actions workflow is configured to maximize build space, so we can
# now use a single, clean RUN command for better maintainability.
RUN sed -i -e 's/^torch>=/#torch>=/' -e 's/^torchvision>=/#torchvision>=/' /opt/wan2gp_source/requirements.txt \
    && python3 -m pip install --no-cache-dir -r /opt/wan2gp_source/requirements.txt \
    && python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6 \
    && rm -rf /root/.cache/pip

# Copy and set up our startup script
COPY start-wan2gp.sh /usr/local/bin/start-wan2gp.sh
RUN chmod +x /usr/local/bin/start-wan2gp.sh

# Expose ports for authenticated Gradio interface and Jupyter Lab
EXPOSE 7862 8888

# Use our startup script as the main command
CMD ["/usr/local/bin/start-wan2gp.sh"] 