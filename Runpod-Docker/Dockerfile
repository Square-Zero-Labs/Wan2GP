# Use the closest current RunPod CUDA 13 base, then upgrade Torch in-image.
FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

# Set environment variables for non-interactive installs.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SHELL=/bin/bash
ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0;12.0"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA=1
ENV MAX_JOBS=8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies 
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    tmux \
    build-essential \
    rsync \
    apache2-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RunPod's base image defaults python/pip to 3.12; force 3.11 for this image.
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3 \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip

# Clone the application source code to a "safe" location that won't be volume-mounted.
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /opt/wan2gp_source \
    && cd /opt/wan2gp_source \
    && git checkout a0ea09158e5ad9a6017536fa0171c35851e5fc07
# Add custom finetunes after cloning upstream.
COPY finetunes/ltx2_distilled_old_vae.json /opt/wan2gp_source/finetunes/ltx2_distilled_old_vae.json
COPY Runpod-Docker/install-python-deps.sh /usr/local/bin/install-python-deps.sh
# Install Python dependencies from the source code.
# The RunPod CUDA 13 base currently tops out at Torch 2.9.1, so we explicitly
# upgrade the PyTorch stack to the CUDA 13.0 wheels we want in this image.
RUN chmod +x /usr/local/bin/install-python-deps.sh
RUN /usr/local/bin/install-python-deps.sh
RUN apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and set up our startup and update scripts
COPY Runpod-Docker/start-wan2gp.sh /usr/local/bin/start-wan2gp.sh
COPY Runpod-Docker/update-wan2gp.sh /usr/local/bin/update-wan2gp.sh
COPY Runpod-Docker/restart-wan2gp.sh /usr/local/bin/restart-wan2gp.sh
RUN chmod +x /usr/local/bin/start-wan2gp.sh \
    && chmod +x /usr/local/bin/update-wan2gp.sh \
    && chmod +x /usr/local/bin/restart-wan2gp.sh

# Expose ports for authenticated Gradio interface and Jupyter Lab
EXPOSE 7862 8888

# Use our startup script as the main command
CMD ["/usr/local/bin/start-wan2gp.sh"] 
