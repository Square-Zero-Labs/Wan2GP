# Use the official RunPod base image, which includes a pre-configured environment.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables for non-interactive installs.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SHELL=/bin/bash
ENV BTBN_FFMPEG_TAG=autobuild-2026-04-16-13-18
ENV BTBN_FFMPEG_ASSET=ffmpeg-n7.1.3-45-g2d6ee37238-linux64-gpl-7.1.tar.xz
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies 
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tmux \
    build-essential \
    curl \
    ca-certificates \
    xz-utils \
    rsync \
    apache2-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 only ships FFmpeg 4.4.x, which is too old for Wan2GP's current
# decode path. Install a pinned BtbN Linux x86_64 GPL build from the 7.1 branch:
# - release-branch build, not master
# - exact asset pinned by tag + filename
# - checksum verified at build time against the release's checksums.sha256 file
# BtbN targets glibc >= 2.28, which is compatible with Ubuntu 22.04.
RUN cd /tmp && \
    curl -fsSLO "https://github.com/BtbN/FFmpeg-Builds/releases/download/${BTBN_FFMPEG_TAG}/checksums.sha256" && \
    curl -fsSLO "https://github.com/BtbN/FFmpeg-Builds/releases/download/${BTBN_FFMPEG_TAG}/${BTBN_FFMPEG_ASSET}" && \
    grep " ${BTBN_FFMPEG_ASSET}\$" checksums.sha256 > "${BTBN_FFMPEG_ASSET}.sha256" && \
    sha256sum -c "${BTBN_FFMPEG_ASSET}.sha256" && \
    mkdir -p /opt/btbn-ffmpeg && \
    tar -xf "${BTBN_FFMPEG_ASSET}" -C /opt/btbn-ffmpeg --strip-components=1 && \
    install -m 755 /opt/btbn-ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg && \
    install -m 755 /opt/btbn-ffmpeg/bin/ffprobe /usr/local/bin/ffprobe && \
    if [ -f /opt/btbn-ffmpeg/bin/ffplay ]; then install -m 755 /opt/btbn-ffmpeg/bin/ffplay /usr/local/bin/ffplay; fi && \
    /usr/local/bin/ffmpeg -version && \
    rm -rf /tmp/checksums.sha256 /tmp/${BTBN_FFMPEG_ASSET} /tmp/${BTBN_FFMPEG_ASSET}.sha256

# Clone the application source code to a "safe" location that won't be volume-mounted.
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /opt/wan2gp_source \
    && cd /opt/wan2gp_source \
    && git checkout 0c0a0a711c72ecadaa7f74cc062dbf103fbeb672
# Add custom finetunes after cloning upstream.
COPY finetunes/ltx2_distilled_old_vae.json /opt/wan2gp_source/finetunes/ltx2_distilled_old_vae.json
# Install Python dependencies from the source code.
# Normalize ORT to a CUDA 12 compatible build (RunPod base image is CUDA 12.8.1).
# The GitHub Actions workflow is configured to maximize build space, so we can
# now use a single, clean RUN command for better maintainability.
RUN sed -i \
    -e '/ort-cuda-13-nightly/d' \
    -e 's/^onnxruntime-gpu==[^;]*; python_version >= "3.11"/onnxruntime-gpu==1.22.0; python_version >= "3.11"/' \
    -e 's/^torch>=/#torch>=/' \
    -e 's/^torchvision>=/#torchvision>=/' \
    /opt/wan2gp_source/requirements.txt \
    && python3 -m pip install --no-cache-dir -r /opt/wan2gp_source/requirements.txt \
    && python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6 \
    && rm -rf /root/.cache/pip

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
