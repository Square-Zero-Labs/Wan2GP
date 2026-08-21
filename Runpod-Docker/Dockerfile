# syntax=docker/dockerfile:1.7

# The build image supplies nvcc-compatible CUDA headers when assembling the
# Python environment. Native attention extensions are downloaded as verified
# release artifacts, so no compiler or CUDA toolkit is copied into the runtime.
FROM runpod/base:1.1.0-cuda1281-ubuntu2404@sha256:f9879c67c7da08d2f7e3c8bfa5ec5f6104ad9aa8577a93a095fc4bc2bd902112 AS dependencies

ARG WAN2GP_REF=238e25fa222a254fc5d29e34319ce0231263a8f5

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_XET_HIGH_PERFORMANCE=1 \
    VIRTUAL_ENV=/opt/wan2gp-venv \
    PATH=/opt/wan2gp-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# OS packages are pinned transitively by the immutable base-image digest and
# Ubuntu snapshot visible to that image build.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    python3.11-dev \
    python3.11-venv \
    && python3.11 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade \
    pip==26.2.1 \
    wheel==0.47.0 \
    packaging==26.3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY Runpod-Docker/core-constraints.txt /opt/wan2gp-container/core-constraints.txt

# Install the image-owned CUDA stack first. PyTorch 2.10's cu128 wheel pins its
# matching Triton 3.6 runtime; constraints prevent upstream requirements from
# replacing any of these ABI-sensitive packages.
RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio==2.10.0 \
    && python -m pip install \
    --constraint /opt/wan2gp-container/core-constraints.txt \
    torchcodec==0.10.0 \
    onnxruntime-gpu==1.26.0 \
    gradio==5.35.0 \
    decord2==3.4.0 \
    hf-xet==1.6.0 \
    jupyterlab==4.5.9 \
    ipykernel==7.3.0

COPY Runpod-Docker/filter-requirements.py /opt/wan2gp-container/filter-requirements.py
COPY Runpod-Docker/validate-runtime.py /opt/wan2gp-container/validate-runtime.py

RUN git clone --filter=blob:none https://github.com/deepbeepmeep/Wan2GP.git /opt/wan2gp_source \
    && git -C /opt/wan2gp_source checkout --detach "$WAN2GP_REF" \
    && python /opt/wan2gp-container/filter-requirements.py \
    /opt/wan2gp_source/requirements.txt \
    /tmp/requirements.filtered.txt \
    && python -m pip install \
    --constraint /opt/wan2gp-container/core-constraints.txt \
    --requirement /tmp/requirements.filtered.txt

COPY finetunes/ltx2_distilled_old_vae.json /opt/wan2gp_source/finetunes/ltx2_distilled_old_vae.json


FROM dependencies AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ATTENTION_WHEEL_RELEASE=8-8-2026
ARG SAGEATTENTION_WHEEL_URL=https://github.com/Square-Zero-Labs/Wan2GP-Runpod-Wheels/releases/download/${ATTENTION_WHEEL_RELEASE}/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl
ARG SAGEATTENTION_WHEEL_SHA256=9fa280f209779caefecf70dcbc368eb1514348e89cd6fcc3e92d50e37965ed65

# Refuse an unverifiable attention artifact. Defaults are pinned to the
# published release; the GitHub workflow can override them with repository
# variables when advancing to a new release.
RUN test -n "$SAGEATTENTION_WHEEL_SHA256" \
    || { echo "SAGEATTENTION_WHEEL_SHA256 is required" >&2; exit 1; } \
    && mkdir -p /opt/wan2gp-container/attention-wheels \
    && curl --fail --location --retry 3 --retry-all-errors \
    "$SAGEATTENTION_WHEEL_URL" \
    --output /opt/wan2gp-container/attention-wheels/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl \
    && echo "$SAGEATTENTION_WHEEL_SHA256  /opt/wan2gp-container/attention-wheels/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl" \
    | sha256sum --check --strict \
    && python -m pip install --no-deps \
    /opt/wan2gp-container/attention-wheels/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl \
    && python -m pip check \
    && python /opt/wan2gp-container/validate-runtime.py \
    && rm -rf /tmp/requirements.filtered.txt /root/.cache


# Lean RunPod service base: nginx, SSH, web terminal, and Jupyter integration.
FROM runpod/base:1.1.0-ubuntu2404@sha256:8fafffd0c67a48117dff09031ed16e79c0a83cc6996472f4b4dcb76343102a66 AS runtime

ARG WAN2GP_REF=5e898b1ef15ed65910fb14f5f1f4a303957f2c60

LABEL org.opencontainers.image.title="Wan2GP RunPod" \
    org.opencontainers.image.description="Wan2GP on PyTorch 2.10 / CUDA 12.8 for A40 and RTX 5090" \
    io.square-zero-labs.wan2gp.source-ref="$WAN2GP_REF" \
    io.square-zero-labs.cuda="12.8" \
    io.square-zero-labs.pytorch="2.10.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_XET_HIGH_PERFORMANCE=1 \
    SHELL=/bin/bash \
    VIRTUAL_ENV=/opt/wan2gp-venv \
    PATH=/opt/wan2gp-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Runtime package versions are bound by the immutable base-image digest and its
# Ubuntu package snapshot.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apache2-utils \
    ca-certificates \
    curl \
    ffmpeg \
    openssl \
    rsync \
    supervisor \
    && mkdir -p /etc/nginx/conf.d /etc/supervisor \
    && getent passwd www-data >/dev/null \
    && getent group www-data >/dev/null \
    && if grep -Eq '^[[:space:]]*user[[:space:]]+[^;]+;' /etc/nginx/nginx.conf; then \
    sed -i -E 's/^[[:space:]]*user[[:space:]]+[^;]+;/user www-data;/' /etc/nginx/nginx.conf; \
    else \
    sed -i '1i user www-data;' /etc/nginx/nginx.conf; \
    fi \
    && sed -i '/^http {/a\    include /etc/nginx/conf.d/*.conf;' /etc/nginx/nginx.conf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/wan2gp-venv /opt/wan2gp-venv
COPY --from=builder /opt/wan2gp_source /opt/wan2gp_source
COPY --from=builder /opt/wan2gp-container /opt/wan2gp-container

COPY Runpod-Docker/start-wan2gp.sh /usr/local/bin/start-wan2gp.sh
COPY Runpod-Docker/update-wan2gp.sh /usr/local/bin/update-wan2gp.sh
COPY Runpod-Docker/restart-wan2gp.sh /usr/local/bin/restart-wan2gp.sh
COPY Runpod-Docker/validate-gpu.py /opt/wan2gp-container/validate-gpu.py
COPY Runpod-Docker/wan2gp-supervisor.conf /etc/supervisor/wan2gp.conf
COPY Runpod-Docker/wan2gp-nginx.conf /etc/nginx/conf.d/wan2gp.conf

RUN chmod 0755 \
    /usr/local/bin/start-wan2gp.sh \
    /usr/local/bin/update-wan2gp.sh \
    /usr/local/bin/restart-wan2gp.sh \
    /opt/wan2gp-container/filter-requirements.py \
    /opt/wan2gp-container/validate-runtime.py \
    /opt/wan2gp-container/validate-gpu.py \
    && python -m ipykernel install \
    --prefix=/usr/local \
    --name=wan2gp \
    --display-name='Wan2GP (Python 3.11)' \
    && nginx -t \
    && /usr/bin/python3 -c \
    'from supervisor.options import ServerOptions; ServerOptions().realize(["-c", "/etc/supervisor/wan2gp.conf"]); print("Supervisor configuration parsed")' \
    && if command -v nvcc >/dev/null; then \
    echo "ERROR: nvcc must not be present in the runtime image" >&2; \
    exit 1; \
    fi

WORKDIR /workspace/Wan2GP

EXPOSE 7862 8888

CMD ["/usr/local/bin/start-wan2gp.sh"]
