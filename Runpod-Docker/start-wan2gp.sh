#!/bin/bash
set -Eeuo pipefail

APP_DIR="/workspace/Wan2GP"
SOURCE_DIR="/opt/wan2gp_source"
STATE_DIR="/workspace/.wan2gp-state"
VENV_DIR="/opt/wan2gp-venv"
CONTAINER_SUPPORT_DIR="/opt/wan2gp-container"

echo "=== Wan2GP Container Startup ==="

if [ ! -f "$APP_DIR/wgp.py" ]; then
  echo "Restoring Wan2GP application files..."
  mkdir -p "$APP_DIR"
  rsync -a "$SOURCE_DIR/" "$APP_DIR/"
else
  echo "Using existing Wan2GP application files in $APP_DIR"
fi

mkdir -p "$STATE_DIR"

# A live update is stored on the persistent volume while the Python environment
# lives on the replaceable container disk. Reapply its compatible non-core
# requirements whenever a new container is created for the same volume.
if [ -f "$STATE_DIR/requirements.filtered.txt" ]; then
  echo "Reconciling dependencies for the persisted live update..."
  uv pip install \
    --python "$VENV_DIR/bin/python" \
    --constraint "$CONTAINER_SUPPORT_DIR/core-constraints.txt" \
    --requirement "$STATE_DIR/requirements.filtered.txt"
fi

"$VENV_DIR/bin/python" "$CONTAINER_SUPPORT_DIR/validate-runtime.py"

WAN2GP_USERNAME="${WAN2GP_USERNAME:-admin}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-gpuPoor2025}"
htpasswd -cb /etc/nginx/.wan2gp-htpasswd "$WAN2GP_USERNAME" "$WAN2GP_PASSWORD" >/dev/null
chown root:nogroup /etc/nginx/.wan2gp-htpasswd
chmod 640 /etc/nginx/.wan2gp-htpasswd
nginx -t

if [ -z "${JUPYTER_PASSWORD:-}" ]; then
  JUPYTER_PASSWORD="$(openssl rand -hex 24)"
  export JUPYTER_PASSWORD
  echo "Jupyter token generated. Retrieve it with: jupyter server list"
else
  echo "Using the configured JUPYTER_PASSWORD token."
fi

echo "Wan2GP login user: $WAN2GP_USERNAME"
echo "Authenticated Wan2GP endpoint: port 7862"

# RunPod's service launcher owns nginx, Jupyter, SSH, and web-terminal setup.
# Supervisor remains PID 1 and owns the Wan2GP application lifecycle.
/start.sh &
RUNPOD_SERVICES_PID=$!
sleep 2
if ! kill -0 "$RUNPOD_SERVICES_PID" 2>/dev/null; then
  echo "ERROR: RunPod services failed during startup" >&2
  wait "$RUNPOD_SERVICES_PID"
fi

exec /usr/bin/supervisord -n -c /etc/supervisor/wan2gp.conf
