#!/bin/bash
set -e

# --- Wan2GP Restart Script ---
# Stops any running Wan2GP process and starts a fresh instance.

function error_handler {
  echo ""
  echo "❌ ERROR: Restart failed on line $1"
  echo "Check /workspace/wan2gp.log for details."
  exit 1
}
trap 'error_handler $LINENO' ERR

echo "--- Restarting Wan2GP ---"

# Ensure helper available to find processes by port (best effort)
if ! command -v lsof &> /dev/null; then
  echo "Installing lsof (one-time)..."
  apt-get update && apt-get install -y --no-install-recommends lsof >/dev/null 2>&1 || true
  apt-get clean >/dev/null 2>&1 || true
  rm -rf /var/lib/apt/lists/* >/dev/null 2>&1 || true
fi

echo "Stopping existing Wan2GP process (if any)..."
PID_TO_KILL=$(lsof -t -i:7860 2>/dev/null || true)
if [ -n "$PID_TO_KILL" ]; then
  echo "Found PID listening on 7860: $PID_TO_KILL. Killing..."
  kill -9 "$PID_TO_KILL" || true
else
  echo "No process bound to port 7860."
fi

# As a fallback, kill by script name.
pkill -9 -f wgp.py >/dev/null 2>&1 || true

sleep 1

# Ensure application files exist (handle fresh/missing workspace)
if [ ! -f "/workspace/Wan2GP/wgp.py" ]; then
  echo "Restoring application files to /workspace/Wan2GP..."
  mkdir -p /workspace/Wan2GP
  rsync -a /opt/wan2gp_source/ /workspace/Wan2GP/
fi

echo "Starting Wan2GP in background..."
cd /workspace/Wan2GP
nohup python3 wgp.py --server-name 127.0.0.1 --server-port 7860 > /workspace/wan2gp.log 2>&1 &

echo "✅ Wan2GP restarted. Monitor logs with: tail -f /workspace/wan2gp.log"

