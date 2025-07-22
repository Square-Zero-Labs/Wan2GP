#!/bin/bash
set -e

echo "=== Wan2GP Container Startup ==="

# Restore application files if needed (handles volume mount scenario)
if [ ! -f "/workspace/Wan2GP/wgp.py" ]; then
    echo "Restoring application files..."
    mkdir -p /workspace/Wan2GP
    rsync -a /opt/wan2gp_source/ /workspace/Wan2GP/
    echo "Application files restored"
else
    echo "Application files already present"
fi

# Start our application in the background
echo "Starting Wan2GP application in background..."
cd /workspace/Wan2GP
nohup python3 wgp.py --server-name 0.0.0.0 > /workspace/wan2gp.log 2>&1 &
echo "Wan2GP started, logs in /workspace/wan2gp.log"

# Now chain to RunPod's natural startup process
echo "Starting RunPod services..."
if [ -f "/start.sh" ]; then
    exec /start.sh
else
    echo "No /start.sh found, keeping container alive"
    tail -f /workspace/wan2gp.log
fi 