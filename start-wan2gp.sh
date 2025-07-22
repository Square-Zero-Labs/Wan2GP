#!/bin/bash
# This script is the entrypoint for the Wan2GP Docker container.
# It starts the standard RunPod services and then launches the Wan2GP application.

set -e

echo "=== RunPod Wan2GP Startup ==="

# 1. Start RunPod services (Jupyter, SSH, etc.) in the background.
# We check if the default /start.sh exists and execute it.
echo "--> Starting RunPod services..."
if [ -f "/start.sh" ]; then
    nohup /start.sh > /tmp/runpod-services.log 2>&1 &
    echo "✅ RunPod services started. Logs are in /tmp/runpod-services.log"
else
    echo "⚠️ RunPod /start.sh not found. Standard services may not be available."
fi

# Give the background services a moment to initialize.
sleep 5

# 2. Start the Wan2GP application.
# Includes debug information to help troubleshoot file path issues.
echo "--> Starting Wan2GP application..."
echo "    - Current working directory: $(pwd)"
echo "    - Contents of /workspace:"
ls -la /workspace/
echo "    - Contents of /workspace/Wan2GP (if it exists):"
ls -la /workspace/Wan2GP/ 2>/dev/null || echo "      -> Directory not found."

# Find and execute the application script.
if [ -f "wgp.py" ]; then
    # This case is when the WORKDIR is correctly set to /workspace/Wan2GP
    echo "✅ Found wgp.py in current directory. Launching..."
    exec python3 wgp.py --server-name 0.0.0.0
elif [ -f "/workspace/Wan2GP/wgp.py" ]; then
    # This is a fallback in case the WORKDIR is /workspace
    echo "✅ Found wgp.py in /workspace/Wan2GP/. Changing directory and launching..."
    cd /workspace/Wan2GP
    exec python3 wgp.py --server-name 0.0.0.0
else
    # If the script isn't found, keep the container alive for debugging.
    echo "❌ ERROR: wgp.py not found in /workspace/Wan2GP/ or the current directory."
    echo "    The container will sleep for 1 hour to allow for debugging."
    echo "    You can connect via Jupyter Lab or SSH (if RunPod services started)."
    sleep 3600
fi 