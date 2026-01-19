#!/bin/bash
set -e

# --- Wan2GP Live Update Script ---
# This script updates the Wan2GP application in a running RunPod container.
# It is designed to work around the environment's auto-restart mechanism.
# Note: Changes are temporary. A full pod restart will revert to the
# version specified in the Dockerfile.

# Safety Net: Define a function to handle errors gracefully.
function error_handler {
  echo ""
  echo "❌ ERROR: The update script failed on line $1."
  echo "An error occurred, and the update could not be completed."

  # Restore the backup script if it exists, to prevent a broken state.
  if [ -f "/workspace/Wan2GP/wgp.py.bak" ] && [ ! -f "/workspace/Wan2GP/wgp.py" ]; then
    echo "Restoring backup of wgp.py..."
    mv /workspace/Wan2GP/wgp.py.bak /workspace/Wan2GP/wgp.py
    echo "Backup restored. You may need to restart the application manually."
  fi
  exit 1
}
trap 'error_handler $LINENO' ERR

echo "--- Starting Wan2GP Live Update ---"

# Step 1: Ensure 'lsof' is available to find the process by port.
# The base RunPod image may not include this tool.
if ! command -v lsof &> /dev/null; then
    echo "Installing lsof..."
    # This requires root, which is the default in RunPod containers.
    apt-get update && apt-get install -y --no-install-recommends lsof
fi

# Step 2: Find and forcefully stop the current Wan2GP process.
echo "Searching for process on port 7860..."
PID_TO_KILL=$(lsof -t -i:7860 || true) # Use '|| true' to prevent exit on error if port is not in use
if [ -n "$PID_TO_KILL" ]; then
    echo "Found process with PID $PID_TO_KILL. Forcefully stopping it..."
    kill -9 "$PID_TO_KILL"
else
    echo "No process was running on port 7860."
fi

# As a backup, kill any process matching the script name.
pkill -9 -f wgp.py || echo "No 'wgp.py' process was found to kill."

# Step 3: IMPORTANT - Temporarily rename the script to break the auto-restart loop.
# The RunPod supervisor will try to restart the script. Renaming it prevents this.
echo "Temporarily renaming wgp.py to prevent auto-restart..."
if [ -f "/workspace/Wan2GP/wgp.py" ]; then
    mv /workspace/Wan2GP/wgp.py /workspace/Wan2GP/wgp.py.bak
    echo "Script renamed. Pausing for 3 seconds to ensure supervisor gives up."
    sleep 3
else
    echo "wgp.py not found at /workspace/Wan2GP/, skipping rename."
fi

# Step 4: Navigate to the application directory and update the code from the 'main' branch.
echo "Navigating to /workspace/Wan2GP and updating source code from 'main' branch..."
cd /workspace/Wan2GP

# Stash any local changes to prevent pull conflicts (e.g., from modified requirements.txt).
echo "Stashing local changes to avoid conflicts..."
git stash

# The git repo is in a "detached HEAD" state. We switch to the 'main' branch to pull updates.
git switch main
git pull origin main
echo "Successfully pulled latest code from the 'main' branch."

# Restore custom finetune(s) baked into the image.
CUSTOM_FINETUNE_SRC="/opt/wan2gp_source/finetunes/ltx2_distilled_old_vae.json"
CUSTOM_FINETUNE_DST="/workspace/Wan2GP/finetunes/ltx2_distilled_old_vae.json"
if [ -f "$CUSTOM_FINETUNE_SRC" ]; then
    echo "Restoring custom finetune: ltx2_distilled_old_vae.json"
    mkdir -p /workspace/Wan2GP/finetunes
    cp "$CUSTOM_FINETUNE_SRC" "$CUSTOM_FINETUNE_DST"
else
    echo "Custom finetune source not found at $CUSTOM_FINETUNE_SRC, skipping."
fi

# The updated 'wgp.py' is now in place. We can remove the backup file.
if [ -f "/workspace/Wan2GP/wgp.py.bak" ]; then
    rm /workspace/Wan2GP/wgp.py.bak
fi

# Step 5: Update Python dependencies.
# This follows the same logic as the original Dockerfile setup.
echo "Updating Python dependencies from requirements.txt..."
sed -i -e 's/^torch>=/#torch>=/' -e 's/^torchvision>=/#torchvision>=/' requirements.txt
python3 -m pip install --no-cache-dir -r requirements.txt
python3 -m pip install --no-cache-dir gradio==5.35.0 sageattention==1.0.6
echo "Dependencies updated."

# Step 6: Restart the Wan2GP application with the new code.
# This uses the same command as the original start-wan2gp.sh script.
echo "Restarting Wan2GP application in the background..."
nohup python3 wgp.py --server-name 127.0.0.1 --server-port 7860 --save-masks > /workspace/wan2gp.log 2>&1 &

echo ""
echo "✅ --- Wan2GP Update Complete ---"
echo "The application has been updated and restarted."
echo "Monitor logs with: tail -f /workspace/wan2gp.log"
