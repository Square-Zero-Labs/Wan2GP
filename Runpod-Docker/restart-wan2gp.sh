#!/bin/bash
set -Eeuo pipefail

SUPERVISOR_CONFIG="/etc/supervisor/wan2gp.conf"
HEALTH_URL="http://127.0.0.1:7860/"
HEALTH_TIMEOUT="${WAN2GP_HEALTH_TIMEOUT:-180}"

echo "--- Restarting Wan2GP ---"
if supervisorctl -c "$SUPERVISOR_CONFIG" status wan2gp | grep -q RUNNING; then
  supervisorctl -c "$SUPERVISOR_CONFIG" restart wan2gp
else
  supervisorctl -c "$SUPERVISOR_CONFIG" start wan2gp
fi

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "ERROR: Wan2GP did not become healthy within ${HEALTH_TIMEOUT}s" >&2
    supervisorctl -c "$SUPERVISOR_CONFIG" status wan2gp || true
    tail -n 80 /workspace/wan2gp.log || true
    exit 1
  fi
  sleep 2
done

supervisorctl -c "$SUPERVISOR_CONFIG" status wan2gp
echo "Wan2GP restarted successfully. Logs: /workspace/wan2gp.log"
