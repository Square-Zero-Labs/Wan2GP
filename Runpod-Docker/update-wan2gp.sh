#!/bin/bash
set -Eeuo pipefail

APP_DIR="/workspace/Wan2GP"
STATE_DIR="/workspace/.wan2gp-state"
VENV_DIR="/opt/wan2gp-venv"
SUPPORT_DIR="/opt/wan2gp-container"
SUPERVISOR_CONFIG="/etc/supervisor/wan2gp.conf"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREVIOUS_HEAD=""
PREVIOUS_FREEZE=""
PREVIOUS_STATE=""
PREVIOUS_SOURCE_STATE=""
UPDATE_COMMITTED=0

restore_previous_version() {
  local exit_code="$1"
  trap - ERR
  set +e
  echo
  echo "ERROR: live update failed; restoring the previous version..." >&2

  cd "$APP_DIR" 2>/dev/null || true
  if [ -n "$PREVIOUS_HEAD" ]; then
    git reset --hard "$PREVIOUS_HEAD"
  fi
  if [ -n "$PREVIOUS_FREEZE" ] && [ -f "$PREVIOUS_FREEZE" ]; then
    # Sync, rather than install, so packages introduced by the failed update are
    # removed and every pre-update version (including user additions) is restored.
    uv pip sync \
      --strict \
      --python "$VENV_DIR/bin/python" \
      --constraint "$SUPPORT_DIR/core-constraints.txt" \
      "$PREVIOUS_FREEZE"
  fi
  if [ -n "$PREVIOUS_STATE" ] && [ -f "$PREVIOUS_STATE" ]; then
    cp "$PREVIOUS_STATE" "$STATE_DIR/requirements.filtered.txt"
  elif [ "$UPDATE_COMMITTED" -eq 1 ]; then
    rm -f "$STATE_DIR/requirements.filtered.txt"
  fi
  if [ -n "$PREVIOUS_SOURCE_STATE" ] && [ -f "$PREVIOUS_SOURCE_STATE" ]; then
    cp "$PREVIOUS_SOURCE_STATE" "$STATE_DIR/source-commit.txt"
  elif [ "$UPDATE_COMMITTED" -eq 1 ]; then
    rm -f "$STATE_DIR/source-commit.txt"
  fi

  supervisorctl -c "$SUPERVISOR_CONFIG" stop wan2gp || true
  supervisorctl -c "$SUPERVISOR_CONFIG" start wan2gp || true
  echo "Previous source commit restored: ${PREVIOUS_HEAD:-unknown}" >&2
  echo "Any named git stash was preserved and was not automatically reapplied." >&2
  exit "$exit_code"
}
trap 'restore_previous_version $?' ERR

echo "--- Starting Wan2GP Safe Live Update ---"
mkdir -p "$STATE_DIR/history"
cd "$APP_DIR"

if [ ! -d .git ]; then
  echo "ERROR: $APP_DIR is not a git checkout; refusing an unsafe update" >&2
  exit 1
fi

PREVIOUS_HEAD="$(git rev-parse HEAD)"
PREVIOUS_FREEZE="$STATE_DIR/history/freeze-$TIMESTAMP.txt"
"$VENV_DIR/bin/python" -m pip freeze > "$PREVIOUS_FREEZE"

if [ -f "$STATE_DIR/requirements.filtered.txt" ]; then
  PREVIOUS_STATE="$STATE_DIR/history/requirements-$TIMESTAMP.txt"
  cp "$STATE_DIR/requirements.filtered.txt" "$PREVIOUS_STATE"
fi
if [ -f "$STATE_DIR/source-commit.txt" ]; then
  PREVIOUS_SOURCE_STATE="$STATE_DIR/history/source-commit-$TIMESTAMP.txt"
  cp "$STATE_DIR/source-commit.txt" "$PREVIOUS_SOURCE_STATE"
fi

supervisorctl -c "$SUPERVISOR_CONFIG" stop wan2gp

if ! git diff --quiet || ! git diff --cached --quiet; then
  STASH_NAME="wan2gp-live-update-$TIMESTAMP"
  git stash push -m "$STASH_NAME"
  echo "Tracked local changes preserved in git stash: $STASH_NAME"
fi

git fetch --prune origin main
if git show-ref --verify --quiet refs/heads/main; then
  git switch main
else
  git switch --create main --track origin/main
fi
git merge --ff-only origin/main

CUSTOM_FINETUNE_SRC="/opt/wan2gp_source/finetunes/ltx2_distilled_old_vae.json"
CUSTOM_FINETUNE_DST="$APP_DIR/finetunes/ltx2_distilled_old_vae.json"
if [ -f "$CUSTOM_FINETUNE_SRC" ]; then
  mkdir -p "$(dirname "$CUSTOM_FINETUNE_DST")"
  cp "$CUSTOM_FINETUNE_SRC" "$CUSTOM_FINETUNE_DST"
fi

CANDIDATE_REQUIREMENTS="$STATE_DIR/requirements.filtered.candidate.txt"
"$VENV_DIR/bin/python" "$SUPPORT_DIR/filter-requirements.py" \
  "$APP_DIR/requirements.txt" "$CANDIDATE_REQUIREMENTS"

uv pip install \
  --python "$VENV_DIR/bin/python" \
  --constraint "$SUPPORT_DIR/core-constraints.txt" \
  --requirement "$CANDIDATE_REQUIREMENTS"

"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" "$SUPPORT_DIR/validate-runtime.py"
"$VENV_DIR/bin/python" -m compileall -q "$APP_DIR/wgp.py" "$APP_DIR/shared"

mv "$CANDIDATE_REQUIREMENTS" "$STATE_DIR/requirements.filtered.txt"
git rev-parse HEAD > "$STATE_DIR/source-commit.txt"
UPDATE_COMMITTED=1

"/usr/local/bin/restart-wan2gp.sh"

trap - ERR
echo
echo "Wan2GP update complete: $PREVIOUS_HEAD -> $(git rev-parse HEAD)"
echo "Core Torch/CUDA/attention packages remained pinned to the container image."
echo "Any tracked local edits remain available in the named git stash."
