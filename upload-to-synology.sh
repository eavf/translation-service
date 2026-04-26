#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./upload-to-synology.sh [NAS_USER] [NAS_HOST] [REMOTE_DIR] [--deploy]

Examples:
  ./upload-to-synology.sh
  ./upload-to-synology.sh vovo 192.168.0.202
  ./upload-to-synology.sh vovo 192.168.0.202 /volume1/docker/translation-service
  ./upload-to-synology.sh vovo 192.168.0.202 /volume1/docker/translation-service --deploy

Defaults:
  NAS_USER=vovo
  NAS_HOST=192.168.0.202
  REMOTE_DIR=/volume1/docker/translation-service
  SSH_PORT=22222

Options:
  --deploy   After upload, run docker compose up -d --build on the NAS.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# Set defaults
NAS_USER="vovo"
NAS_HOST="192.168.0.202"
REMOTE_DIR="/volume1/docker/translation-service"
DEPLOY=""
SSH_PORT="22222"

# Parse arguments - separate flags from positional args
pos_args=()
for arg in "$@"; do
    if [[ "$arg" == "--deploy" ]]; then
        DEPLOY="--deploy"
    else
        pos_args+=("$arg")
    fi
done

# Process positional arguments
if [[ ${#pos_args[@]} -gt 0 ]]; then
    NAS_USER="${pos_args[0]}"
fi
if [[ ${#pos_args[@]} -gt 1 ]]; then
    NAS_HOST="${pos_args[1]}"
fi
if [[ ${#pos_args[@]} -gt 2 ]]; then
    REMOTE_DIR="${pos_args[2]}"
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required on your local machine." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${NAS_USER}@${NAS_HOST}"

echo "Creating remote project directory: ${REMOTE}:${REMOTE_DIR}"
ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_DIR' '$REMOTE_DIR/models'"

echo "Uploading project files..."
rsync -az --delete -e "ssh -p $SSH_PORT" \
    --exclude 'models/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '.DS_Store' \
    "${SCRIPT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

echo "Ensuring model cache directory is writable by containers..."
ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_DIR/models' && chmod -R 777 '$REMOTE_DIR/models' 2>/dev/null || true"

if [[ "$DEPLOY" == "--deploy" ]]; then
    echo "Building and starting the Synology project..."
    echo "(sudo password for ${NAS_USER}@${NAS_HOST} may be required)"
    ssh -t -p "$SSH_PORT" "$REMOTE" "$(cat <<ENDSSH
set -e
cd '$REMOTE_DIR'
sudo /usr/local/bin/docker compose up -d --build --remove-orphans
echo "Cleaning up dangling images..."
sudo /usr/local/bin/docker image prune -f
echo "Done."
ENDSSH
)"
else
    echo "Upload complete."
    echo "To deploy on the NAS, run:"
    echo "  ssh -p $SSH_PORT ${REMOTE}"
    echo "  cd ${REMOTE_DIR}"
    echo "  sudo docker compose up -d --build --remove-orphans"
fi
