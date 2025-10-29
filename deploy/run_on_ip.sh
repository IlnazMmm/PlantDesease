#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <bind-ip> [port]" >&2
  echo "Example: $0 87.228.99.74 8080" >&2
  exit 1
fi

BIND_IP="$1"
PORT="${2:-8080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found. Install Docker and Docker Compose v2." >&2
  exit 1
fi

echo "Building images for PlantDesease deployment..."
FRONTEND_BIND_IP="$BIND_IP" FRONTEND_PORT="$PORT" \
  docker compose -f "$COMPOSE_FILE" build

echo "Starting containers bound to $BIND_IP:$PORT..."
FRONTEND_BIND_IP="$BIND_IP" FRONTEND_PORT="$PORT" \
  docker compose -f "$COMPOSE_FILE" up -d

echo "Deployment complete. Frontend should now be reachable at http://$BIND_IP:$PORT"