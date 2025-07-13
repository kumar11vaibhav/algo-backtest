#!/bin/bash
echo "Starting Nifty Paper Trader..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker compose -f "$SCRIPT_DIR/../../docker/docker-compose.yml" up -d
echo "Paper trader started successfully! Check logs with: docker-compose -f docker/docker-compose.yml logs -f"
