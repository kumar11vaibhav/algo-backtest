#!/bin/bash
echo "Starting Nifty Paper Trader..."
cd "$(dirname "$0")/.."
docker compose -f docker/docker-compose.yml up -d
echo "Paper trader started successfully! Check logs with: docker-compose -f docker/docker-compose.yml logs -f"
