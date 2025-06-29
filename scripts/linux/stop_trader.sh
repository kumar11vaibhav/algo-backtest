#!/bin/bash
echo "Stopping Nifty Paper Trader..."
cd "$(dirname "$0")/../.." && cd docker && docker-compose down
echo "Paper trader stopped successfully!"
