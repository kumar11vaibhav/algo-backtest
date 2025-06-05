@echo off
echo Starting Nifty Paper Trader...
cd /d %~dp0
cd ..
docker-compose -f docker/docker-compose.yml up -d
echo Paper trader started successfully! Check logs with: docker-compose -f docker/docker-compose.yml logs -f
