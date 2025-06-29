@echo off
echo Stopping Nifty Paper Trader...
cd /d %~dp0
cd ..\..\docker && docker-compose down
echo Paper trader stopped successfully!
