@echo off
echo Stopping Nifty Paper Trader...
cd ../docker && docker-compose down
echo Paper trader stopped successfully!
