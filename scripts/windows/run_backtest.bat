@echo off
setlocal enabledelayedexpansion

:: Default values
set INITIAL_BALANCE=60000
set LOT_SIZE=75
set START_DATE=2024-01-01
set END_DATE=2025-05-31

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--initial-balance" set INITIAL_BALANCE=%~2& shift & shift & goto :parse_args
if /i "%~1"=="--lot-size" set LOT_SIZE=%~2& shift & shift & goto :parse_args
if /i "%~1"=="--start-date" set START_DATE=%~2& shift & shift & goto :parse_args
if /i "%~1"=="--end-date" set END_DATE=%~2& shift & shift & goto :parse_args
if /i "%~1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: run_backtest.bat [options]
echo.
echo Options:
echo   --initial-balance N    Set initial balance (default: 60000)
echo   --lot-size N          Set lot size (default: 75)
echo   --start-date DATE     Set start date (format: YYYY-MM-DD, default: 2024-01-01)
echo   --end-date DATE       Set end date (format: YYYY-MM-DD, default: 2025-05-31)
echo   --help                Show this help message
echo.
echo Example:
echo   run_backtest.bat --initial-balance 100000 --lot-size 75 --start-date 2024-01-01 --end-date 2024-12-31
exit /b 0

:done_parsing
echo Running backtest with the following parameters:
echo Initial Balance: Rs. %INITIAL_BALANCE%
echo Lot Size: %LOT_SIZE%
echo Date Range: %START_DATE% to %END_DATE%
echo.

:: Change to the project root directory
cd /d "%~dp0..\..\src"

:: Run the backtest with the full path to main.py
python main.py --initial-balance %INITIAL_BALANCE% --lot-size %LOT_SIZE% --start-date %START_DATE% --end-date %END_DATE%

if errorlevel 1 (
    echo Error running backtest
    exit /b 1
)

echo.
echo Backtest completed successfully!
echo Results are available in the reports and charts directories.
