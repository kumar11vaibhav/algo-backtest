#!/bin/bash

# Default values
INITIAL_BALANCE=60000
LOT_SIZE=75
START_DATE="2024-01-01"
END_DATE="2025-05-31"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --initial-balance)
            INITIAL_BALANCE="$2"
            shift 2
            ;;
        --lot-size)
            LOT_SIZE="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --initial-balance <amount>  Set initial balance (default: 60000)"
            echo "  --lot-size <size>          Set lot size (default: 75)"
            echo "  --start-date <date>        Set start date (YYYY-MM-DD) (default: 2024-01-01)"
            echo "  --end-date <date>          Set end date (YYYY-MM-DD) (default: 2025-05-31)"
            echo "  --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the backtest with the specified parameters
cd "$(dirname "$0")/.."
echo "Starting backtest with the following parameters:"
echo "Initial Balance: $INITIAL_BALANCE"
echo "Lot Size: $LOT_SIZE"
echo "Start Date: $START_DATE"
echo "End Date: $END_DATE"

# Change to the src directory
cd src

# Run the Python script with the parameters
python main.py \
    --initial-balance "$INITIAL_BALANCE" \
    --lot-size "$LOT_SIZE" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

# Check if the backtest was successful
if [ $? -eq 0 ]; then
    echo "Backtest completed successfully!"
    echo "Results are available in the reports and charts directories."
else
    echo "Backtest failed with error code $?"
    exit 1
fi
