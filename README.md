# Nifty Options Trading System

This system includes both backtesting and paper trading capabilities for Nifty options trading.

## Project Structure

```text
backtesting/
├── src/           # Python source code
├── docker/        # Docker configuration files
├── scripts/       # Batch scripts for management
├── config/        # Configuration files
├── data/          # Trading data and state
└── logs/          # Application logs
```

## Features

- Backtesting with historical data
- Paper trading with real-time market data
- Automated trading strategy
- Performance analytics and reporting
- Containerized deployment

## Installation Guide

### Prerequisites

1. Install Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop)
2. Make sure Docker is running on your system

### Running Backtests

Use the backtest script to test your strategy with historical data:

```bash
cd scripts
run_backtest.bat --initial-balance 100000 --lot-size 75 --start-date 2024-01-01 --end-date 2024-12-31
```

Options:
- `--initial-balance`: Starting capital (default: 60000)
- `--lot-size`: Number of contracts per trade (default: 75)
- `--start-date`: Backtest start date (format: YYYY-MM-DD)
- `--end-date`: Backtest end date (format: YYYY-MM-DD)

### Running the Paper Trader

1. Navigate to the `scripts` directory
2. Double-click `start_trader.bat` to start the paper trading system
3. The system will run continuously and automatically restart if there are any crashes
4. Logs are stored in the `logs` directory
5. Trading state is persisted in `data/paper_trade_state.json`

### Monitoring

1. View logs in real-time:

   ```bash
   docker-compose logs -f
   ```

1. Check container status:

   ```bash
   docker-compose ps
   ```

### Stopping the Paper Trader

1. Navigate to the `scripts` directory
2. Double-click `stop_trader.bat` to stop the paper trading system

### Auto-start on System Boot

1. Press Win + R
1. Type `shell:startup`
1. Create a shortcut to `scripts/start_trader.bat` in this folder

## System Configuration

- Initial balance and lot size can be modified in `docker/docker-compose.yml`
- Trading parameters can be adjusted in `src/paper_trader.py`
- Time zone settings can be changed in `docker/Dockerfile`

## System Maintenance

1. Logs are automatically rotated (max 3 files of 10MB each)
1. Trading state is persisted across restarts
1. System automatically handles market hours

## Technical Details

### Data Sources

- Historical data fetching from Yahoo Finance
- Real-time market data for paper trading

### Core Components

- Black-Scholes option pricing model
- Trading strategy implementation
- Performance visualization and analytics
- Excel report generation

### Source Files

- `src/data_handler.py`: Data fetching and processing
- `src/option_pricer.py`: Black-Scholes pricing implementation
- `src/strategy.py`: Trading strategy logic
- `src/performance_analytics.py`: Performance metrics and visualization
- `src/paper_trader.py`: Paper trading implementation
- `src/main.py`: Main backtesting script

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the backtester:
```bash
python main.py
```

## Features

- Fetches historical Nifty data from Yahoo Finance
- Implements Black-Scholes option pricing
- Supports custom trading strategies
- Generates performance metrics and visualizations
- Exports trade data to Excel with multiple sheets
- Saves high-resolution charts for analysis

## Project Structure

### Core Modules

1. `data_handler.py`
   - Handles data fetching from Yahoo Finance
   - Manages historical data access
   - Calculates technical indicators and volatility

2. `option_pricer.py`
   - Implements Black-Scholes option pricing
   - Handles dynamic volatility adjustments
   - Supports both call and put options

3. `strategy.py`
   - Contains core trading logic
   - Manages positions and trade execution
   - Tracks results and trade logs

4. `performance_analytics.py`
   - Calculates performance metrics
   - Generates equity curves
   - Exports trade data to CSV reports

5. `main.py`
   - Main entry point
   - Orchestrates all components
   - Runs the backtest

### Output

The system generates the following outputs:

1. Console Output
   - Real-time trade notifications
   - Daily price and trigger information
   - Final performance summary

2. CSV Reports (in `reports` directory)
   - Detailed trade log
   - Daily performance metrics
   - Strategy statistics

3. Visualizations (in `charts` directory)
   - Equity curve chart (portfolio value over time)
   - Trade distribution chart (win/loss analysis)
   - Daily returns chart (percentage returns)

## Usage

1. Configure strategy parameters in `config.py`
2. Run the backtest:

   ```bash
   python main.py
   ```

3. View results in the console and check the `reports` directory for detailed CSV reports

## Example Strategy

The default strategy implements:
- Entry on 50% price increase from previous close
- Exit at next day's market open (9:30 AM)
- Position sizing of 1 lot (50 units)
- ATM option selection
