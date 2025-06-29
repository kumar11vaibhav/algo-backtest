import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import yfinance as yf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy_fixed import Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataHandler:
    """Simple data handler for backtesting"""
    def __init__(self, data):
        self.data = data
        
    def get_data_length(self):
        return len(self.data)

def fetch_nifty_data(start_date, end_date):
    """Fetch NIFTY 50 data from Yahoo Finance"""
    logger.info(f"Fetching NIFTY 50 data from {start_date} to {end_date}")
    
    # Download data
    ticker = "^NSEI"
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Clean up data
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.ffill().dropna()
    
    logger.info(f"Fetched {len(df)} days of data")
    return df

def main():
    # Set up date range (last 1 year)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        # Fetch and prepare data
        data = fetch_nifty_data(start_date, end_date)
        data_handler = DataHandler(data)
        
        # Initialize and run strategy
        strategy = Strategy(data_handler)
        results = strategy.run()
        
        # Print final results
        print("\n=== Backtest Results ===")
        print(f"Initial Balance: {strategy.initial_balance:,.2f}")
        print(f"Final Balance: {strategy.initial_balance + results['total_pnl']:,.2f}")
        print(f"Total Return: {(results['total_pnl']/strategy.initial_balance)*100:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            print(f"Winning Trades: {results['winning_trades']} ({(results['winning_trades']/results['total_trades']*100):.1f}%)")
            print(f"Losing Trades: {results['losing_trades']} ({(results['losing_trades']/results['total_trades']*100):.1f}%)")
            
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
