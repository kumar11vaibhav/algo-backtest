from datetime import datetime, timedelta
from data_handler import DataHandler
from option_pricer import OptionPricer
from strategy_fixed import Strategy
from performance_analytics import PerformanceAnalytics
import argparse
import logging
import sys
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run Nifty Options backtesting')
        parser.add_argument('--initial-balance', type=float, default=60000,
                          help='Initial balance for backtesting')
        parser.add_argument('--lot-size', type=int, default=75,
                          help='Lot size for trading')
        parser.add_argument('--start-date', type=str, default='2024-01-01',
                          help='Start date for backtesting (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, default='2025-05-31',
                          help='End date for backtesting (YYYY-MM-DD)')
        
        args = parser.parse_args()
        
        logger.info("Starting backtest with parameters:")
        logger.info(f"Initial Balance: {args.initial_balance}")
        logger.info(f"Lot Size: {args.lot_size}")
        logger.info(f"Date Range: {args.start_date} to {args.end_date}")
        
        # Convert dates to datetime objects
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format. Please use YYYY-MM-DD format. Error: {e}")
            return 1
            
        # Initialize components
        logger.info("Initializing DataHandler...")
        data_handler = DataHandler(symbol="^NSEI", start_date=start_date, end_date=end_date)
        
        logger.info("Initializing OptionPricer...")
        option_pricer = OptionPricer(risk_free_rate=0.07)  # Using risk-free rate of 7%
        
        logger.info("Initializing Strategy...")
        strategy = Strategy(data_handler, option_pricer, initial_balance=args.initial_balance)
        
        # Fetch data
        logger.info("Fetching historical data...")
        if not data_handler.fetch_historical_data(start_date, end_date):
            logger.error("Failed to fetch historical data. Exiting.")
            return 1
            
        logger.info(f"Successfully loaded {len(data_handler.data)} data points")
        
        # Set up indicators now that data is loaded
        logger.info("Setting up technical indicators...")
        if not strategy._setup_indicators():
            logger.error("Failed to set up indicators. Exiting.")
            return 1
            
        # Run strategy
        logger.info("Running strategy...")
        results_df, trade_log_df, stats = strategy.run()
        
        if results_df is None or trade_log_df is None:
            logger.error("Strategy execution failed. No results returned.")
            return 1
            
        # Analyze performance
        logger.info("Analyzing performance...")
        analytics = PerformanceAnalytics(results_df, trade_log_df, stats)
        analytics.print_summary(args.initial_balance)
        
        # Generate charts and reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info("Generating charts and reports...")
        try:
            analytics.plot_and_save_charts(timestamp)
            analytics.export_to_excel(args.initial_balance)
            logger.info(f"Reports and charts saved with timestamp: {timestamp}")
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return 1
            
        logger.info("Backtest completed successfully!")
        return 0
        
    except Exception as e:
        logger.exception("An unexpected error occurred during backtesting:")
        return 1
    


if __name__ == "__main__":
    main()
