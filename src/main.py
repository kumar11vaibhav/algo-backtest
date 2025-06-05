from datetime import datetime, timedelta
from data_handler import DataHandler
from option_pricer import OptionPricer
from strategy import Strategy
from performance_analytics import PerformanceAnalytics
import argparse
import matplotlib.pyplot as plt

def main():
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
    
    # Convert dates to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize components
    data_handler = DataHandler(symbol="^NSEI", start_date=start_date, end_date=end_date)
    option_pricer = OptionPricer(base_volatility=0.15, interest_rate=0.07)
    strategy = Strategy(data_handler, option_pricer, lot_size=args.lot_size)
    
    # Fetch data
    if not data_handler.fetch_historical_data(start_date, end_date):
        return
    
    # Run strategy
    results_df, trade_log_df, stats = strategy.run(args.initial_balance)
    
    # Analyze performance
    analytics = PerformanceAnalytics(results_df, trade_log_df, stats)
    analytics.print_summary(args.initial_balance)
    # Generate charts and reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analytics.plot_and_save_charts(timestamp)
    analytics.export_to_excel(args.initial_balance)
    


if __name__ == "__main__":
    main()
