from src.paper_trader import PaperTrader
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run paper trading for Nifty options strategy')
    parser.add_argument('--initial-balance', type=float, default=60000,
                      help='Initial balance for paper trading (default: 60000)')
    parser.add_argument('--lot-size', type=int, default=75,
                      help='Lot size for trading (default: 75)')
    
    args = parser.parse_args()
    
    # Create paper trader instance
    trader = PaperTrader(initial_balance=args.initial_balance, lot_size=args.lot_size)
    
    try:
        print("Starting paper trading...")
        print(f"Initial balance: Rs. {args.initial_balance:,.2f}")
        print(f"Lot size: {args.lot_size}")
        print("\nPress Ctrl+C to stop and view performance report\n")
        
        # Start paper trading
        trader.run_paper_trading()
        
    except KeyboardInterrupt:
        print("\nStopping paper trading...")
        print("\nGenerating performance report...\n")
        print(trader.generate_performance_report())

if __name__ == "__main__":
    main()
