import yfinance as yf
import pandas as pd
from datetime import datetime, time as dt_time
import pytz
import time
import time as time_module
import zoneinfo
import os
import json
import logging
from src.option_pricer import OptionPricer
from src.data_handler import DataHandler


class PaperTrader:
    def __init__(self, initial_balance=60000, lot_size=75):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.lot_size = lot_size
        self.positions = []
        self.trade_log = []
        self.data_handler = DataHandler()
        self.option_pricer = OptionPricer()
        self.csv_log_file = None

        # Create necessary directories
        os.makedirs("data/trade_logs", exist_ok=True)
        os.makedirs("paper_trade_logs", exist_ok=True)

        # Setup logging
        self.setup_logging()
        self.setup_trade_log()

        # Load or create trade state
        self.state_file = "data/paper_trade_state.json"
        self.load_state()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = "paper_trade_logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir, f'paper_trade_{datetime.now().strftime("%Y%m%d")}.log'
        )

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create file handler with debug level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Create console handler with info level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Get the root logger and set its level to DEBUG
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add the handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Set specific loggers to reduce noise
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("yfinance").setLevel(logging.WARNING)

    def load_state(self):
        """Load previous trading state if exists"""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self.current_balance = state.get("balance", self.initial_balance)
                self.positions = state.get("positions", [])
                self.trade_log = state.get("trade_log", [])
                self.logger.info(
                    f"Loaded previous state. Balance: {self.current_balance}"
                )
        else:
            self.save_state()

    def save_state(self):
        """Save current trading state"""
        state = {
            "balance": self.current_balance,
            "positions": self.positions,
            "trade_log": self.trade_log,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=4, default=str)

    def get_current_market_data(self, max_retries=3, retry_delay=5):
        """Get current Nifty spot price with retry logic

        Args:
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds

        Returns:
            float: Current market price or None if unable to fetch
        """
        symbols_to_try = [
            "^NSEI",  # NIFTY 50 Index
            "^NSEI.NS",  # NSE NIFTY 50
            "NSEI.BO",  # BSE NIFTY 50
            "^NSEBANK",  # NIFTY Bank as fallback
        ]

        for attempt in range(max_retries):
            for symbol in symbols_to_try:
                try:
                    self.logger.debug(
                        f"Attempting to fetch data for {symbol} (attempt {attempt + 1}/{max_retries})"
                    )
                    ticker = yf.Ticker(symbol)
                    current_data = ticker.history(period="1d", interval="1m")

                    if not current_data.empty and "Close" in current_data.columns:
                        price = float(current_data["Close"].iloc[-1])
                        self.logger.info(
                            f"Successfully fetched price for {symbol}: {price}"
                        )
                        return price

                except Exception as e:
                    self.logger.warning(f"Error fetching {symbol}: {str(e)}")
                    continue

            if attempt < max_retries - 1:
                self.logger.info(f"Retrying in {retry_delay} seconds...")
                time_module.sleep(retry_delay)

        self.logger.error("Failed to fetch market data after all attempts")
        return None

    def check_market_hours(self):
        """Check if market is open"""
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_start <= now <= market_end

    def setup_trade_log(self):
        """Initialize the trade log CSV file with headers if it doesn't exist"""
        today = datetime.now().strftime("%Y%m%d")
        self.csv_log_file = f"data/trade_logs/trades_{today}.csv"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.csv_log_file), exist_ok=True)
        
        # Write headers if file doesn't exist
        if not os.path.exists(self.csv_log_file):
            with open(self.csv_log_file, 'w') as f:
                f.write("timestamp,action,strike,price,quantity,balance_before,pnl,reason\n")

    def log_trade_to_csv(self, trade):
        """Log trade details to the CSV file"""
        try:
            with open(self.csv_log_file, 'a') as f:
                f.write(
                    f"{trade['date'].isoformat()},"
                    f"{trade['action']},"
                    f"{trade.get('strike', '')},"
                    f"{trade.get('price', '')},"
                    f"{trade.get('quantity', '')},"
                    f"{trade.get('balance_before', '')},"
                    f"{trade.get('pnl', '')},"
                    f"{trade.get('reason', '')}\n"
                )
        except Exception as e:
            self.logger.error(f"Error writing to trade log CSV: {str(e)}")

    def execute_paper_trade(self, action, strike=None, price=None, reason=None):
        """Execute and log a paper trade"""
        # Mark that we've executed at least one trade today
        self.trades_executed_today = True
        
        trade = {
            "date": datetime.now(),
            "action": action,
            "strike": strike,
            "price": price,
            "quantity": self.lot_size,
            "balance_before": self.current_balance,
            "reason": reason,
        }

        if action == "BUY":
            cost = price * self.lot_size
            self.current_balance -= cost
            self.positions.append(
                {
                    "strike": strike,
                    "entry_price": price,
                    "entry_time": datetime.now(),
                    "quantity": self.lot_size,
                }
            )
            trade["pnl"] = 0  # Initialize PnL as 0 for BUY
        elif action == "EXIT":
            if self.positions:
                position = self.positions[0]  # Get the first position
                pnl = (price - position["entry_price"]) * self.lot_size
                self.current_balance += price * self.lot_size
                trade["pnl"] = pnl
                self.positions = []  # Clear position
            else:
                trade["pnl"] = 0  # No position to exit
        else:
            trade["pnl"] = 0  # Default PnL for other actions

        # Log the trade
        self.trade_log.append(trade)
        self.logger.info(f"Trade executed: {trade}")
        
        # Log to CSV
        self.log_trade_to_csv(trade)
        
        # Save state
        self.save_state()
        
        return trade  # Return the trade details for further processing if needed

    def check_entry_conditions(self, current_price, prev_close):
        """Check if entry conditions are met"""
        try:
            # Ensure inputs are numeric
            current_price = float(current_price)
            prev_close = float(prev_close)

            # Log input values for debugging
            self.logger.debug(
                f"check_entry_conditions - current_price: {current_price}, prev_close: {prev_close}"
            )

            # Calculate trigger price (1.5x previous close)
            trigger_price = prev_close * 1.5

            # Calculate how much of the trigger price the current price represents (as percentage)
            price_ratio = (current_price / trigger_price) * 100

            # Log calculations for debugging
            self.logger.debug(
                f"check_entry_conditions - trigger_price: {trigger_price}, price_ratio: {price_ratio:.2f}%"
            )

            # Return whether current price meets or exceeds trigger, along with calculated values
            return current_price >= trigger_price, trigger_price, price_ratio

        except (TypeError, ValueError) as e:
            self.logger.error(f"Error in check_entry_conditions: {str(e)}")
            self.logger.error(
                f"current_price type: {type(current_price)}, value: {current_price}"
            )
            self.logger.error(
                f"prev_close type: {type(prev_close)}, value: {prev_close}"
            )
            # Return False to prevent entry if there's an error
            return False, 0, 0

    def run_paper_trading(self):
        """Main paper trading loop"""
        self.logger.info("Starting paper trading session...")
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Track if any trades were executed today
        self.trades_executed_today = False
        self.last_trade_check_date = datetime.now().date()

        while True:
            try:
                self.logger.debug("Checking market hours...")
                market_open = self.check_market_hours()
                self.logger.debug(f"Market open: {market_open}")

                if not market_open:
                    self.logger.info("Market is closed. Waiting for market hours...")
                    # Wait for 5 minutes before checking again
                    time_module.sleep(300)  # 300 seconds = 5 minutes
                    consecutive_failures = 0  # Reset failures when market is closed
                    continue

                try:
                    # Try to get market data with retries
                    self.logger.debug("Fetching current market data...")
                    spot_price = self.get_current_market_data(
                        max_retries=3, retry_delay=5
                    )
                    self.logger.debug(f"Got spot price: {spot_price}")

                    if spot_price is None:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error(
                                f"Failed to fetch market data after {max_consecutive_failures} attempts. Exiting..."
                            )
                            break
                        wait_time = min(
                            60 * consecutive_failures, 300
                        )  # Exponential backoff up to 5 minutes
                        self.logger.warning(
                            f"Could not fetch market data. Will retry in {wait_time} seconds. Attempt {consecutive_failures}/{max_consecutive_failures}"
                        )
                        time_module.sleep(wait_time)
                        continue

                    # Reset failure counter on successful data fetch
                    consecutive_failures = 0
                    
                    # Check if we need to log a 'NO_TRADE' record for the previous day
                    current_date = datetime.now().date()
                    if current_date != self.last_trade_check_date:
                        if not self.trades_executed_today and self.check_market_hours():
                            # Log 'NO_TRADE' for the previous day
                            self.logger.info("No trades executed yesterday, logging 'NO_TRADE' record")
                            no_trade_record = {
                                'date': datetime.combine(self.last_trade_check_date, datetime.min.time()),
                                'action': 'NO_TRADE',
                                'strike': '',
                                'price': '',
                                'quantity': '',
                                'balance_before': self.current_balance,
                                'pnl': 0,
                                'reason': 'No trade conditions met'
                            }
                            self.log_trade_to_csv(no_trade_record)
                        
                        # Reset for the new day
                        self.trades_executed_today = False
                        self.last_trade_check_date = current_date

                    # Get historical data for volatility calculation
                    self.logger.debug("Calculating historical volatility...")
                    hist_data = self.data_handler.calculate_historical_volatility()

                    if hist_data is None or len(hist_data) == 0:
                        self.logger.warning(
                            "Could not calculate historical volatility. Using default value."
                        )
                        hist_vol = 0.15  # Default 15% volatility if calculation fails
                    else:
                        hist_vol = float(hist_data.iloc[-1])
                        self.logger.info(
                            f"Calculated historical volatility: {hist_vol*100:.2f}%"
                        )

                    # Get strike price (3 strikes below spot)
                    self.logger.debug("Getting strike price...")
                    strike = self.data_handler.get_strike_price(spot_price)
                    self.logger.info(
                        f"Current spot: {spot_price}, Selected strike: {strike}"
                    )

                    # Get current time in IST
                    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
                    current_datetime = datetime.now(ist)
                    current_time = current_datetime.time()
                    self.logger.debug(f"Current time in IST: {current_time}")

                    # Check for exit at 9:30 AM
                    if current_time >= time(9, 30) and self.positions:
                        self.logger.info("Checking for exit conditions...")
                        current_option_price = self.option_pricer.calculate_price(
                            spot_price,
                            self.positions[0]["strike"],
                            30,
                            hist_vol,
                            option_type="call",
                        )
                        self.logger.debug(f"Exit option price: {current_option_price}")
                        self.execute_paper_trade(
                            "EXIT",
                            self.positions[0]["strike"],
                            current_option_price,
                            "Exit at 9:30 AM",
                        )
                        continue

                    # Check for new entry if no positions
                    if not self.positions:
                        current_option_price = self.option_pricer.calculate_price(
                            spot_price, strike, 30, hist_vol, option_type="call"
                        )

                        # Get previous day's close for comparison
                        self.logger.debug(
                            "Fetching previous day's data for entry condition check"
                        )
                        prev_data = yf.Ticker("^NSEI").history(period="2d")
                        if len(prev_data) >= 2:
                            self.logger.debug(
                                f"Previous data available. Current price: {spot_price}, Strike: {strike}, Hist Vol: {hist_vol}"
                            )

                            # Calculate previous day's option price
                            prev_close_price = float(prev_data["Close"].iloc[-2])
                            self.logger.debug(
                                f"Previous day's close price: {prev_close_price}"
                            )

                            prev_option_price = self.option_pricer.calculate_price(
                                prev_close_price,
                                strike,
                                30,
                                hist_vol,
                                option_type="call",
                            )
                            self.logger.debug(
                                f"Previous day's option price: {prev_option_price}"
                            )

                            # Log values before calling check_entry_conditions
                            self.logger.debug(
                                f"Calling check_entry_conditions with current_option_price={current_option_price}, prev_option_price={prev_option_price}"
                            )

                            try:
                                should_enter, trigger_price, price_ratio = (
                                    self.check_entry_conditions(
                                        current_option_price, prev_option_price
                                    )
                                )

                                self.logger.debug(
                                    f"check_entry_conditions result - should_enter: {should_enter}, trigger_price: {trigger_price}, price_ratio: {price_ratio}"
                                )

                                if should_enter:
                                    self.execute_paper_trade(
                                        "BUY",
                                        strike,
                                        current_option_price,
                                        f"Trigger price {trigger_price:.2f} hit",
                                    )
                            except Exception as e:
                                self.logger.error(
                                    f"Error in check_entry_conditions: {str(e)}",
                                    exc_info=True,
                                )

                except Exception as e:
                    self.logger.error(
                        f"Error in market data processing: {str(e)}", exc_info=True
                    )
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(
                            f"Too many consecutive failures ({consecutive_failures}). Exiting..."
                        )
                        break
                    wait_time = min(60 * consecutive_failures, 300)
                    self.logger.warning(
                        f"Will retry in {wait_time} seconds. Attempt {consecutive_failures}/{max_consecutive_failures}"
                    )
                    time_module.sleep(wait_time)
                    continue

                # Sleep for 1 minute before next check
                time_module.sleep(60)  # 60 seconds = 1 minute
            except Exception as e:
                self.logger.error(f"Error in paper trading: {str(e)}")
                time_module.sleep(60)  # 60 seconds = 1 minute

    def generate_performance_report(self):
        """Generate performance report"""
        if not self.trade_log:
            return "No trades executed yet."

        total_trades = len([t for t in self.trade_log if t["action"] == "EXIT"])
        winning_trades = len([t for t in self.trade_log if t.get("pnl", 0) > 0])
        total_pnl = sum([t.get("pnl", 0) for t in self.trade_log])

        report = f"""
Paper Trading Performance Report
------------------------------
Initial Balance: Rs. {self.initial_balance:,.2f}
Current Balance: Rs. {self.current_balance:,.2f}
Total Return: {((self.current_balance - self.initial_balance) / self.initial_balance * 100):.2f}%
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {(winning_trades/total_trades*100 if total_trades > 0 else 0):.1f}%
Total P&L: Rs. {total_pnl:,.2f}

Last 5 Trades:
"""

        last_5_trades = self.trade_log[-5:]
        for trade in last_5_trades:
            report += f"\nDate: {trade['date']}"
            report += f"\nAction: {trade['action']}"
            report += f"\nStrike: {trade['strike']}"
            report += f"\nPrice: {trade['price']}"
            report += f"\nP&L: {trade.get('pnl', 'N/A')}"
            report += f"\nReason: {trade['reason']}\n"

        return report
