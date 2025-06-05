import yfinance as yf
import pandas as pd
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

class NiftyOptionsBacktester:
    def __init__(self):
        self.nifty_data = None
        self.symbol = "^NSEI"  # Nifty 50 index symbol
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.trading_time = time(9, 30)  # 9:30 AM IST
        self.ist_tz = ZoneInfo('Asia/Kolkata')
        self.lot_size = 50  # Nifty options lot size
        self.base_volatility = 0.15  # Base annualized volatility
        self.interest_rate = 0.07  # 7% risk-free rate
        
    def fetch_historical_data(self, start_date, end_date):
        """Fetch historical Nifty data from Yahoo Finance"""
        try:
            self.nifty_data = yf.download(self.symbol, start=start_date, end=end_date)
            print(f"Successfully downloaded {len(self.nifty_data)} days of data")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def calculate_historical_volatility(self, lookback=30):
        """Calculate historical volatility using daily returns"""
        returns = np.log(self.nifty_data['Close'] / self.nifty_data['Close'].shift(1))
        vol = returns.rolling(window=lookback).std() * np.sqrt(252)
        # Fill NaN values with base_volatility
        vol = vol.fillna(self.base_volatility)
        return vol

    def calculate_option_price(self, spot, strike, days_to_expiry, hist_vol=None, option_type='call'):
        """Calculate theoretical option price using Black-Scholes formula with dynamic volatility"""
        S = spot
        K = strike
        T = days_to_expiry / 365
        r = self.interest_rate

        # Use historical volatility if provided, otherwise use base volatility
        sigma = hist_vol if hist_vol is not None else self.base_volatility

        # Adjust volatility based on moneyness and time to expiry
        moneyness = K/S
        if moneyness < 0.95 or moneyness > 1.05:  # OTM options
            sigma *= 1.1  # Increase volatility for OTM options

        # Minimum volatility floor
        sigma = max(sigma, 0.1)

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type.lower() == 'call':
            option_price = S*norm_cdf(d1) - K*np.exp(-r*T)*norm_cdf(d2)
        else:  # put
            option_price = K*np.exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)
            
        return max(option_price, 0.01)  # Minimum price of 0.01

    def get_one_strike_below_atm(self, spot_price, step=100):
        """Get strike price one floor below ATM"""
        atm_strike = round(spot_price / step) * step
        return atm_strike - step

    def backtest_strategy(self, start_date, end_date, strategy_params):
        """Run backtesting for the specified strategy"""
        if not self.fetch_historical_data(start_date, end_date):
            return None

        results = []
        balance = strategy_params.get('initial_capital', 100000)
        positions = []
        prev_day_option_price = None
        trade_log = []
        pending_orders = []  # Store orders to be executed next day

        for i in range(1, len(self.nifty_data)):
            date = self.nifty_data.index[i]
            spot = self.nifty_data.iloc[i]['Close'].iloc[0]
            
            # Convert to IST for day of week check
            ist_datetime = date.tz_localize('UTC').astimezone(self.ist_tz)
            is_thursday = ist_datetime.weekday() == 3  # 3 represents Thursday
            
            # Get previous day's data
            prev_spot = self.nifty_data.iloc[i-1]['Close'].iloc[0]
            strike = self.get_one_strike_below_atm(prev_spot)  # One strike below ATM based on previous day's close
            
            # Calculate historical volatility
            hist_vol = float(self.calculate_historical_volatility().iloc[i])
            
            # Calculate option prices
            current_option_price = self.calculate_option_price(
                float(spot), float(strike), 30, hist_vol, option_type='call'
            )
            
            # Print diagnostic information
            if i % 20 == 0:
                print(f"\nDiagnostic - Date: {ist_datetime.strftime('%Y-%m-%d %H:%M')} IST")
                print(f"Spot: {spot:.2f}, Strike: {strike}")
                print(f"Option Price: {current_option_price:.2f}")
            
            # Handle existing positions - Check for stop loss and exit at 9:30 AM
            positions_to_remove = []
            for pos in positions:
                current_price = self.calculate_option_price(
                    spot, pos['strike'], 
                    max(1, 30 - (len(results) - pos['entry_day'])),
                    self.base_volatility, 
                    option_type='call'
                )
                
                # Calculate P&L percentage
                pnl_percent = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                
                # Exit conditions: Stop loss (-15%) or next day 9:30 AM
                days_held = len(results) - pos['entry_day']
                if pnl_percent <= -15 or days_held >= 1:
                    profit = (current_price - pos['entry_price']) * pos['quantity']
                    balance += (pos['entry_price'] + profit) * pos['quantity']
                    positions_to_remove.append(pos)
                    
                    # Update trade statistics
                    if profit > 0:
                        self.winning_trades += 1
                        self.total_profit += profit
                    else:
                        self.losing_trades += 1
                        self.total_loss += profit

                    trade_log.append({
                        'date': date,
                        'action': 'STOP_LOSS' if pnl_percent <= -15 else 'EXIT_AT_930',
                        'strike': pos['strike'],
                        'entry': pos['entry_price'],
                        'exit': current_price,
                        'pnl': profit,
                        'trade_number': pos.get('trade_number', 0)
                    })
            
            # Remove stopped out positions
            for pos in positions_to_remove:
                positions.remove(pos)
            
            # Entry logic - Place order at 9:30 AM if not Thursday
            if not positions and prev_day_option_price is not None:
                trigger_price = prev_day_option_price * 1.5  # 50% more than previous day's close
                
                # Print diagnostic info for potential trades
                if i % 5 == 0:  # Print every 5th day to avoid too much output
                    print(f"\nDate: {ist_datetime.strftime('%Y-%m-%d')}")
                    print(f"Previous close: {prev_day_option_price:.2f}")
                    print(f"Trigger price: {trigger_price:.2f}")
                    print(f"Current price: {current_option_price:.2f}")
                    print(f"Price/Trigger ratio: {(current_option_price/trigger_price*100):.1f}%")
                    
                # If current price hits trigger price, enter position
                if current_option_price >= trigger_price:
                    if trigger_price * self.lot_size <= balance:  # Check if we have enough balance
                        positions.append({
                            'type': 'call',
                            'strike': strike,
                            'entry_price': trigger_price,
                            'quantity': self.lot_size,  # Trading 1 lot
                            'entry_day': len(results)
                        })
                        balance -= trigger_price * self.lot_size
                        self.total_trades += 1
                        trade_log.append({
                            'date': date,
                            'action': 'BUY',
                            'strike': strike,
                            'price': trigger_price,
                            'reason': f'Trigger price {trigger_price:.2f} hit',
                            'trade_number': self.total_trades
                        })
                        print(f"Trade entered on {ist_datetime.strftime('%Y-%m-%d')} at strike {strike}, price {trigger_price:.2f}")

            # Remove positions that were stopped out or exited
            positions = [p for p in positions if p not in positions_to_remove]

            # Store current option price for next day's reference
            prev_day_option_price = current_option_price

            # Store daily results
            results.append({
                'date': date,
                'balance': balance,
                'positions': len(positions),
                'strike': strike,
                'option_price': current_option_price,
                'trigger_price': prev_day_option_price * 1.5 if prev_day_option_price else None
            })

        results_df = pd.DataFrame(results).set_index('date')
        trade_log_df = pd.DataFrame(trade_log)
        
        return results_df, trade_log_df

def norm_cdf(x):
    """Calculate cumulative distribution function for standard normal"""
    return (1.0 + special.erf(x / np.sqrt(2.0))) / 2.0

if __name__ == "__main__":
    # Initialize backtester
    backtester = NiftyOptionsBacktester()
    
    # Set up strategy parameters
    strategy_params = {
        'initial_capital': 100000
    }
    
    # Run backtest for last 1 year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    results_df, trade_log_df = backtester.backtest_strategy(start_date, end_date, strategy_params)
    
    if results_df is not None:
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(results_df.index, results_df['balance'])
        plt.title('ATM Call Option Strategy - Account Balance')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.show()
        
        # Print summary
        initial_balance = strategy_params['initial_capital']
        final_balance = results_df['balance'].iloc[-1]
        returns = (final_balance - initial_balance) / initial_balance * 100
        
        print(f"\nBacktesting Results Summary:")
        print(f"Initial Balance: Rs. {initial_balance:,.2f}")
        print(f"Final Balance: Rs. {final_balance:,.2f}")
        print(f"Total Return: {returns:.2f}%")
        
        if len(trade_log_df) > 0:
            print(f"\nTrade Statistics:")
            print(f"Total Trades: {len(trade_log_df)}")
            
            # Calculate win rate
            profitable_trades = trade_log_df[trade_log_df['pnl'] > 0]
            win_rate = len(profitable_trades) / len(trade_log_df) * 100
            print(f"Win Rate: {win_rate:.1f}%")
            
            # Detailed statistics
            avg_profit = trade_log_df['pnl'].mean()
            max_profit = trade_log_df['pnl'].max()
            max_loss = trade_log_df['pnl'].min()
            print(f"Average P&L per trade: Rs. {avg_profit:,.2f}")
            print(f"Largest Profit: Rs. {max_profit:,.2f}")
            print(f"Largest Loss: Rs. {max_loss:,.2f}")
            
            # Calculate profit factor
            winning_trades = trade_log_df[trade_log_df['pnl'] > 0]
            losing_trades = trade_log_df[trade_log_df['pnl'] < 0]
            total_profits = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_profits / total_losses if total_losses != 0 else float('inf')
            print(f"Profit Factor: {profit_factor:.2f}")
            
            # Print trade distribution
            print(f"\nTrade Distribution:")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            print(f"Break-even Trades: {len(trade_log_df) - len(winning_trades) - len(losing_trades)}")
            
            # Print last 5 trades
            print(f"\nLast 5 Trades:")
            if 'reason' in trade_log_df.columns:
                print(trade_log_df[['date', 'action', 'strike', 'entry', 'exit', 'pnl', 'reason']].tail())
            else:
                print(trade_log_df[['date', 'action', 'strike', 'entry', 'exit', 'pnl']].tail())
