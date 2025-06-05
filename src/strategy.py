from datetime import datetime, time
import pandas as pd

class Strategy:
    def __init__(self, data_handler, option_pricer, lot_size=75):
        self.data_handler = data_handler
        self.option_pricer = option_pricer
        self.lot_size = lot_size
        self.positions = []
        self.results = []
        self.trade_log = []
        
    def check_entry_conditions(self, current_price, prev_close):
        """Check if entry conditions are met"""
        trigger_price = prev_close * 1.5
        price_ratio = (current_price / trigger_price) * 100
        
        return current_price >= trigger_price, trigger_price, price_ratio
    
    def execute_trade(self, date, action, strike=None, price=None, pnl=None, reason=None):
        """Log a trade execution"""
        self.trade_log.append({
            'date': date,
            'action': action,
            'strike': strike,
            'price': price,
            'quantity': self.lot_size,
            'pnl': pnl,
            'reason': reason
        })
    
    def run(self, initial_balance=100000):
        """Run the strategy"""
        balance = initial_balance
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        break_even_trades = 0
        max_profit = 0
        max_loss = 0
        
        for i in range(1, self.data_handler.get_data_length()):
            date = self.data_handler.get_date(i)
            spot = self.data_handler.get_close_price(i)
            prev_spot = self.data_handler.get_close_price(i-1)
            strike = self.data_handler.get_strike_price(prev_spot)
            
            # Calculate historical volatility
            hist_vol = float(self.data_handler.calculate_historical_volatility().iloc[i])
            
            # Calculate option prices
            current_option_price = self.option_pricer.calculate_price(
                spot, strike, 30, hist_vol, option_type='call'
            )
            
            # Check for stop loss in existing positions
            for pos in self.positions:
                current_price = self.option_pricer.calculate_price(
                    spot, pos['strike'], 
                    max(1, 30 - (len(self.results) - pos['entry_day'])),
                    hist_vol,
                    option_type='call'
                )
                
                # Exit at 9:30 AM next day
                pnl = (current_price - pos['entry_price']) * self.lot_size
                self.execute_trade(date, 'EXIT_AT_930', pos['strike'], current_price, pnl)
                
                # Update statistics
                total_trades += 1
                if pnl > 100:
                    winning_trades += 1
                    max_profit = max(max_profit, pnl)
                elif pnl < -100:
                    losing_trades += 1
                    max_loss = min(max_loss, pnl)
                else:
                    break_even_trades += 1
                
                balance += pnl
            
            self.positions = []  # Clear positions after exit
            
            # Check entry conditions for new trades
            if not self.positions:
                prev_option_price = self.option_pricer.calculate_price(
                    prev_spot, strike, 30, hist_vol, option_type='call'
                )
                
                should_enter, trigger_price, price_ratio = self.check_entry_conditions(
                    current_option_price, prev_option_price
                )
                
                # Print diagnostic information
                print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                print(f"Previous close: {prev_option_price:.2f}")
                print(f"Trigger price: {trigger_price:.2f}")
                print(f"Current price: {current_option_price:.2f}")
                print(f"Price/Trigger ratio: {price_ratio:.1f}%")
                
                if should_enter:
                    self.positions.append({
                        'entry_day': len(self.results),
                        'entry_price': trigger_price,
                        'strike': strike
                    })
                    self.execute_trade(
                        date, 'BUY', strike, trigger_price,
                        reason=f"Trigger price {trigger_price:.2f} hit"
                    )
            
            # Store daily results
            self.results.append({
                'date': date,
                'spot': spot,
                'balance': balance
            })
        
        return pd.DataFrame(self.results), pd.DataFrame(self.trade_log), {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'break_even_trades': break_even_trades,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'final_balance': balance
        }
