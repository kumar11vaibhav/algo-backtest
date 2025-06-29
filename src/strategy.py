import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict, Tuple, Optional

# Technical indicators
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_moving_average(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

class Strategy:
    def __init__(self, data_handler, option_pricer, lot_size=75, risk_per_trade=0.01):
        """
        Initialize the strategy
        :param data_handler: DataHandler instance
        :param option_pricer: OptionPricer instance
        :param lot_size: Default lot size
        :param risk_per_trade: Percentage of account to risk per trade (0.01 = 1%)
        """
        self.data_handler = data_handler
        self.option_pricer = option_pricer
        self.lot_size = lot_size
        self.risk_per_trade = risk_per_trade
        self.positions = []
        self.results = []
        self.trade_log = []
        self.atr_period = 14
        self.rsi_period = 14
        self.ma_fast = 9
        self.ma_slow = 21
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.trailing_stop_pct = 0.05  # 5% trailing stop
        
    def calculate_indicators(self, data: pd.DataFrame) -> dict:
        """Calculate technical indicators"""
        return {
            'rsi': calculate_rsi(data['Close'], self.rsi_period),
            'atr': calculate_atr(data['High'], data['Low'], data['Close'], self.atr_period),
            'ma_fast': calculate_moving_average(data['Close'], self.ma_fast),
            'ma_slow': calculate_moving_average(data['Close'], self.ma_slow)
        }

    def _get_scalar(self, value, default=None):
        """Helper to safely get scalar value from pandas/numpy types"""
        try:
            if hasattr(value, 'iloc') and hasattr(value.iloc, '__getitem__'):
                return value.iloc[0] if len(value) > 0 else default
            return float(value) if value is not None and not pd.isna(value) else default
        except Exception:
            return default

    def check_entry_conditions(self, current_price: float, prev_data: pd.DataFrame, indicators: dict, idx: int) -> Tuple[bool, str]:
        """
        Check if entry conditions are met
        Returns: (should_enter, reason)
        """
        try:
            # Ensure we have valid index and data
            if idx < 0 or idx >= len(prev_data) or idx >= len(indicators['ma_fast']):
                return False, "Invalid index"
            
            # Safely get indicator values as scalars
            ma_fast = self._get_scalar(indicators['ma_fast'].iloc[idx])
            ma_slow = self._get_scalar(indicators['ma_slow'].iloc[idx])
            rsi = self._get_scalar(indicators['rsi'].iloc[idx])
            
            # Check for valid indicator values
            if ma_fast is None or ma_slow is None:
                return False, "Missing MA values"
                
            # Check for trend alignment (fast MA above slow MA)
            if ma_fast <= ma_slow:
                return False, f"Downtrend (Fast MA {ma_fast:.2f} <= Slow MA {ma_slow:.2f})"
                
            # Check RSI for overbought conditions
            if rsi is not None and rsi > 70:
                return False, f"Overbought (RSI: {rsi:.1f} > 70)"
                
            # Check for volume confirmation (if available)
            if 'Volume' in prev_data.columns and idx > 0 and len(prev_data) > idx:
                try:
                    current_vol = self._get_scalar(prev_data['Volume'].iloc[idx], 0)
                    prev_vol = self._get_scalar(prev_data['Volume'].iloc[idx-1], 0)
                    if current_vol < prev_vol:
                        return False, f"Decreasing volume ({current_vol:,.0f} < {prev_vol:,.0f})"
                except Exception as e:
                    # If volume check fails, continue with other checks
                    pass
            
            # Price action confirmation (close above fast MA)
            try:
                current_close = self._get_scalar(prev_data['Close'].iloc[idx])
                if current_close is not None and current_close < ma_fast:
                    return False, f"Price {current_close:.2f} below fast MA {ma_fast:.2f}"
            except Exception as e:
                # If price check fails, continue with other checks
                pass
                
            return True, "All conditions met"
            
        except Exception as e:
            error_msg = f"Error in check_entry_conditions: {str(e)}"
            if hasattr(self, 'option_pricer') and hasattr(self.option_pricer, 'logger'):
                self.option_pricer.logger.error(error_msg)
            else:
                print(error_msg)
            return False, error_msg
    
    def execute_trade(self, date, action, strike=None, price=None, pnl=0.0, reason=None):
        """
        Log a trade execution with proper PnL handling
        
        Args:
            date: Trade date
            action: Trade action (BUY/SELL/EXIT_AT_930)
            strike: Option strike price
            price: Execution price
            pnl: Profit and Loss (default: 0.0)
            reason: Reason for the trade
        """
        try:
            # Ensure numeric values
            price_float = float(price) if price is not None else 0.0
            pnl_float = float(pnl) if pnl is not None else 0.0
            strike_float = float(strike) if strike is not None else 0.0
            
            self.trade_log.append({
                'date': date,
                'action': action,
                'strike': strike_float,
                'price': price_float,
                'quantity': self.lot_size,
                'pnl': pnl_float,
                'reason': str(reason) if reason else ''
            })
        except Exception as e:
            self.option_pricer.logger.error(f"Error executing trade: {str(e)}")
    
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk per trade"""
        risk_amount = account_balance * self.risk_per_trade
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0
        return int(risk_amount / risk_per_share)
        
    def run(self, initial_balance=100000):
        """Run the strategy with enhanced risk management"""
        balance = initial_balance
        daily_balance = [initial_balance]
        daily_dates = [self.data_handler.get_date(0)]
        
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        break_even_trades = 0
        max_profit = 0
        max_loss = 0
        max_drawdown = 0
        peak_balance = initial_balance
        
        # Calculate indicators
        data = self.data_handler.get_all_data()
        indicators = self.calculate_indicators(data)
        
        for i in range(1, min(len(data), len(indicators['rsi']))):
            if i < max(self.atr_period, self.rsi_period, self.ma_slow):
                continue
                
            date = self.data_handler.get_date(i)
            spot = self.data_handler.get_close_price(i)
            prev_spot = self.data_handler.get_close_price(i-1)
            strike = self.data_handler.get_strike_price(prev_spot)
            
            # Check for end of day
            if len(daily_dates) == 0 or date.date() != daily_dates[-1].date():
                daily_balance.append(balance)
                daily_dates.append(date)
                
                # Check for max daily loss
                if len(daily_balance) > 1 and (daily_balance[-1] / daily_balance[-2] - 1) < -self.max_daily_loss:
                    self.option_pricer.logger.warning(f"Max daily loss reached at {date}")
                    break
            
            # Calculate historical volatility with a default value if calculation fails
            hist_vol = 0.15  # Default volatility
            try:
                hist_vol_series = self.data_handler.calculate_historical_volatility()
                if hist_vol_series is not None and not hist_vol_series.empty and i < len(hist_vol_series):
                    hist_vol = float(hist_vol_series.iloc[i]) if not pd.isna(hist_vol_series.iloc[i]) else hist_vol
            except Exception as e:
                self.option_pricer.logger.warning(f"Error getting historical volatility, using default: {str(e)}")
            
            # Calculate option prices
            current_option_price = 0
            try:
                current_option_price = self.option_pricer.calculate_price(
                    float(spot), float(strike), 30, float(hist_vol), option_type='call'
                )
            except Exception as e:
                self.option_pricer.logger.error(f"Error calculating option price: {str(e)}")
                continue
            
            # Check for stop loss in existing positions
            for pos in self.positions:
                current_price = self.option_pricer.calculate_price(
                    spot, pos['strike'], 
                    max(1, 30 - (len(self.results) - pos['entry_day'])),
                    hist_vol,
                    option_type='call'
                )
                
                # Calculate PnL for the position
                try:
                    entry_price = float(pos.get('entry_price', 0))
                    current_price_float = float(current_price) if current_price is not None else 0.0
                    position_size = int(pos.get('position_size', self.lot_size))
                    pnl = (current_price_float - entry_price) * position_size
                    
                    # Execute trade with calculated PnL
                    self.execute_trade(
                        date=date,
                        action='EXIT_AT_930',
                        strike=pos.get('strike'),
                        price=current_price_float,
                        pnl=pnl,
                        reason="Scheduled exit at 9:30 AM"
                    )
                except Exception as e:
                    self.option_pricer.logger.error(f"Error calculating PnL: {str(e)}")
                    continue
                
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
                
                should_enter, reason = self.check_entry_conditions(
                    current_option_price, data, indicators, i
                )
                
                try:
                    rsi_val = float(indicators['rsi'].iloc[i]) if not pd.isna(indicators['rsi'].iloc[i]) else 0
                    atr_val = float(indicators['atr'].iloc[i]) if not pd.isna(indicators['atr'].iloc[i]) else 0
                    ma_fast_val = float(indicators['ma_fast'].iloc[i]) if not pd.isna(indicators['ma_fast'].iloc[i]) else 0
                    ma_slow_val = float(indicators['ma_slow'].iloc[i]) if not pd.isna(indicators['ma_slow'].iloc[i]) else 0
                    
                    print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                    print(f"Previous close: {float(prev_option_price):.2f}" if not pd.isna(prev_option_price) else "Previous close: N/A")
                    print(f"Current price: {float(current_option_price):.2f}" if not pd.isna(current_option_price) else "Current price: N/A")
                    print(f"RSI: {rsi_val:.2f}")
                    print(f"ATR: {atr_val:.2f}")
                    print(f"Fast MA: {ma_fast_val:.2f}")
                    print(f"Slow MA: {ma_slow_val:.2f}")
                    print(f"Entry signal: {'Yes' if should_enter else 'No'} - {reason}")
                except Exception as e:
                    self.option_pricer.logger.error(f"Error printing diagnostics: {str(e)}")
                    
                if should_enter:
                    # Calculate position size based on risk
                    atr = indicators['atr'].iloc[i]
                    stop_loss = current_option_price - (atr * 1.5)  # 1.5x ATR stop
                    position_size = self.calculate_position_size(balance, current_option_price, stop_loss)
                    
                    if position_size > 0:
                        self.positions.append({
                            'entry_price': current_option_price,
                            'strike': strike,
                            'entry_day': len(self.results),
                            'entry_date': date,
                            'stop_loss': stop_loss,
                            'trailing_stop': current_option_price * (1 - self.trailing_stop_pct),
                            'position_size': position_size,
                            'max_price': current_option_price  # For trailing stop
                        })
                        self.execute_trade(
                            date, 'BUY', strike, current_option_price, 
                            reason=f"Entry signal: {reason}"
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
