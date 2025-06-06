import logging
import os
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up file handler with DEBUG level
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Set up console handler with INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler],
    force=True
)
logger = logging.getLogger('strategy')
logger.setLevel(logging.DEBUG)

# Strategy Parameters
MAX_RISK_PER_TRADE = 0.02      # 2% of account per trade
MAX_PORTFOLIO_RISK = 0.10      # 10% max portfolio risk
TRAILING_STOP_ATR = 2.5        # Trailing stop distance in ATRs
PROFIT_TARGET_ATR = 4.0        # Profit target in ATRs
RSI_OVERBOUGHT = 70            # Standard RSI overbought
RSI_OVERSOLD = 30              # Standard RSI oversold
MAX_HOLDING_DAYS = 10          # Increased to 10 days to let winners run
MAX_DAILY_DRAWDOWN = 0.10      # 10% max daily drawdown (increased from 5%)
MIN_WIN_RATE = 0.55            # Slightly reduced target win rate
MIN_PROFIT_FACTOR = 1.3        # Slightly reduced target profit factor

# Entry Condition Thresholds (Relaxed)
MIN_IV_PERCENTILE = 30         # Reduced from 40 for more opportunities
MAX_IV_PERCENTILE = 85         # Slightly increased from 80
MIN_VOLUME_RATIO = 1.0         # Reduced from 1.2 for more entries
MIN_EMA_SLOPE_DAYS = 3         # Reduced from 5 for more signals
MIN_ADX = 20                   # Reduced from 25 for more trend signals
MIN_EMA_SLOPE = 0.15           # Reduced from 0.2 for more trend signals
MIN_ENTRY_SCORE = 0.55         # Reduced from 0.7 for more entries

# Position Sizing
MAX_POSITION_SIZE = 0.15       # Increased from 0.1 to allow larger positions
SCALING_FACTOR = 0.5           # Scale in/out factor
RISK_REWARD_RATIO = 2.0        # Minimum reward:risk ratio

# Position Sizing
MAX_POSITION_SIZE = 0.1        # Max 10% of portfolio in single trade
SCALING_FACTOR = 0.5           # Scale in/out factor

# Market Regime Parameters
TREND_CONFIRMATION_DAYS = 3    # Days to confirm trend
RANGE_THRESHOLD = 0.3          # ATR % threshold for range-bound market

class Strategy:
    def __init__(self, data_handler, option_pricer, initial_balance=100000):
        self.data_handler = data_handler
        self.option_pricer = option_pricer
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trade_log = []
        self.daily_stats = {}  # Initialize empty dictionary for daily stats
        self.current_date = None
        self.indicators_setup = False
        self.peak_balance = initial_balance  # Track peak balance for drawdown calculation
        self.daily_high_balance = initial_balance  # Track daily high balance for daily drawdown
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize trade statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.break_even_trades = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.max_profit = float('-inf')
        self.max_loss = float('inf')
        self.current_day = None  # Track current trading day
        self.results = []  # Initialize results list for storing backtest results

    def _setup_indicators(self):
        """Initialize enhanced technical indicators"""
        try:
            df = self.data_handler.data
            
            # Calculate EMAs
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
            
            # Calculate RSI with different periods
            for period in [14, 21]:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate ADX for trend strength
            plus_dm = df['High'].diff()
            minus_dm = -df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            df['ADX'] = dx.rolling(window=14).mean()
            
            # Volume analysis
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            # Volatility analysis
            df['IV_Percentile'] = df['IV'].rolling(window=252).rank(pct=True) * 100
            df['ATR'] = tr.rolling(window=14).mean()
            df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
            
            # Support/Resistance levels
            df['20d_high'] = df['High'].rolling(window=20).max()
            df['20d_low'] = df['Low'].rolling(window=20).min()
            
            self.indicators_setup = True
            return True
            
        except Exception as e:
            logging.error(f"Error setting up indicators: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
        
    def _identify_market_regime(self, idx, lookback=20):
        """Identify if market is trending or range-bound"""
        if idx < lookback:
            return 'neutral'
            
        data = self.data_handler.data.iloc[idx-lookback:idx+1]
        
        # Calculate ATR and price range
        atr = data['ATR'].iloc[-1]
        price_range = data['High'].max() - data['Low'].min()
        range_atr_ratio = price_range / atr
        
        # Check trend using ADX
        adx = data['ADX'].iloc[-1]
        
        if adx > 30 and range_atr_ratio > 5:
            return 'trending'
        elif adx < 20 and range_atr_ratio < 3:
            return 'ranging'
        return 'neutral'
    
    def _calculate_entry_score(self, current_bar, prev_bar, idx):
        """Calculate entry score based on multiple factors with detailed logging"""
        score = 0
        reasons = []
        score_details = {}
        
        # 0. Basic price data
        close = float(current_bar['Close'])
        high = float(current_bar['High'])
        low = float(current_bar['Low'])
        volume = float(current_bar['Volume'])
        
        # 1. Trend Strength (30% weight)
        ema20 = float(current_bar['EMA20'])
        ema50 = float(current_bar['EMA50'])
        ema100 = float(current_bar['EMA100'])
        ema200 = float(current_bar.get('EMA200', ema100))
        
        # Calculate EMA slopes
        ema20_slope = (ema20 / float(self.data_handler.data['EMA20'].iloc[max(0, idx-3)])) - 1
        ema50_slope = (ema50 / float(self.data_handler.data['EMA50'].iloc[max(0, idx-3)])) - 1
        
        # Trend conditions
        ema_alignment = ema20 > ema50 > ema100
        ema_slope_positive = ema20_slope > MIN_EMA_SLOPE and ema50_slope > MIN_EMA_SLOPE/2
        price_above_ema = close > ema20 > ema50
        
        trend_score = 0
        if ema_alignment and ema_slope_positive and price_above_ema:
            trend_score = 1
            reasons.append('strong_uptrend')
        elif ema_alignment and ema_slope_positive:
            trend_score = 0.7
            reasons.append('moderate_uptrend')
        elif price_above_ema and ema_slope_positive:
            trend_score = 0.5
            reasons.append('weak_uptrend')
            
        score += trend_score * 0.3
        score_details['trend'] = f"{trend_score:.2f} ({', '.join([r for r in reasons if 'trend' in r]) or 'none'})"
        
        # 2. Momentum (25% weight)
        rsi = float(current_bar.get('RSI_14', 50))
        macd = float(current_bar.get('MACD', 0))
        signal = float(current_bar.get('Signal_Line', 0))
        macd_hist = macd - signal
        
        momentum_score = 0
        momentum_reasons = []
        
        # RSI conditions
        if 40 < rsi < 70:  # More lenient RSI range
            momentum_score += 0.4
            momentum_reasons.append('good_rsi')
            
        # MACD conditions
        if macd > signal and macd_hist > 0:
            momentum_score += 0.3
            momentum_reasons.append('bullish_macd')
            
        # Recent price action
        if close > float(prev_bar['Close']) * 1.01:  # 1% up move
            momentum_score += 0.3
            momentum_reasons.append('strong_move')
            
        score += (momentum_score / 3) * 0.25  # Normalize to 0-0.25 range
        score_details['momentum'] = f"{momentum_score:.2f} ({', '.join(momentum_reasons) or 'none'})"
        
        # 3. Volume Confirmation (20% weight)
        volume_ma20 = float(current_bar['Volume_MA20'])
        volume_ratio = volume / volume_ma20 if volume_ma20 > 0 else 1.0
        
        volume_score = min(1.0, (volume_ratio - 0.8) / 0.5)  # Scale from 0.8x to 1.3x
        volume_score = max(0, min(1, volume_score))  # Clamp to 0-1 range
        
        if volume_ratio > 1.0:
            reasons.append(f'volume_{volume_ratio:.1f}x')
            
        score += volume_score * 0.2
        score_details['volume'] = f"{volume_score:.2f} ({volume_ratio:.1f}x)"
        
        # 4. Volatility (15% weight)
        iv_percentile = self.calculate_iv_percentile(idx)
        atr = float(current_bar['ATR'])
        atr_percent = (atr / close) * 100
        
        volatility_score = 0
        volatility_reasons = []
        
        # IV Percentile conditions
        if 25 <= iv_percentile <= 85:  # Wider IV range
            volatility_score += 0.5
            volatility_reasons.append(f'iv_{iv_percentile:.0f}%')
            
        # ATR conditions
        if 1.0 <= atr_percent <= 5.0:  # Wider ATR range
            volatility_score += 0.5
            volatility_reasons.append(f'atr_{atr_percent:.1f}%')
            
        score += (volatility_score / 2) * 0.15  # Normalize to 0-0.15 range
        score_details['volatility'] = f"{volatility_score:.2f} ({', '.join(volatility_reasons)})"
        
        # 5. Price Action (10% weight)
        support = float(current_bar['20d_low'])
        resistance = float(current_bar['20d_high'])
        
        price_action_score = 0
        price_action_reasons = []
        
        # Support/Resistance conditions
        support_dist = (close - support) / close
        resist_dist = (resistance - close) / close
        
        if support_dist < 0.01:  # Near support
            price_action_score += 0.7
            price_action_reasons.append('near_support')
        elif resist_dist < 0.01:  # Near resistance
            price_action_score -= 0.3
            price_action_reasons.append('near_resistance')
            
        # Recent price action
        if close > float(prev_bar['Close']):
            price_action_score += 0.3
            price_action_reasons.append('up_day')
            
        score += max(0, price_action_score) * 0.1  # Don't let resistance hurt the score too much
        score_details['price_action'] = f"{price_action_score:.2f} ({', '.join(price_action_reasons) or 'neutral'})"
        
        # Log detailed score breakdown
        if score >= 0.5:  # Only log for potential entries to reduce noise
            logger.info(f"\n--- Entry Score Details ---")
            for factor, detail in score_details.items():
                logger.info(f"{factor.upper():<12}: {detail}")
            logger.info(f"{'TOTAL SCORE':<12}: {score:.2f} (Threshold: {MIN_ENTRY_SCORE})")
            
            try:
                option_result = self.option_pricer.calculate_price(
                    current_bar['Close'], 
                    strike, 
                    30,  # Days to expiry
                    current_bar['IV'],
                    option_type='call'
                )
                option_price = option_result['price']  # Extract just the price from the result
                self.logger.info(f"Calculated option price: {option_price:.2f} (Strike: {strike}, IV: {current_bar['IV']:.4f})")
                
                if option_price > 0:  # Valid price
                    position = {
                        'entry_day': i,
                        'entry_price': option_price,
                        'strike': strike,
                        'quantity': position_size,
                        'stop_loss': option_price * (1 - STOP_LOSS_PCT),
                        'take_profit': option_price * (1 + TARGET_PCT),
                        'entry_iv': current_bar['IV'],
                        'entry_time': current_date
                    }
                    # Debug logging
                    self.logger.info(f"Debug - current_bar RSI_14 type: {type(current_bar.get('RSI_14'))}")
                    self.logger.info(f"Debug - current_bar keys: {list(current_bar.keys())}")
                    
                    # Safely get RSI value
                    rsi_value = current_bar.get('RSI_14', 0)
                    if isinstance(rsi_value, dict):
                        rsi_value = rsi_value.get('value', 0)
                        
                    # Safely get IV percentile from current bar
                    iv_pct = current_bar.get('IV_Percentile', 0)
                    if isinstance(iv_pct, dict):
                        iv_pct = iv_pct.get('value', 0)
                        
                    self.positions.append(position)
                    
                    # Process the position data
                    iv_pct = float(iv_pct)
                    reason = f"Entry: IV% {iv_pct:.1f}, RSI {float(rsi_value):.1f}"
                    # Execute the trade with proper parameters
                    self.execute_trade(
                        date=current_date,
                        action='BUY',
                        strike=strike,
                        price=option_price,
                        quantity=position_size,
                        pnl=0,  # Initial PnL is 0 for new positions
                        reason=reason
                    )
                    
                    stats['total_trades'] += 1
                    self.logger.info(f"New position created: {reason}")
                    
            except Exception as e:
                self.logger.error(f"Error calculating option price: {str(e)}")
                self.logger.error(f"Parameters - Close: {current_bar['Close']}, Strike: {strike}, IV: {current_bar['IV']}")
        
        # Close any remaining positions at the end
        if self.positions:
            last_bar = self.data_handler.data.iloc[-1]
            self._close_all_positions(len(self.data_handler.data) - 1, stats)
        
        # Calculate final statistics
        stats['final_balance'] = self.balance
        stats['total_return'] = (self.balance / self.initial_balance - 1) * 100
        
        # Convert trade log to DataFrame
        trade_log_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        
        # Create results DataFrame with daily stats
        try:
            # Ensure we have valid data to create the DataFrame
            if not hasattr(self.data_handler, 'data') or self.data_handler.data is None:
                logger.error("No data available to create results DataFrame")
                return pd.DataFrame(), pd.DataFrame(), stats
            
            # Get the data and ensure it's a DataFrame
            data = self.data_handler.data
            if not isinstance(data, pd.DataFrame):
                logger.error("Data handler data is not a DataFrame")
                return pd.DataFrame(), pd.DataFrame(), stats
                
            # Ensure we have the required columns
            required_columns = ['Close']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                logger.error(f"Missing required columns in data: {missing}")
                return pd.DataFrame(), pd.DataFrame(), stats
            
            # Create a simple results DataFrame with just the close prices and balance
            results_df = pd.DataFrame(index=data.index)
            results_df['close'] = data['Close']
            results_df['balance'] = self.initial_balance  # Will be updated with actual values
            
            # Ensure all numeric columns are float type
            results_df = results_df.astype(float, errors='ignore')
            
            logger.info(f"Created results DataFrame with {len(results_df)} rows")
            
        except Exception as e:
            logger.error(f"Error creating results DataFrame: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(), pd.DataFrame(), stats
        
        return results_df, trade_log_df, stats
        
    def _manage_existing_positions(self, idx, stats, current_bar, current_time):
        """Enhanced position management with dynamic exits and scaling"""
            exit_reason = None
            exit_price = None
            exit_quantity = position_size  # Default to full position
            
            # 1. Hard Stop Loss (emergency)
            max_loss = entry_price * (1 - position['stop_loss_pct']/100 * 1.5)  # 1.5x ATR stop
            if current_price <= max_loss:
                exit_reason = 'EMERGENCY_STOP'
                exit_price = max_loss
            
            # 2. Trailing Stop (normal operation)
            elif current_price <= position['trailing_stop']:
                exit_reason = 'TRAILING_STOP'
                exit_price = position['trailing_stop']
            
            # 3. Take Profit Scaling (partial exits)
            elif current_price >= entry_price * (1 + position['take_profit_pct']/100 * 0.5):  # 50% of target
                if 'partial_exit' not in position:
                    # Take 50% profit at first target
                    exit_reason = 'PROFIT_TARGET_50'
                    exit_price = current_price
                    exit_quantity = int(position_size * 0.5)
                    position['partial_exit'] = True
                    position['take_profit_pct'] *= 1.5  # Move target higher for remaining
                elif current_price >= entry_price * (1 + position['take_profit_pct']/100):
                    # Take remaining at final target
                    exit_reason = 'PROFIT_TARGET_100'
                    exit_price = current_price
            
            # 4. Time-based exit (after 2x average holding period)
            elif days_in_trade >= MAX_HOLDING_DAYS * 2:
                exit_reason = 'MAX_HOLDING_PERIOD'
                exit_price = current_price
            
            # Execute exit if any condition met
            if exit_reason:
                self._close_position(
                    position=position,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    current_date=current_date,
                    stats=stats,
                    exit_quantity=exit_quantity
                )
                
                # Update position size if partial exit
                if exit_quantity < position_size and exit_reason == 'PROFIT_TARGET_50':
                    position['quantity'] -= exit_quantity
                    position['price'] = (position['price'] * position_size - exit_price * exit_quantity) / (position_size - exit_quantity)
                    position['entry_value'] = position['price'] * position['quantity']
                    logger.info(f"Partial exit: Took 50% profit at {exit_price:.2f}, new position: {position['quantity']} @ {position['price']:.2f}")
                    
    def _close_position(self, position, exit_price, exit_reason, current_date, stats, exit_quantity=None):
        """Close a position (fully or partially) and update statistics"""
        if exit_quantity is None:
            exit_quantity = position['quantity']
            
        entry_price = position['price']
        position_size = position['quantity']
        
        # Ensure we don't close more than we have
        exit_quantity = min(exit_quantity, position_size)
        
        # Calculate P&L for this exit
        pnl = (exit_price - entry_price) * exit_quantity
        pnl_pct = (exit_price / entry_price - 1) * 100
        
        # Calculate holding period
        holding_days = (current_date - position['entry_time']).days + 1
        annualized_return = ((1 + pnl_pct/100) ** (365/holding_days) - 1) * 100 if holding_days > 0 else 0
        
        # Update account balance
        self.balance += pnl
        
        # Update trade statistics
        self.total_trades += 1 if exit_quantity == position_size else 0.5
        self.total_pnl += pnl
        
        # Update win/loss statistics
        if pnl > 0:
            self.winning_trades += 1 if exit_quantity == position_size else 0.5
            self.max_profit = max(self.max_profit, pnl)
        elif pnl < 0:
            self.losing_trades += 1 if exit_quantity == position_size else 0.5
            self.max_loss = min(self.max_loss, pnl)
        else:
            self.break_even_trades += 1 if exit_quantity == position_size else 0.5
            
        # Update stats dictionary
        stats['last_balance'] = self.balance
        stats['peak_balance'] = self.peak_balance
        stats['max_drawdown'] = max(stats.get('max_drawdown', 0.0), drawdown)
        if 'daily_returns' not in stats:
            stats['daily_returns'] = []
        stats['daily_returns'].append(daily_return)
        
        # Log daily stats
        self.results.append({
            'date': date,
            'balance': self.balance,
            'drawdown': drawdown,
            'daily_return': daily_return,
            'positions': len(self.positions),
            'high_price': float(current_bar['High']),
            'low_price': float(current_bar['Low']),
            'close_price': float(current_bar['Close']),
            'volume': float(current_bar['Volume'])
        })
        
    def update_positions(self, current_bar, current_date, stats):
        """
        Update all open positions (check for stop loss, take profit, etc.)
        
        Args:
            current_bar: Current market data
            current_date: Current date
            stats: Statistics dictionary to update
        """
        if not hasattr(self, 'positions') or not self.positions:
            return
            
        for position in self.positions[:]:
            if position.get('status') != 'OPEN':
                continue
                
            # Get current price (using close price as an approximation)
            current_price = current_bar.get('Close', 0)
            if not current_price:
                self.logger.warning(f"No valid price in current bar: {current_bar}")
                continue
                
            # Check for stop loss / take profit
            pnl_pct = (current_price / position['entry_price'] - 1) * 100
            
            # Update running PnL for the position
            position['current_pnl'] = pnl_pct
            
            # Check exit conditions
            exit_reason = None
            
            # Check stop loss
            if pnl_pct <= -position.get('stop_loss_pct', 5.0):
                exit_reason = 'STOP_LOSS'
            # Check take profit
            elif pnl_pct >= position.get('take_profit_pct', 10.0):
                exit_reason = 'TAKE_PROFIT'
            # Check time-based exit (e.g., end of day)
            elif (current_date - position['entry_date']).days >= position.get('max_holding_days', 5):
                exit_reason = 'TIME_EXIT'
                
            # Close position if exit condition met
            if exit_reason:
                self._close_position(
                    position=position,
                    exit_price=current_price,
                    exit_reason=exit_reason,
                    current_date=current_date,
                    stats=stats
                )

    def _update_trade_stats(self, stats, pnl):
        """Update trade statistics with enhanced metrics"""
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        
        # Update running win rate and profit factor
        win_rate = (stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
        profit_factor = (stats['total_profits'] / abs(stats['total_losses'])) if stats['total_losses'] < 0 else float('inf')
        
        # Update max win/loss streaks
        if pnl > 0:
            stats['winning_trades'] += 1
            stats['total_profits'] += pnl
            stats['current_win_streak'] += 1
            stats['current_loss_streak'] = 0
            stats['max_win_streak'] = max(stats['max_win_streak'], stats['current_win_streak'])
            stats['max_profit'] = max(stats['max_profit'], pnl)
            stats['avg_win'] = stats['total_profits'] / stats['winning_trades'] if stats['winning_trades'] > 0 else 0
            
        elif pnl < 0:
            stats['losing_trades'] += 1
            stats['total_losses'] += pnl
            stats['current_loss_streak'] += 1
            stats['current_win_streak'] = 0
            stats['max_loss_streak'] = max(stats['max_loss_streak'], stats['current_loss_streak'])
            stats['max_loss'] = min(stats['max_loss'], pnl)
            stats['avg_loss'] = stats['total_losses'] / stats['losing_trades'] if stats['losing_trades'] > 0 else 0
        else:
            stats['break_even_trades'] += 1
            
        # Calculate risk/reward metrics
        if stats['losing_trades'] > 0:
            stats['profit_factor'] = abs(stats['total_profits'] / stats['total_losses'])
            stats['expectancy'] = (stats['winning_trades'] * stats['avg_win'] + 
                                 stats['losing_trades'] * stats['avg_loss']) / stats['total_trades']
    
    def _update_daily_stats(self, date, current_bar, stats):
        """Update daily performance metrics with enhanced analytics"""
        try:
            # Initialize metrics if not exists
            if 'last_balance' not in stats:
                stats.update({
                    'last_balance': self.initial_balance,
                    'peak_balance': self.initial_balance,
                    'max_drawdown': 0,
                    'max_drawdown_pct': 0,
                    'winning_days': 0,
                    'losing_days': 0,
                    'total_days': 0,
                    'avg_daily_return': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'daily_returns': []
                })
            
            # Calculate daily P&L and returns
            daily_pnl = self.balance - stats['last_balance']
            daily_return = (daily_pnl / stats['last_balance']) * 100 if stats['last_balance'] > 0 else 0
            
            # Update peak balance and drawdown
            self.peak_balance = max(self.peak_balance, self.balance)
            drawdown = self.peak_balance - self.balance
            drawdown_pct = (drawdown / self.peak_balance * 100) if self.peak_balance > 0 else 0
            
            # Update max drawdown
            stats['max_drawdown'] = max(stats['max_drawdown'], drawdown)
            stats['max_drawdown_pct'] = max(stats['max_drawdown_pct'], drawdown_pct)
            
            # Update win/loss days
            stats['total_days'] += 1
            if daily_pnl > 0:
                stats['winning_days'] += 1
            elif daily_pnl < 0:
                stats['losing_days'] += 1
                
            # Track daily returns for volatility metrics
            stats['daily_returns'].append(daily_return)
            
            # Calculate performance metrics
            if len(stats['daily_returns']) > 1:
                returns = np.array(stats['daily_returns'])
                avg_return = np.mean(returns)
                std_dev = np.std(returns)
                downside_returns = returns[returns < 0]
                
                # Sharpe Ratio (assuming 0% risk-free rate)
                stats['sharpe_ratio'] = (avg_return / std_dev * np.sqrt(252)) if std_dev > 0 else 0
                
                # Sortino Ratio
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                stats['sortino_ratio'] = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
                
                # Average daily return
                stats['avg_daily_return'] = avg_return
            
            # Update stats dictionary
            stats.update({
                'last_balance': self.balance,
                'peak_balance': self.peak_balance,
                'drawdown': drawdown,
                'drawdown_pct': drawdown_pct,
                'win_rate': (stats['winning_days'] / stats['total_days']) * 100 if stats['total_days'] > 0 else 0,
                'profit_factor': abs(stats['total_profits'] / stats['total_losses']) if stats['total_losses'] < 0 else float('inf')
            })
            
            # Update daily stats
            if date not in self.daily_stats:
                self.daily_stats[date] = {}
                
            self.daily_stats[date].update({
                'balance': self.balance,
                'pnl': daily_pnl,
                'return_pct': daily_return,
                'drawdown_pct': drawdown_pct,
                'open_trades': len(self.positions),
                'win_rate': stats['win_rate'],
                'sharpe': stats['sharpe_ratio'],
                'sortino': stats['sortino_ratio']
            })
            
            # Log daily stats
            self.logger.info(
                f"Daily Stats - Balance: {self.balance:,.2f}, "
                f"Daily P&L: {daily_pnl:+,.2f} ({daily_return:+.2f}%), "
                f"Drawdown: {drawdown_pct:.2f}%, "
                f"Win Rate: {stats['win_rate']:.1f}%, "
                f"Sharpe: {stats['sharpe_ratio']:.2f}"
            )
                
        # Close position if exit condition met
        if exit_reason:
            self._close_position(position, current_price, exit_reason, current_date, stats)
        self.positions = []
        
        # Calculate final statistics
        stats = {
            'total_trades': self.winning_trades + self.losing_trades + self.break_even_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'break_even_trades': self.break_even_trades,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'final_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0
        }
        
        # Ensure we have results to return
        if not hasattr(self, 'results') or not self.results:
            self.results = [{'date': datetime.now().strftime('%Y-%m-%d'), 'balance': self.balance}]
            
        if not hasattr(self, 'trade_log') or not self.trade_log:
            self.trade_log = []
            
        return pd.DataFrame(self.results), pd.DataFrame(self.trade_log), stats
