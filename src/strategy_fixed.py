import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy.stats import linregress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Strategy:
    def __init__(self, data_handler, option_pricer, initial_balance=100000.0):
        """
        Initialize the Strategy with data handler, option pricer, and initial balance.
        
        Args:
            data_handler: Instance of DataHandler for market data
            option_pricer: Instance of OptionPricer for pricing options
            initial_balance: Initial account balance (default: 100,000)
        """
        self.data_handler = data_handler
        self.option_pricer = option_pricer
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # List to track open positions
        self.trade_log = []
        self.current_date = None
        self.daily_high_balance = initial_balance
        self.indicators_setup = False
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('backtest_debug.log', mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize logger for this instance
        self.logger = logging.getLogger('Strategy')
        
        # Set console output to UTF-8
        import sys
        import codecs
        if sys.stdout.encoding != 'UTF-8':
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if sys.stderr.encoding != 'UTF-8':
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
            
        # Log initialization
        logger.info("Strategy initialized")
        if hasattr(self.data_handler, 'data') and self.data_handler.data is not None:
            logger.info(f"DataFrame head:\n{self.data_handler.data.head()}")
            logger.info(f"DataFrame columns: {self.data_handler.data.columns.tolist()}")
            logger.info(f"DataFrame index: {self.data_handler.data.index}")
            if len(self.data_handler.data) > 0:
                logger.info(f"First row as dict: {self.data_handler.data.iloc[0].to_dict()}")
            
    def run(self):
        """
        Execute the backtest and return results, trade log, and statistics.
        
        Returns:
            tuple: (results_df, trade_log_df, stats)
        """
        logger.info("Starting backtest...")
        
        # Initialize statistics
        stats = {
            'initial_balance': self.initial_balance,
            'final_balance': self.initial_balance,
            'total_return': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'break_even_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'avg_trade_duration': 0.0,
            'total_days': 0,
            'trading_days_per_year': 252,
            'risk_free_rate': 0.05  # 5% annual risk-free rate
        }
        
        # Initialize trade log and results
        trade_log = []
        results = []
        
        # Ensure positions list exists
        if not hasattr(self, 'positions'):
            self.positions = []
        
        try:
            # Set up indicators if not already done
            if not hasattr(self, 'indicators_setup') or not self.indicators_setup:
                if not self._setup_indicators():
                    logger.error("Failed to set up indicators")
                    return pd.DataFrame(), pd.DataFrame(), stats
            
            # Main backtest loop
            for i in range(len(self.data_handler.data)):
                current_bar = self.data_handler.data.iloc[i].to_dict()
                current_date = self.data_handler.data.index[i]
                self.current_date = current_date
                
                # Update positions (trailing stops, take profit, etc.)
                self.update_positions(current_bar, current_date, stats)
                
                # Check for new entries
                self.check_entry_conditions(current_bar, i, stats)
                
                # Update daily stats
                self.update_daily_stats(current_date, current_bar, stats)
            
            # Close all positions at the end of backtest
            self.close_all_positions(stats)
            
            # Calculate final statistics
            stats['final_balance'] = self.balance
            stats['total_return'] = (self.balance / self.initial_balance - 1) * 100
            
            # Convert trade log to DataFrame
            trade_log_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'date': self.data_handler.data.index,
                'balance': [self.initial_balance] * len(self.data_handler.data),
                'equity': [self.initial_balance] * len(self.data_handler.data),
                'drawdown': [0.0] * len(self.data_handler.data)
            })
            
            logger.info("Backtest completed successfully")
            return results_df, trade_log_df, stats
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}", exc_info=True)
            return None, None, None

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
                logger.warning(f"No valid price in current bar: {current_bar}")
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



    def update_daily_stats(self, date, current_bar, stats):
        """
        Update daily performance statistics
        
        Args:
            date: Current date
            current_bar: Current market data
            stats: Statistics dictionary to update
        """
        if not hasattr(self, 'daily_stats'):
            self.daily_stats = {}
            
        # Calculate daily PnL
        daily_pnl = 0.0
        for position in getattr(self, 'positions', []):
            if position.get('status') == 'OPEN':
                current_price = current_bar.get('Close', 0)
                entry_price = position.get('entry_price', 0)
                if entry_price > 0:
                    pnl_pct = (current_price / entry_price - 1) * 100
                    daily_pnl += pnl_pct * position.get('quantity', 0)
        
        # Update daily stats
        self.daily_stats[date] = {
            'balance': self.balance,
            'equity': self.balance + daily_pnl,
            'pnl': daily_pnl,
            'open_positions': len([p for p in getattr(self, 'positions', []) if p.get('status') == 'OPEN'])
        }
        
        # Update stats
        if daily_pnl > 0:
            stats['winning_days'] = stats.get('winning_days', 0) + 1
        elif daily_pnl < 0:
            stats['losing_days'] = stats.get('losing_days', 0) + 1
        else:
            stats['break_even_days'] = stats.get('break_even_days', 0) + 1
            
        stats['total_days'] = stats.get('total_days', 0) + 1
        
    def _setup_indicators(self):
        """
        Set up technical indicators required for the strategy.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Add RSI with 14-day period
            if 'RSI_14' not in self.data_handler.data.columns:
                delta = self.data_handler.data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                self.data_handler.data['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Add 200-day Simple Moving Average
            if 'SMA_200' not in self.data_handler.data.columns:
                self.data_handler.data['SMA_200'] = self.data_handler.data['Close'].rolling(window=200).mean()
            
            # Add Bollinger Bands
            if 'BB_upper' not in self.data_handler.data.columns or 'BB_lower' not in self.data_handler.data.columns:
                sma = self.data_handler.data['Close'].rolling(window=20).mean()
                std = self.data_handler.data['Close'].rolling(window=20).std()
                self.data_handler.data['BB_upper'] = sma + (std * 2)
                self.data_handler.data['BB_lower'] = sma - (std * 2)
            
            # Add MACD
            if 'MACD' not in self.data_handler.data.columns:
                exp1 = self.data_handler.data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = self.data_handler.data['Close'].ewm(span=26, adjust=False).mean()
                self.data_handler.data['MACD'] = exp1 - exp2
                self.data_handler.data['Signal_Line'] = self.data_handler.data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Fill any NaN values that might have been created
            self.data_handler.data.ffill(inplace=True)
            self.data_handler.data.bfill(inplace=True)
            
            self.indicators_setup = True
            logger.info("Successfully set up technical indicators")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up indicators: {str(e)}")
            return False
            
    def run(self):
        """
        Main strategy execution loop
        
        Returns:
            tuple: (results_df, trade_log_df, stats)
        """
        try:
            self.logger.info("Starting strategy execution...")
            
            # Initialize statistics
            stats = {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'break_even_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'avg_trade_duration': 0.0,
                'total_days': 0,
                'risk_free_rate': 0.05,  # 5% annual risk-free rate
                'trading_days_per_year': 252
            }
            
            # Initialize trade log
            trade_log = []
            
            # Ensure positions list exists
            if not hasattr(self, 'positions'):
                self.positions = []
                
            # Initialize results list with the initial balance
            results = [{
                'date': self.data_handler.data.index[0],
                'balance': self.balance,
                'equity': self.balance,
                'open_positions': 0
            }]
                
            # Main strategy loop (start from 1 since we already have the first row)
            for i in range(1, len(self.data_handler.data)):
                try:
                    current_bar = self.data_handler.data.iloc[i].to_dict()
                    current_date = self.data_handler.data.index[i]
                    self.current_date = current_date
                    
                    # Update existing positions (check for exits)
                    self.update_positions(current_bar, current_date, stats)
                    
                    # Check for new entry signals
                    self.check_entry_conditions(current_bar, i, stats)
                    
                    # Update daily statistics
                    self.update_daily_stats(current_date, current_bar, stats)
                    
                    # Calculate current equity (balance + open positions value)
                    open_positions = [p for p in self.positions if p.get('status') == 'open']
                    positions_value = sum(pos.get('current_value', 0) for pos in open_positions)
                    
                    # Append daily results
                    results.append({
                        'date': current_date,
                        'balance': self.balance,
                        'equity': self.balance + positions_value,
                        'open_positions': len(open_positions)
                    })
                    
                    # Log daily update
                    logger.debug(f"Day {i+1}/{len(self.data_handler.data)} - "
                                f"Balance: {self.balance:,.2f}, "
                                f"Equity: {self.balance + positions_value:,.2f}, "
                                f"Open Positions: {len(open_positions)}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing bar {i}: {str(e)}", exc_info=True)
                    continue
                    
            # Close all open positions at the end of backtest
            self.close_all_positions(stats)
            
            # Calculate final statistics
            stats['final_balance'] = self.balance
            stats['total_return'] = ((self.balance / self.initial_balance) - 1) * 100
            
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
                
            # Calculate profit factor if there are winning or losing trades
            if stats['losing_trades'] > 0:
                stats['profit_factor'] = (stats['winning_trades'] * stats['avg_win']) / \
                                       (stats['losing_trades'] * stats['avg_loss']) if stats['losing_trades'] > 0 else float('inf')
            
            # Log final statistics
            self.logger.info("\n=== BACKTEST COMPLETE ===")
            self.logger.info(f"Initial Balance: {self.initial_balance:,.2f}")
            self.logger.info(f"Final Balance: {self.balance:,.2f}")
            self.logger.info(f"Total Return: {stats['total_return']:.2f}%")
            self.logger.info(f"Total Trades: {stats['total_trades']}")
            self.logger.info(f"Winning Trades: {stats['winning_trades']}")
            self.logger.info(f"Losing Trades: {stats['losing_trades']}")
            
            # Log trade statistics if trades were taken
            if stats['total_trades'] > 0:
                self.logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
                self.logger.info(f"Average Win: {stats['avg_win']:.2f}")
                self.logger.info(f"Average Loss: {stats['avg_loss']:.2f}")
                self.logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
                self.logger.info(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
            else:
                self.logger.info("No trades taken")
            
            # Create DataFrames for results and trade log
            results_df = pd.DataFrame(results)
            trade_log_df = pd.DataFrame(trade_log if trade_log else [])
            
            return results_df, trade_log_df, stats
            
        except Exception as e:
            self.logger.error(f"Error in strategy execution: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), stats

    def close_all_positions(self, stats):
        """
        Close all open positions at the end of backtest
        
        Args:
            stats: Statistics dictionary to update
        """
        if not hasattr(self, 'positions') or not self.positions:
            return
            
        last_bar = self.data_handler.data.iloc[-1].to_dict()
        last_price = last_bar.get('Close', 0)
        
        for position in self.positions[:]:
            if position.get('status') == 'OPEN':
                self._close_position(
                    position=position,
                    exit_price=last_price,
                    exit_reason='END_OF_BACKTEST',
                    current_date=self.current_date,
                    stats=stats
                )
    
    def _close_position(self, position, exit_price, exit_reason, current_date, stats, exit_quantity=None):
        """
        Close a position and update statistics
        
        Args:
            position: Position to close
            exit_price: Price at which to close the position
            exit_reason: Reason for closing
            current_date: Current date
            stats: Statistics dictionary to update
            exit_quantity: Quantity to close (None for full position)
        """
        if position.get('status') != 'OPEN':
            return
            
        # Calculate PnL
        entry_price = position.get('entry_price', 0)
        quantity = exit_quantity if exit_quantity is not None else position.get('quantity', 0)
        
        if entry_price == 0 or quantity == 0:
            logger.warning(f"Invalid position data: {position}")
            return
            
        pnl_pct = (exit_price / entry_price - 1) * 100
        pnl_amount = (exit_price - entry_price) * quantity
        
        # Update balance
        self.balance += pnl_amount
        
        # Update position
        position.update({
            'exit_date': current_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'status': 'CLOSED'
        })
        
        # Update trade log
        if not hasattr(self, 'trade_log'):
            self.trade_log = []
            
        self.trade_log.append({
            'entry_date': position['entry_date'],
            'exit_date': current_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'exit_reason': exit_reason
        })
        
        # Update statistics
        if pnl_amount > 0:
            stats['winning_trades'] = stats.get('winning_trades', 0) + 1
        elif pnl_amount < 0:
            stats['losing_trades'] = stats.get('losing_trades', 0) + 1
            
        stats['total_trades'] = stats.get('total_trades', 0) + 1
        
        logger.info(f"Position closed: {exit_reason}, PnL: {pnl_pct:.2f}% (${pnl_amount:.2f})")
        
        # Strategy parameters - More lenient for testing
        self.MIN_ENTRY_SCORE = 0.3  # Further lowered to generate more trades
        self.MAX_RISK_PER_TRADE = 0.05  # 5% risk per trade (increased from 3%)
        self.MIN_POSITION_VALUE = 100  # Lowered minimum position value to ₹100
        self.MAX_POSITION_SIZE = 0.8  # Increased to 80% of account
        self.PROFIT_TARGET_MULTIPLIER = 1.5  # 1.5:1 reward:risk ratio
        self.ATR_STOP_MULTIPLIER = 1.0  # Tighter stop loss (1.0 ATR)
        self.MAX_HOLDING_DAYS = 5  # Reduced maximum holding period
        self.DAILY_LOSS_LIMIT = 0.05  # 5% maximum daily loss
        self.MIN_ACCOUNT_BALANCE = 10000  # Lowered minimum account balance
        
        logger.info(f"Strategy parameters - Min Score: {self.MIN_ENTRY_SCORE}, Risk: {self.MAX_RISK_PER_TRADE*100}%")
        logger.info(f"Position Sizing - Min: ₹{self.MIN_POSITION_VALUE}, Max Size: {self.MAX_POSITION_SIZE*100}% of account")
        
    def setup_indicators(self) -> bool:
        """Initialize technical indicators"""
        try:
            df = self.data_handler.data
            
            # Basic indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # ATR for volatility
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            df['TrueRange'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = df['TrueRange'].rolling(window=14).mean()
            
            # Volume MA
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            
            # Clean up
            df.fillna(0, inplace=True)
            self.indicators_setup = True
            return True
            
        except Exception as e:
            logger.error(f"Error setting up indicators: {str(e)}")
            return False
    
    def _calculate_entry_score(self, current_bar: dict, prev_bar: dict, idx: int) -> tuple[float, list]:
        """Calculate entry score based on multiple factors"""
        reasons = []
        
        # Safely get and convert indicator values to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if not pd.isna(value) else default
            except (TypeError, ValueError):
                return default
                
        # Get current bar values
        close = safe_float(current_bar.get('Close'))
        current_date = current_bar.get('Date', 'Unknown date')
        
        # Log the current bar date and price for debugging
        logger.debug(f"\n=== Calculating entry score for {current_date} (Price: ₹{close:,.2f}) ===")
        
        # Safely get and convert indicator values to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if not pd.isna(value) else default
            except (TypeError, ValueError):
                return default
        
        # Get current bar values
        close = safe_float(current_bar.get('Close'))
        sma_20 = safe_float(current_bar.get('SMA_20'))
        sma_50 = safe_float(current_bar.get('SMA_50'))
        sma_200 = safe_float(current_bar.get('SMA_200'))
        rsi = safe_float(current_bar.get('RSI'))
        volume = safe_float(current_bar.get('Volume'))
        volume_ma = safe_float(current_bar.get('Volume_MA_20'))
        atr = safe_float(current_bar.get('ATR'))
        
        # Calculate MACD if available
        macd_line = safe_float(current_bar.get('MACD', 0))
        macd_signal = safe_float(current_bar.get('MACD_Signal', 0))
        macd_hist = macd_line - macd_signal if 'MACD' in current_bar else 0
        
        # 1. Trend (50% weight) - Increased importance
        trend_score = 0.0
        sma_aligned = sma_20 > sma_50 > sma_200  # Uptrend
        price_above_sma = close > sma_20 > sma_50  # Price above key moving averages
        
        if sma_aligned and price_above_sma:
            trend_score += 0.5  # Increased weight
            reasons.append('uptrend')
            
            # Additional confirmation from MACD if available
            if 'MACD' in current_bar and macd_hist > 0:
                trend_score += 0.1  # Bonus for MACD confirmation
                reasons.append('macd_bullish')
        
        # 2. Momentum (30% weight) - Increased weight
        momentum_score = 0.0
        rsi_ok = 30 < rsi < 85  # Even wider RSI range
        
        if rsi_ok:
            momentum_score += 0.2  # Base score for RSI in range
            reasons.append('rsi_ok')
            
            # Add more weight if RSI is in strong trending area
            if 40 < rsi < 80:
                momentum_score += 0.1
                reasons.append('strong_momentum')
            
            if rsi > 75:
                reasons.append('approaching_overbought')
            elif rsi < 40:
                reasons.append('approaching_oversold')
            
        # 3. Volume (20% weight) - Increased weight
        volume_score = 0.0
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1
        
        if volume_ratio > 1.3:  # Volume 30% above average
            volume_score += 0.2
            reasons.append('high_volume')
        elif volume_ratio > 1.1:  # Volume 10% above average
            volume_score += 0.1  # Partial score for moderate volume
            reasons.append('moderate_volume')
        else:
            volume_score += 0.05  # Small score for any volume
            reasons.append('low_volume')
            
        # 4. Volatility (10% weight)
        volatility_score = 0.0
        atr_percent = atr / close if close > 0 else 0
        
        if 0.005 < atr_percent < 0.03:  # Wider range for volatility
            volatility_score += 0.1
            reasons.append('good_volatility')
            
            # Bonus for increasing volatility (potential breakout)
            try:
                if prev_bar is not None and 'ATR' in prev_bar and not pd.isna(prev_bar['ATR']):
                    prev_atr = float(prev_bar['ATR'])
                    if atr > prev_atr * 1.1:  # Volatility increasing
                        volatility_score += 0.05
                        reasons.append('volatility_increasing')
            except (TypeError, ValueError) as e:
                logger.debug(f"Error checking volatility increase: {e}")
                pass
            
        total_score = trend_score + momentum_score + volume_score + volatility_score
        
        # Cap the total score at 1.0
        total_score = min(1.0, total_score)
        
        # Log detailed scoring breakdown
        logger.debug(f"Scoring Breakdown - Trend: {trend_score:.2f}, Momentum: {momentum_score:.2f}, "
                   f"Volume: {volume_score:.2f}, Volatility: {volatility_score:.2f} => Total: {total_score:.2f}")
        logger.debug(f"Reasons: {', '.join(reasons) if reasons else 'None'}")
        
        return total_score, reasons
    
    def check_entry_conditions(self, current_bar, idx: int, stats) -> None:
        """
        Check if entry conditions are met for new positions
        
        Args:
            current_bar: Current market data
            idx: Current index in the data
            stats: Statistics dictionary to update
        """
        try:
            if idx < 1 or idx >= len(self.data_handler.data):
                return
                
            # Get previous bar for comparison
            prev_bar = self.data_handler.data.iloc[idx - 1].to_dict() if idx > 0 else None
            
            # Check for missing data
            required_indicators = ['Close', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'Volume', 'ATR']
            missing_indicators = [k for k in required_indicators if pd.isna(current_bar.get(k, None))]
            
            if missing_indicators:
                logger.debug(f"Skipping bar - Missing indicators: {', '.join(missing_indicators)}")
                return
                
            # Convert all necessary values to float to avoid Series issues
            close = float(current_bar['Close'])
            prev_close = float(prev_bar['Close'])
            change_pct = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0
            
            logger.debug(f"\n--- Checking Entry at {self.current_date} ---")
            logger.debug(f"Price: {close:.2f} ({change_pct:+.2f}%)")
            
            # Check if we already have an open position
            if len(self.positions) > 0:
                logger.debug("Skipping - Already in a position")
                return
                
            # Calculate entry score
            entry_score, reasons = self._calculate_entry_score(current_bar, prev_bar, idx)
            
            # Log detailed entry information
            logger.info(f"\n=== Potential Trade Setup ===")
            logger.info(f"Date: {self.current_date}, Price: ₹{close:,.2f}")
            logger.info(f"Entry Score: {entry_score:.2f} (Min: {self.MIN_ENTRY_SCORE:.2f})")
            logger.info(f"Reasons: {', '.join(reasons) if reasons else 'No strong signals'}")
            
            # Log indicator values for debugging
            logger.debug(f"Indicator Values - RSI: {safe_float(current_bar.get('RSI')):.2f}, "
                       f"SMA 20/50/200: {safe_float(current_bar.get('SMA_20')):.2f}/"
                       f"{safe_float(current_bar.get('SMA_50')):.2f}/"
                       f"{safe_float(current_bar.get('SMA_200')):.2f}, "
                       f"Volume: {safe_float(current_bar.get('Volume')):,.0f} (MA: {safe_float(current_bar.get('Volume_MA_20')):,.0f}), "
                       f"ATR: {safe_float(current_bar.get('ATR')):.2f}")
            
            if entry_score < self.MIN_ENTRY_SCORE:
                logger.debug(f"Entry score too low: {entry_score:.2f} < {self.MIN_ENTRY_SCORE}")
                return
                
            # Calculate position size based on risk
            atr = float(current_bar.get('ATR', 0))
            entry_price = close
            stop_loss = entry_price - (atr * 1.5)  # 1.5 * ATR stop loss
            take_profit = entry_price + (atr * 3)  # 2:1 reward:risk ratio
            
            # Risk 1% of account per trade
            risk_amount = self.balance * 0.01
            position_size = risk_amount / (entry_price - stop_loss) if (entry_price - stop_loss) > 0 else 0
            
            # Ensure we don't use more than 10% of account equity per trade
            max_position_size = (self.balance * 0.1) / entry_price
            position_size = min(position_size, max_position_size)
            
            if position_size <= 0:
                logger.debug("Position size is zero or negative, skipping")
                return
                
            logger.info(f"✅ Entry Signal - Score: {entry_score:.2f}")
            logger.info(f"Entry: {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            logger.info(f"Position Size: {position_size:.2f} shares (${position_size * entry_price:.2f})")
            
            # Create new position
            position = {
                'entry_date': self.current_date,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': position_size,
                'status': 'OPEN',
                'current_pnl': 0.0,
                'max_holding_days': 10  # Max holding period in days
            }
            take_profit_pct = stop_loss_pct * self.PROFIT_TARGET_MULTIPLIER
            
            # Check account balance
            if self.balance < self.MIN_ACCOUNT_BALANCE:
                logger.warning(f"Account balance below minimum: ₹{self.balance:,.2f} < ₹{self.MIN_ACCOUNT_BALANCE:,.2f}")
                return
                
            # Check daily loss limit
            daily_pnl_pct = (self.balance - self.initial_balance) / self.initial_balance * 100
            if daily_pnl_pct < -self.DAILY_LOSS_LIMIT * 100:
                logger.warning(f"Daily loss limit reached: {daily_pnl_pct:.2f}%")
                return
            
            # Add position to positions list
            self.positions.append(position)
            
            # Update statistics
            stats['total_trades'] = stats.get('total_trades', 0) + 1
            
            # Log the entry
            logger.info(f"\n=== ENTRY SIGNAL at {self.current_date} ===")
            logger.info(f"Price: {close:.2f}, ATR: {atr:.2f}, Score: {entry_score:.2f}")
            logger.info(f"Position: {position_size:.2f} shares, Value: ₹{position_size * entry_price:,.2f}")
            logger.info(f"Stop Loss: ₹{stop_loss:.2f}, Take Profit: ₹{take_profit:.2f}")
            logger.info(f"Risk: ₹{risk_amount:.2f} ({(risk_amount/self.balance*100):.1f}% of capital)")
            logger.info(f"Reasons: {', '.join(reasons) if reasons else 'No signals'}")
            
            # Add to trade log
            trade = {
                'entry_date': self.current_date,
                'exit_date': None,
                'entry_price': entry_price,
                'exit_price': None,
                'quantity': position_size,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'holding_period': 0,
                'exit_reason': None,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            self.trade_log.append(trade)
            
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    

