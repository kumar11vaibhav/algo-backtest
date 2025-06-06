import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

class DataHandler:
    def __init__(self, symbol="^NSEI", start_date=None, end_date=None):
        self.symbol = symbol  # Using NSEI.BO for BSE data which is more reliable
        self.data = None
        self.trading_time = time(9, 15)  # 9:15 AM IST (market open time)
        self.ist_tz = ZoneInfo('Asia/Kolkata')
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
    def fetch_historical_data(self, start_date, end_date) -> bool:
        """
        Fetch historical Nifty data from Yahoo Finance with additional market data
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            bool: True if data was fetched successfully, False otherwise
        """
        try:
            # Ensure dates are in the correct format
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()
                
            self.logger.info(f"Fetching historical data for {self.symbol} from {start_date} to {end_date}")
            
            # Add one day to end_date to include the end date in the results
            end_date_plus_1 = end_date + pd.Timedelta(days=1)
            
            # Fetch OHLCV data
            try:
                # Download the data with progress
                self.logger.info("Downloading data from Yahoo Finance...")
                data = yf.download(
                    self.symbol, 
                    start=start_date, 
                    end=end_date_plus_1,
                    progress=True,
                    auto_adjust=True,  # Adjust for corporate actions
                    threads=True       # Use threads for faster download
                )
                
                if data is None or data.empty:
                    self.logger.error(f"No data returned for {self.symbol}")
                    return False
                    
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    self.logger.error(f"Missing required columns in downloaded data. Got: {data.columns}")
                    return False
                
                # Log the raw data structure
                self.logger.info("Raw data info:")
                self.logger.info(f"Columns: {data.columns.tolist()}")
                self.logger.info(f"Index type: {type(data.index)}")
                self.logger.info(f"Shape: {data.shape}")
                self.logger.info(f"First 5 rows: \n{data.head()}")
                
                # Flatten MultiIndex columns if they exist
                if isinstance(data.columns, pd.MultiIndex):
                    self.logger.info("Flattening MultiIndex columns...")
                    data.columns = data.columns.get_level_values(0)
                    self.logger.info(f"Flattened columns: {data.columns.tolist()}")
                
                # Ensure we have all required columns
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    self.logger.error(f"Missing required columns: {missing_cols}")
                    return False
                
                # Create a clean copy of the data with only required columns
                self.data = data[required_columns].copy()
                
                # Convert all columns to numeric
                for col in required_columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                
                # Ensure we have enough data points
                if len(self.data) < 50:  # Need at least 50 data points for indicators
                    self.logger.error(f"Insufficient data points: {len(self.data)}. Need at least 50.")
                    return False
                
                # Add technical indicators
                if not self._add_technical_indicators():
                    self.logger.error("Failed to add technical indicators")
                    return False
                
                # Add implied volatility data
                if not self._add_implied_volatility():
                    self.logger.error("Failed to add implied volatility data")
                    return False
                
                # Ensure all required columns exist and are numeric
                if not self._ensure_numeric_columns():
                    self.logger.error("Failed to ensure numeric columns")
                    return False
                    # Keep only the first level of column names (the actual data labels)
                    data.columns = data.columns.get_level_values(0)
                    self.logger.info(f"Flattened columns: {data.columns.tolist()}")
                
                # Ensure all required columns exist
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    self.logger.error(f"Missing required columns: {missing_cols}")
                    return False
                
                # Create a clean copy of the data with only required columns
                self.data = data[required_columns].copy()
                
                # Convert all columns to numeric
                for col in required_columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                
                # Log basic info about the processed data
                self.logger.info(f"Successfully processed {len(self.data)} rows of data")
                self.logger.info(f"Final columns: {self.data.columns.tolist()}")
                self.logger.info(f"Data types: \n{self.data.dtypes}")
                self.logger.info(f"First 5 rows: \n{self.data.head()}")
                
            except Exception as e:
                self.logger.error(f"Error downloading data: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return False
            
            # Add additional required columns
            if not self._add_technical_indicators():
                self.logger.error("Failed to add technical indicators")
                return False
                
            if not self._add_implied_volatility():
                self.logger.error("Failed to add implied volatility data")
                return False
            
            # Ensure all required columns exist
            required_columns = ['EMA20', 'EMA50', 'RSI', 'ATR', 'IV', 'IV_Percentile']
            for col in required_columns:
                if col not in self.data.columns:
                    self.logger.error(f"Missing required column after indicator setup: {col}")
                    return False
            
            # Handle NaN values more gracefully
            initial_count = len(self.data)
            
            # First, try to fill NaN values with reasonable defaults
            for col in self.data.columns:
                if self.data[col].isnull().any():
                    if col in ['Open', 'High', 'Low', 'Close']:
                        # For price data, fill with previous close or next available
                        self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
                    elif col == 'Volume':
                        # For volume, fill with 0 or mean of recent values
                        self.data[col] = self.data[col].fillna(0)
                    elif col in ['EMA20', 'EMA50', 'RSI', 'ATR', 'IV', 'IV_Percentile']:
                        # For indicators, fill with most recent value or neutral value
                        if col == 'RSI':
                            self.data[col] = self.data[col].fillna(50.0)  # Neutral RSI
                        elif col == 'IV_Percentile':
                            self.data[col] = self.data[col].fillna(50.0)  # Neutral percentile
                        else:
                            self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
            
            # After filling, check if we still have any NaNs in critical columns
            critical_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA20', 'EMA50', 'RSI', 'ATR', 'IV', 'IV_Percentile']
            critical_columns = [col for col in critical_columns if col in self.data.columns]
            
            # Only drop rows where all critical columns are NaN
            mask = self.data[critical_columns].isnull().all(axis=1)
            if mask.any():
                rows_before = len(self.data)
                self.data = self.data[~mask]
                rows_dropped = rows_before - len(self.data)
                if rows_dropped > 0:
                    self.logger.warning(f"Dropped {rows_dropped} rows with all critical values missing")
            
            if len(self.data) == 0:
                self.logger.error("No valid data points remaining after cleaning")
                return False
                
            self.logger.info(f"Successfully processed data. Remaining rows: {len(self.data)}")
            
            # Ensure all required columns exist with default values if missing
            for col in critical_columns:
                if col not in self.data.columns:
                    self.logger.warning(f"Missing column {col}, filling with default values")
                    if col == 'RSI':
                        self.data[col] = 50.0
                    elif col == 'IV_Percentile':
                        self.data[col] = 50.0
                    elif col in ['EMA20', 'EMA50', 'ATR', 'IV']:
                        self.data[col] = self.data['Close'].rolling(window=20).mean()
                    elif col == 'Volume':
                        self.data[col] = 0
                    else:
                        self.data[col] = self.data['Close']  # For price columns
            
            self.logger.info(f"Successfully downloaded and processed {len(self.data)} days of data")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
            
    def _add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        try:
            self.logger.info("Starting to add technical indicators...")
            
            # Make a copy of the data to avoid SettingWithCopyWarning
            df = self.data.copy()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return False
            
            # Log initial data types and sample data
            self.logger.debug("Initial data types and sample data before conversion:")
            self.logger.debug(f"Data types: \n{df.dtypes}")
            self.logger.debug(f"First 5 rows: \n{df.head()}")
            
            # Log initial data info
            self.logger.info("Starting data conversion...")
            self.logger.info(f"DataFrame columns: {df.columns.tolist()}")
            self.logger.info(f"DataFrame shape: {df.shape}")
            
            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                self.logger.info("Flattening MultiIndex columns...")
                df.columns = df.columns.get_level_values(0)
                self.logger.info(f"Flattened columns: {df.columns.tolist()}")
            
            # Ensure all required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Convert all numeric columns to float using a more robust approach
            for col in required_columns:
                try:
                    self.logger.info(f"\n--- Processing column: {col} ---")
                    
                    # Ensure the column exists
                    if col not in df.columns:
                        self.logger.error(f"Column {col} not found in DataFrame")
                        return False
                    
                    # Get the column data as a Series
                    series = df[col].copy()
                    
                    # Log basic info
                    self.logger.info(f"Series type: {type(series).__name__}")
                    self.logger.info(f"Series shape: {series.shape}")
                    
                    # Safely log the first few values
                    try:
                        if not series.empty:
                            self.logger.info(f"First 5 values: {series.head().tolist()}")
                        else:
                            self.logger.warning("Series is empty")
                    except Exception as e:
                        self.logger.warning(f"Could not log first 5 values: {str(e)}")
                    
                    # Skip if already float
                    if pd.api.types.is_float_dtype(series):
                        self.logger.info("Column is already float, skipping conversion")
                        continue
                    
                    # Skip if already float
                    if pd.api.types.is_float_dtype(series):
                        self.logger.info("Column is already float, skipping conversion")
                        continue
                    
                    # Convert to numeric using pandas' to_numeric
                    try:
                        self.logger.info("Attempting pd.to_numeric conversion...")
                        converted = pd.to_numeric(series, errors='coerce')
                        
                        # Check if conversion was successful
                        if converted.isna().all():
                            self.logger.warning("All values became NaN after conversion, trying manual method...")
                            raise ValueError("All values became NaN")
                            
                        df[col] = converted
                        self.logger.info("Successfully converted using pd.to_numeric")
                        
                    except Exception as e:
                        self.logger.warning(f"pd.to_numeric failed: {str(e)}, trying manual conversion...")
                        try:
                            # Manual conversion for problematic data
                            df[col] = series.astype(str).str.replace('[^\d.-]', '', regex=True).replace('', np.nan).astype(float)
                            self.logger.info("Successfully converted using manual method")
                        except Exception as e2:
                            self.logger.error(f"Manual conversion failed: {str(e2)}")
                            return False
                    
                    # Handle any remaining NaN values
                    if df[col].isna().any():
                        null_count = df[col].isna().sum()
                        self.logger.warning(f"Found {null_count} NaN values in column {col} after conversion")
                        
                        # Try to fill NaNs using forward and backward fill
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                        
                        # If there are still NaNs, fill with 0
                        if df[col].isna().any():
                            df[col] = df[col].fillna(0)
                            self.logger.warning(f"Filled remaining {df[col].isna().sum()} NaNs with 0 in column {col}")
                    
                    # Ensure the final type is float
                    df[col] = df[col].astype(float)
                    
                    # Log success
                    self.logger.info(f"Successfully processed column {col}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing column {col}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return False
            
            # Log final data types and sample data
            self.logger.debug("\nFinal data after conversion:")
            self.logger.debug(f"Data types: \n{df.dtypes}")
            self.logger.debug("\nFirst 5 rows:")
            self.logger.debug(df[required_columns].head().to_string())
            self.logger.debug("\nLast 5 rows:")
            self.logger.debug(df[required_columns].tail().to_string())
            
            # Verify all required columns are numeric
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.logger.error(f"Column {col} is not numeric after conversion")
                    return False
            
            # 1. Calculate Returns
            self.logger.debug("Calculating daily returns...")
            try:
                df['Returns'] = df['Close'].pct_change()
                
                # 2. Calculate Volatility
                self.logger.debug("Calculating daily volatility...")
                df['Daily_Volatility'] = df['Returns'].rolling(window=20, min_periods=5).std() * np.sqrt(252)
                
                # Fill any NaN values that might have been introduced
                df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Log sample of calculated values
                self.logger.debug("Sample returns and volatility:")
                self.logger.debug(f"First 5 returns: {df['Returns'].head().values}")
                self.logger.debug(f"First 5 volatilities: {df['Daily_Volatility'].head().values}")
                
            except Exception as e:
                self.logger.error(f"Error calculating returns/volatility: {str(e)}")
                return False
            
            # 3. Volume Indicators
            self.logger.debug("Calculating volume indicators...")
            if 'Volume' not in df.columns:
                self.logger.error("Volume column not found in data")
                return False
                
            df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # Calculate volume ratio safely
            df['Volume_Ratio'] = 1.0  # Default value
            mask = df['Volume_MA20'] > 0
            df.loc[mask, 'Volume_Ratio'] = df.loc[mask, 'Volume'] / df.loc[mask, 'Volume_MA20']
            
            # 4. Moving Averages - Simplified calculation with more robust handling
            self.logger.debug("Calculating EMAs...")
            
            try:
                # Calculate EMAs with fixed min_periods=1 to avoid NaNs
                df['EMA20'] = df['Close'].ewm(span=20, min_periods=1, adjust=False).mean()
                df['EMA50'] = df['Close'].ewm(span=50, min_periods=1, adjust=False).mean()
                
                # Log some sample values for debugging
                self.logger.debug(f"Sample EMA20 values: {df['EMA20'].head(5).tolist()}")
                self.logger.debug(f"Sample EMA50 values: {df['EMA50'].head(5).tolist()}")
                
            except Exception as e:
                self.logger.error(f"Error calculating EMAs: {str(e)}")
                return False
            
            # 5. RSI - Simplified and more robust calculation
            self.logger.debug("Calculating RSI...")
            try:
                # Calculate price changes
                delta = df['Close'].diff()
                
                # Separate gains and losses
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                
                # Calculate average gain and loss
                window = 14
                avg_gain = gain.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
                
                # Calculate RS and RSI with zero division protection
                rs = np.where(avg_loss != 0, avg_gain / avg_loss, 1.0)  # Default to 1.0 if avg_loss is 0
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Ensure RSI is within bounds and fill any NaNs
                df['RSI'] = df['RSI'].fillna(50).clip(0, 100)
                
                # Log some sample values for debugging
                self.logger.debug(f"Sample RSI values: {df['RSI'].head(5).tolist()}")
                
            except Exception as e:
                self.logger.error(f"Error calculating RSI: {str(e)}")
                return False
            
            # 6. ATR - Simplified calculation to avoid dimension issues
            self.logger.debug("Calculating ATR...")
            
            # Calculate True Range components directly as Series
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift(1)).abs()
            tr3 = (df['Low'] - df['Close'].shift(1)).abs()
            
            # Calculate True Range as the maximum of the three components
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # Update the original data
            self.data = df
            
            # Clean up any remaining NaN values
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info("Successfully added all technical indicators")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in _add_technical_indicators: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
        
        # Ensure all required columns exist
        if 'IV' not in self.data.columns:
            self.data['IV'] = self.data['Daily_Volatility']  # Use historical vol as proxy if IV not available
            
    def _add_implied_volatility(self):
        """
        Add implied volatility data (placeholder - in practice, use real IV data if available)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or len(self.data) < 10:  # Need at least 10 data points
                self.logger.error("Not enough data points for IV calculation")
                return False
                
            # Make a copy of the data to avoid SettingWithCopyWarning
            df = self.data.copy()
            
            # Ensure we have the Returns column
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()
            
            # Calculate historical volatility as a proxy for IV
            df['IV'] = df['Returns'].rolling(window=20, min_periods=5).std() * np.sqrt(252)  # Annualized
            
            # Fill any remaining NaN values with forward fill
            df['IV'] = df['IV'].ffill().bfill()
            
            # Add some noise to make it look more like real IV data
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 0.03, len(df))  # Reduced noise for more stability
            df['IV'] = df['IV'] * (1 + noise)
            
            # Keep IV in reasonable range (10% to 100%)
            df['IV'] = df['IV'].clip(lower=0.1, upper=1.0)
            
            # Calculate IV percentile (0-100) with sufficient lookback
            min_periods = min(10, len(df) // 2)  # Use at least 10 or half the data points
            
            # Use expanding window for percentile calculation
            df['IV_Percentile'] = df['IV'].expanding(min_periods=min_periods).rank(pct=True) * 100
            
            # Fill any remaining NaN values
            df['IV_Percentile'] = df['IV_Percentile'].fillna(50.0)  # Neutral value for initial data points
            
            # Update the original data
            self.data = df
            
            self.logger.info("Successfully added implied volatility data")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in _add_implied_volatility: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def get_close_price(self, index):
        """Get closing price for a specific index"""
        return float(self.data.iloc[index]['Close'])
    
    def get_date(self, index):
        """Get date for a specific index"""
        return self.data.index[index]
    
    def get_data_length(self):
        """Get total number of data points"""
        return len(self.data)
        
    def _ensure_numeric_columns(self):
        """
        Ensure all required columns are numeric.
        
        Returns:
            bool: True if all columns are numeric, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA20', 'EMA50', 'RSI', 'ATR', 'IV']
        
        for col in required_columns:
            if col not in self.data.columns:
                self.logger.warning(f"Column {col} not found in data")
                continue
                
            try:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                    
                # Try to convert to numeric
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                
                # Check if conversion was successful
                if self.data[col].isna().any():
                    self.logger.warning(f"Column {col} contains non-numeric values that couldn't be converted")
                    
            except Exception as e:
                self.logger.error(f"Error converting column {col} to numeric: {str(e)}")
                return False
                
        return True
    
    def calculate_historical_volatility(self, lookback: int = 20, annualize: bool = True) -> pd.Series:
        """
        Calculate historical volatility using daily returns
        
        Args:
            lookback: Number of days to look back for volatility calculation
            annualize: Whether to annualize the volatility
            
        Returns:
            pd.Series: Historical volatility series
        """
        try:
            if self.data is None or len(self.data) == 0:
                raise ValueError("No data available for volatility calculation")
                
            if len(self.data) < lookback:
                self.logger.warning(f"Not enough data points for {lookback}-day volatility calculation")
                return pd.Series(dtype=float, index=self.data.index)
            
            # Calculate daily log returns
            returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            # Calculate rolling standard deviation
            vol = returns.rolling(window=lookback).std()
            
            # Annualize if needed
            if annualize:
                vol = vol * np.sqrt(252)  # 252 trading days in a year
                
            return vol
            
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return pd.Series(dtype=float, index=self.data.index if self.data is not None else [])
            
    def get_market_regime(self, lookback: int = 50) -> str:
        """
        Determine the current market regime
        
        Args:
            lookback: Number of days to analyze
            
        Returns:
            str: Market regime ('bull', 'bear', 'neutral')
        """
        if self.data is None or len(self.data) < lookback:
            return 'neutral'
            
        returns = self.data['Close'].pct_change(lookback).iloc[-1]
        vol = self.data['Close'].pct_change().std() * np.sqrt(252)
        
        if returns > 0.1 and vol < 0.2:  # Strong uptrend with low vol
            return 'bull'
        elif returns < -0.1 and vol > 0.25:  # Strong downtrend with high vol
            return 'bear'
        else:
            return 'neutral'
    
    @staticmethod
    def get_strike_price(spot_price: float, iv_percentile: Optional[float] = None, step: int = 50) -> int:
        """
        Get optimal strike price based on spot price and IV percentile
        
        Args:
            spot_price: Current spot price
            iv_percentile: Current IV percentile (0-100)
            step: Strike price step size
            
        Returns:
            int: Optimal strike price
        """
        atm_strike = int(round(spot_price / step) * step)
        
        if iv_percentile is None:
            return atm_strike - (3 * step)  # Default to 3 strikes OTM
            
        # Adjust strike based on IV percentile
        if iv_percentile < 30:  # Low IV - get closer to ATM
            return atm_strike - (2 * step)
        elif iv_percentile > 70:  # High IV - go further OTM
            return atm_strike - (4 * step)
        else:  # Medium IV
            return atm_strike - (3 * step)
