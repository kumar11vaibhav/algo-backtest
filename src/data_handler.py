import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

class DataHandler:
    def __init__(self, symbol="^NSEI", start_date=None, end_date=None):
        self.symbol = symbol  # Using NSEI.BO for BSE data which is more reliable
        self.data = None
        self.trading_time = time(9, 15)  # 9:15 AM IST (market open time)
        self.ist_tz = ZoneInfo('Asia/Kolkata')
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
    def fetch_historical_data(self, start_date, end_date):
        """Fetch historical Nifty data from Yahoo Finance"""
        try:
            self.logger.info(f"Fetching historical data for {self.symbol} from {start_date} to {end_date}")
            self.data = yf.download(self.symbol, start=start_date, end=end_date)
            
            if self.data is None or self.data.empty:
                self.logger.error(f"No data returned for {self.symbol}")
                return False
                
            self.logger.info(f"Successfully downloaded {len(self.data)} days of data")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
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
    
    def calculate_historical_volatility(self, lookback=30):
        """Calculate historical volatility using daily returns"""
        try:
            if self.data is None or len(self.data) == 0:
                # Fetch 1 year of historical data if not already loaded
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(years=1)
                if not self.fetch_historical_data(start_date, end_date):
                    self.logger.error("Failed to fetch historical data for volatility calculation")
                    return None
            
            if len(self.data) < lookback:
                self.logger.warning(f"Not enough data points for {lookback}-day volatility calculation")
                return None
                
            returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
            vol = returns.rolling(window=lookback).std() * np.sqrt(252)
            return vol
            
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    @staticmethod
    def get_strike_price(spot_price, step=100):
        """Get strike price three steps below spot price for cheaper premiums"""
        atm_strike = int(spot_price / step) * step
        return atm_strike - (3 * step)  # Return three strikes below ATM
        
    def get_all_data(self):
        """Return the complete dataset with OHLCV data"""
        if self.data is None or self.data.empty:
            if not self.fetch_historical_data(self.start_date, self.end_date):
                self.logger.error("Failed to fetch historical data")
                return pd.DataFrame()
        return self.data
