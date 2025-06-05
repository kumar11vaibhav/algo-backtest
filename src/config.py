class Config:
    # Data settings
    SYMBOL = "^NSEI"
    LOOKBACK_DAYS = 365
    
    # Option settings
    BASE_VOLATILITY = 0.15
    INTEREST_RATE = 0.07
    DAYS_TO_EXPIRY = 30
    STRIKE_STEP = 100
    
    # Trading settings
    LOT_SIZE = 50
    INITIAL_BALANCE = 100000
    TRADING_TIME = "09:30"
    
    # Strategy parameters
    PRICE_INCREASE_TRIGGER = 1.5  # 50% increase
    MIN_PROFIT_THRESHOLD = 100  # Min profit to count as winning trade
    MAX_LOSS_THRESHOLD = -100  # Max loss to count as losing trade
    
    # Risk management
    MAX_POSITION_SIZE = 5  # Maximum number of concurrent positions
    MAX_DAILY_LOSS = -10000  # Maximum daily loss
    STOP_LOSS_PCT = 0.15  # 15% stop loss
    TRAILING_STOP_PCT = 0.10  # 10% trailing stop
    
    # Volatility adjustments
    OTM_VOL_MULTIPLIER = 1.1  # Increase volatility for OTM options
    MIN_VOLATILITY = 0.10  # Minimum volatility floor
    HIST_VOL_WINDOW = 30  # Historical volatility calculation window
