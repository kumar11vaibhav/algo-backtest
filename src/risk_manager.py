from config import Config

class RiskManager:
    def __init__(self):
        self.daily_pnl = 0
        self.trailing_highs = {}  # Track trailing highs for each position
    
    def reset_daily_pnl(self):
        """Reset daily P&L at the start of each day"""
        self.daily_pnl = 0
    
    def can_enter_trade(self, positions, balance):
        """Check if we can enter a new trade"""
        # Check position limits
        if len(positions) >= Config.MAX_POSITION_SIZE:
            return False, "Maximum position size reached"
            
        # Check if we've hit daily loss limit
        if self.daily_pnl <= Config.MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
        
        return True, None
    
    def should_exit_trade(self, position, current_price):
        """Check if we should exit a trade based on risk management rules"""
        entry_price = position['entry_price']
        current_pnl_pct = (current_price - entry_price) / entry_price
        
        # Check stop loss
        if current_pnl_pct <= -Config.STOP_LOSS_PCT:
            return True, "Stop loss hit"
        
        # Check trailing stop
        position_id = position['id']
        if position_id not in self.trailing_highs:
            self.trailing_highs[position_id] = entry_price
        
        # Update trailing high if we have a new high
        if current_price > self.trailing_highs[position_id]:
            self.trailing_highs[position_id] = current_price
        
        # Check if we've fallen below trailing stop
        trailing_stop_price = self.trailing_highs[position_id] * (1 - Config.TRAILING_STOP_PCT)
        if current_price < trailing_stop_price:
            return True, "Trailing stop hit"
        
        return False, None
    
    def update_daily_pnl(self, pnl):
        """Update daily P&L"""
        self.daily_pnl += pnl
    
    def remove_position(self, position_id):
        """Clean up when a position is closed"""
        if position_id in self.trailing_highs:
            del self.trailing_highs[position_id]
