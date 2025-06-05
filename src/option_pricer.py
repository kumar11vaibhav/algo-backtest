import numpy as np
from scipy.stats import norm

class OptionPricer:
    def __init__(self, base_volatility=0.15, interest_rate=0.07):
        self.base_volatility = base_volatility
        self.interest_rate = interest_rate
        
    def calculate_price(self, spot, strike, days_to_expiry, hist_vol=None, option_type='call'):
        """Calculate theoretical option price using Black-Scholes formula with dynamic volatility"""
        S = float(spot)
        K = float(strike)
        T = days_to_expiry / 365
        r = self.interest_rate

        # Use historical volatility if provided, otherwise use base volatility
        sigma = float(hist_vol) if hist_vol is not None else self.base_volatility

        # Adjust volatility based on moneyness
        moneyness = K/S
        if moneyness < 0.95 or moneyness > 1.05:  # OTM options
            sigma *= 1.1  # Increase volatility for OTM options

        # Minimum volatility floor
        sigma = max(sigma, 0.1)

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type.lower() == 'call':
            option_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # put
            option_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return max(option_price, 0.01)  # Minimum price of 0.01
