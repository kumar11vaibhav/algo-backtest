import numpy as np
from scipy.stats import norm
from typing import Optional, Dict, Tuple
from datetime import datetime

class OptionPricer:
    """
    Enhanced option pricing model with Greeks calculation and volatility adjustments.
    Implements Black-Scholes model with volatility smile/skew adjustments.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.05,
                 dividend_yield: float = 0.0,
                 days_in_year: int = 252):
        """
        Initialize the option pricer.
        
        Args:
            risk_free_rate: Annual risk-free interest rate (default: 5%)
            dividend_yield: Annual dividend yield (default: 0%)
            days_in_year: Number of trading days in a year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.days_in_year = days_in_year
        self.volatility_floor = 0.05  # Minimum volatility
        self.volatility_ceiling = 2.0  # Maximum volatility
        
    def calculate_price(self, 
                        spot: float, 
                        strike: float, 
                        days_to_expiry: int, 
                        iv: float, 
                        option_type: str = 'call',
                        use_vol_smile: bool = True) -> Dict[str, float]:
        """
        Calculate option price and Greeks using Black-Scholes model.
        
        Args:
            spot: Current spot price of the underlying
            strike: Option strike price
            days_to_expiry: Days until option expiration
            iv: Implied volatility (0-1)
            option_type: 'call' or 'put'
            use_vol_smile: Whether to adjust volatility based on moneyness
            
        Returns:
            Dict containing price and Greeks
        """
        # Input validation
        if spot <= 0 or strike <= 0 or days_to_expiry < 0 or iv < 0:
            raise ValueError("Invalid input parameters")
            
        # Convert inputs to float
        S = float(spot)
        K = float(strike)
        T = max(1/365, days_to_expiry / self.days_in_year)  # At least 1 day
        r = self.risk_free_rate
        q = self.dividend_yield
        
        # Adjust implied volatility based on moneyness if enabled
        if use_vol_smile:
            iv = self._adjust_volatility(iv, S, K, T)
        
        # Ensure volatility is within reasonable bounds
        iv = max(self.volatility_floor, min(iv, self.volatility_ceiling))
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        # Calculate Greeks
        greeks = self._calculate_greeks(S, K, T, r, q, iv, d1, d2, option_type)
        
        # Ensure minimum price
        price = max(price, 0.01)
        
        return {
            'price': price,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'rho': greeks['rho'],
            'iv': iv,
            'intrinsic': max(0, S - K) if option_type == 'call' else max(0, K - S),
            'extrinsic': max(0, price - max(0, S - K) if option_type == 'call' else max(0, K - S)),
            'days_to_expiry': days_to_expiry
        }
    
    def _adjust_volatility(self, base_iv: float, spot: float, strike: float, T: float) -> float:
        """
        Adjust implied volatility based on moneyness and time to expiration.
        Implements volatility smile/skew.
        """
        moneyness = strike / spot
        
        # Volatility smile adjustment (higher IV for OTM options)
        if moneyness < 0.95:  # Deep ITM call / Deep OTM put
            iv = base_iv * 1.15
        elif moneyness > 1.05:  # Deep OTM call / Deep ITM put
            iv = base_iv * 1.10
        else:  # Near the money
            iv = base_iv
            
        # Term structure adjustment (higher IV for near-term options)
        if T < 7/365:  # Less than a week
            iv *= 1.20
        elif T < 30/365:  # Less than a month
            iv *= 1.10
            
        return iv
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float, q: float, 
                         iv: float, d1: float, d2: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks."""
        # Common calculations
        sqrt_T = np.sqrt(T)
        d1 = d1 if not np.isnan(d1) else 0
        d2 = d2 if not np.isnan(d2) and not np.isinf(d2) else 0
        
        # Delta
        if option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            delta = -np.exp(-q * T) * norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * iv * sqrt_T)
        
        # Theta (per day)
        if option_type.lower() == 'call':
            theta = (-S * iv * np.exp(-q * T) * norm.pdf(d1) / (2 * sqrt_T) -
                    r * K * np.exp(-r * T) * norm.cdf(d2) +
                    q * S * np.exp(-q * T) * norm.cdf(d1)) / self.days_in_year
        else:  # put
            theta = (-S * iv * np.exp(-q * T) * norm.pdf(d1) / (2 * sqrt_T) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2) -
                    q * S * np.exp(-q * T) * norm.cdf(-d1)) / self.days_in_year
        
        # Vega (per 1% change in IV)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T * 0.01
        
        # Rho (per 1% change in interest rate)
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_iv(self, price: float, spot: float, strike: float, 
                    days_to_expiry: int, option_type: str = 'call', 
                    max_iter: int = 100, precision: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            price: Current option price
            spot: Current spot price
            strike: Option strike price
            days_to_expiry: Days until expiration
            option_type: 'call' or 'put'
            max_iter: Maximum iterations
            precision: Desired precision
            
        Returns:
            Implied volatility (0-1)
        """
        if price <= 0 or spot <= 0 or strike <= 0 or days_to_expiry < 0:
            return 0.0
            
        # Initial guess for IV (20%)
        iv = 0.20
        
        for _ in range(max_iter):
            # Calculate price with current IV
            result = self.calculate_price(spot, strike, days_to_expiry, iv, option_type, False)
            current_price = result['price']
            vega = result['vega']
            
            # Check for convergence
            price_diff = current_price - price
            if abs(price_diff) < precision:
                return max(self.volatility_floor, min(iv, self.volatility_ceiling))
                
            # Avoid division by zero
            if vega < 1e-10:
                break
                
            # Update IV using Newton-Raphson
            iv = iv - price_diff / vega * 100  # vega is for 1% change
            
            # Ensure IV stays within bounds
            iv = max(self.volatility_floor, min(iv, self.volatility_ceiling))
        
        return iv
