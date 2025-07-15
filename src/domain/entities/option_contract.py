"""
Option contract domain entity with rich business behaviors.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum
import math

from ...data.models.options import OptionQuote, OptionType, Greeks


class MoneynessCategoryE(str, Enum):
    """Option moneyness categories."""
    DEEP_ITM = "deep_itm"  # >20% ITM
    ITM = "itm"            # 0-20% ITM
    ATM = "atm"            # ±5% of ATM
    OTM = "otm"            # 0-20% OTM
    DEEP_OTM = "deep_otm"  # >20% OTM


class LiquidityTier(str, Enum):
    """Option liquidity tiers."""
    EXCELLENT = "excellent"  # High volume, tight spreads
    GOOD = "good"           # Moderate volume, reasonable spreads
    FAIR = "fair"           # Low volume, wide spreads
    POOR = "poor"           # Very low volume, very wide spreads
    ILLIQUID = "illiquid"   # No volume, no market


class OptionContract:
    """
    Rich domain entity for option contracts with business behaviors.
    
    This class wraps the data model OptionQuote and adds business logic,
    calculations, and domain-specific behaviors.
    """
    
    def __init__(self, option_quote: OptionQuote, underlying_price: Optional[Decimal] = None):
        """
        Initialize option contract domain entity.
        
        Args:
            option_quote: The underlying data model
            underlying_price: Current underlying asset price
        """
        self._option_quote = option_quote
        self._underlying_price = underlying_price
        self._cached_calculations: Dict[str, Any] = {}
    
    @property
    def quote(self) -> OptionQuote:
        """Get the underlying option quote."""
        return self._option_quote
    
    @property
    def underlying_price(self) -> Optional[Decimal]:
        """Get the underlying price."""
        return self._underlying_price
    
    def update_underlying_price(self, price: Decimal) -> None:
        """Update underlying price and clear cached calculations."""
        self._underlying_price = price
        self._cached_calculations.clear()
    
    def get_intrinsic_value(self) -> Decimal:
        """Calculate intrinsic value."""
        if not self._underlying_price:
            return Decimal('0')
        
        cache_key = f"intrinsic_{self._underlying_price}"
        if cache_key not in self._cached_calculations:
            self._cached_calculations[cache_key] = self._option_quote.get_intrinsic_value(self._underlying_price)
        
        return self._cached_calculations[cache_key]
    
    def get_time_value(self) -> Optional[Decimal]:
        """Calculate time value."""
        if not self._underlying_price or not self._option_quote.mark:
            return None
        
        cache_key = f"time_value_{self._underlying_price}"
        if cache_key not in self._cached_calculations:
            self._cached_calculations[cache_key] = self._option_quote.get_time_value(self._underlying_price)
        
        return self._cached_calculations[cache_key]
    
    def get_moneyness_category(self) -> Optional[MoneynessCategoryE]:
        """Determine moneyness category."""
        if not self._underlying_price:
            return None
        
        cache_key = f"moneyness_{self._underlying_price}"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        strike = self._option_quote.strike
        underlying = self._underlying_price
        
        # Calculate percentage difference
        if self._option_quote.option_type == OptionType.CALL:
            pct_diff = float((underlying - strike) / strike)
        else:  # PUT
            pct_diff = float((strike - underlying) / strike)
        
        # Categorize based on percentage
        if pct_diff > 0.20:
            category = MoneynessCategoryE.DEEP_ITM
        elif pct_diff > 0:
            category = MoneynessCategoryE.ITM
        elif abs(pct_diff) <= 0.05:
            category = MoneynessCategoryE.ATM
        elif pct_diff > -0.20:
            category = MoneynessCategoryE.OTM
        else:
            category = MoneynessCategoryE.DEEP_OTM
        
        self._cached_calculations[cache_key] = category
        return category
    
    def get_liquidity_tier(self) -> LiquidityTier:
        """Assess option liquidity tier."""
        cache_key = "liquidity_tier"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        volume = self._option_quote.volume or 0
        open_interest = self._option_quote.open_interest or 0
        spread_pct = self._option_quote.bid_ask_spread_percent or 1.0
        
        # Score based on multiple factors
        score = 0
        
        # Volume scoring
        if volume >= 100:
            score += 3
        elif volume >= 50:
            score += 2
        elif volume >= 10:
            score += 1
        
        # Open interest scoring
        if open_interest >= 1000:
            score += 3
        elif open_interest >= 500:
            score += 2
        elif open_interest >= 100:
            score += 1
        
        # Spread scoring (lower is better)
        if spread_pct <= 0.05:  # 5%
            score += 3
        elif spread_pct <= 0.15:  # 15%
            score += 2
        elif spread_pct <= 0.35:  # 35%
            score += 1
        
        # Determine tier
        if score >= 7:
            tier = LiquidityTier.EXCELLENT
        elif score >= 5:
            tier = LiquidityTier.GOOD
        elif score >= 3:
            tier = LiquidityTier.FAIR
        elif score >= 1:
            tier = LiquidityTier.POOR
        else:
            tier = LiquidityTier.ILLIQUID
        
        self._cached_calculations[cache_key] = tier
        return tier
    
    def calculate_probability_itm(self, volatility: Optional[float] = None) -> Optional[float]:
        """
        Calculate probability of finishing in-the-money using Black-Scholes.
        
        Args:
            volatility: Annual volatility (use IV if not provided)
            
        Returns:
            Probability between 0 and 1, or None if insufficient data
        """
        if not self._underlying_price:
            return None
        
        vol = volatility or self._option_quote.implied_volatility
        if not vol:
            return None
        
        cache_key = f"prob_itm_{vol}_{self._underlying_price}"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        # Black-Scholes probability calculation
        S = float(self._underlying_price)  # Current price
        K = float(self._option_quote.strike)  # Strike price
        T = self._option_quote.days_to_expiration / 365.0  # Time to expiration
        r = 0.05  # Risk-free rate (simplified)
        sigma = vol  # Volatility
        
        if T <= 0:
            # Expired or expiring today
            if self._option_quote.option_type == OptionType.CALL:
                prob = 1.0 if S > K else 0.0
            else:
                prob = 1.0 if S < K else 0.0
        else:
            try:
                # Calculate d2 from Black-Scholes
                d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                
                # Standard normal CDF approximation
                prob = self._norm_cdf(d2)
                
                # For puts, probability is 1 - N(d2)
                if self._option_quote.option_type == OptionType.PUT:
                    prob = 1.0 - prob
                    
            except (ValueError, ZeroDivisionError):
                prob = None
        
        self._cached_calculations[cache_key] = prob
        return prob
    
    def calculate_delta_neutral_hedge(self) -> Optional[int]:
        """
        Calculate shares needed to delta hedge this option.
        
        Returns:
            Number of shares (positive = buy, negative = sell)
        """
        if not self._option_quote.greeks or not self._option_quote.greeks.delta:
            return None
        
        # For 1 contract (100 shares), hedge ratio is -delta * 100
        delta = self._option_quote.greeks.delta
        shares_per_contract = 100
        
        # If long option, need to sell shares to hedge
        # If short option, need to buy shares to hedge
        hedge_shares = -int(delta * shares_per_contract)
        
        return hedge_shares
    
    def get_leverage_ratio(self) -> Optional[float]:
        """
        Calculate option leverage relative to underlying.
        
        Returns:
            Leverage ratio (option % move / underlying % move)
        """
        if (not self._underlying_price or not self._option_quote.mark or 
            not self._option_quote.greeks or not self._option_quote.greeks.delta):
            return None
        
        cache_key = f"leverage_{self._underlying_price}"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        # Leverage = (Delta * Underlying Price) / Option Price
        delta = abs(self._option_quote.greeks.delta)
        underlying_price = float(self._underlying_price)
        option_price = float(self._option_quote.mark)
        
        if option_price == 0:
            return None
        
        leverage = (delta * underlying_price) / option_price
        
        self._cached_calculations[cache_key] = leverage
        return leverage
    
    def is_liquid_enough(self, min_volume: int = 10, min_oi: int = 100, max_spread: float = 0.5) -> bool:
        """Check if option meets liquidity requirements."""
        tier = self.get_liquidity_tier()
        
        # Direct checks
        volume_ok = (self._option_quote.volume or 0) >= min_volume
        oi_ok = (self._option_quote.open_interest or 0) >= min_oi
        spread_ok = (self._option_quote.bid_ask_spread_percent or 1.0) <= max_spread
        
        return volume_ok and oi_ok and spread_ok
    
    def is_quote_fresh(self, max_age_minutes: float = 10.0) -> bool:
        """Check if quote is fresh enough."""
        if not self._option_quote.quote_time:
            return False
        
        age_minutes = (datetime.utcnow() - self._option_quote.quote_time).total_seconds() / 60
        return age_minutes <= max_age_minutes
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics for this option."""
        metrics = {
            'symbol': self._option_quote.symbol,
            'underlying': self._option_quote.underlying,
            'strike': self._option_quote.strike,
            'expiration': self._option_quote.expiration,
            'option_type': self._option_quote.option_type,
            'days_to_expiration': self._option_quote.days_to_expiration,
            'mark_price': self._option_quote.mark,
            'implied_volatility': self._option_quote.implied_volatility,
            'liquidity_tier': self.get_liquidity_tier(),
            'is_liquid': self.is_liquid_enough(),
            'is_quote_fresh': self.is_quote_fresh(),
        }
        
        # Add underlying-dependent metrics
        if self._underlying_price:
            metrics.update({
                'underlying_price': self._underlying_price,
                'intrinsic_value': self.get_intrinsic_value(),
                'time_value': self.get_time_value(),
                'moneyness_category': self.get_moneyness_category(),
                'probability_itm': self.calculate_probability_itm(),
                'leverage_ratio': self.get_leverage_ratio(),
                'delta_hedge_shares': self.calculate_delta_neutral_hedge(),
            })
        
        # Add Greeks if available
        if self._option_quote.greeks:
            metrics.update({
                'delta': self._option_quote.greeks.delta,
                'gamma': self._option_quote.greeks.gamma,
                'theta': self._option_quote.greeks.theta,
                'vega': self._option_quote.greeks.vega,
                'rho': self._option_quote.greeks.rho,
            })
        
        return metrics
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximate standard normal cumulative distribution function."""
        # Abramowitz and Stegun approximation
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        # Constants
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    def __str__(self) -> str:
        """String representation of the option contract."""
        return (f"{self._option_quote.underlying} "
                f"{self._option_quote.strike} "
                f"{self._option_quote.option_type.value.upper()} "
                f"{self._option_quote.expiration}")
    
    def __repr__(self) -> str:
        """Detailed representation of the option contract."""
        return (f"OptionContract({self._option_quote.symbol}, "
                f"mark={self._option_quote.mark}, "
                f"iv={self._option_quote.implied_volatility}, "
                f"liquidity={self.get_liquidity_tier()})")


class OptionContractBuilder:
    """Builder for creating OptionContract instances with validation."""
    
    def __init__(self):
        self._option_quote: Optional[OptionQuote] = None
        self._underlying_price: Optional[Decimal] = None
    
    def with_option_quote(self, option_quote: OptionQuote) -> 'OptionContractBuilder':
        """Set the option quote."""
        self._option_quote = option_quote
        return self
    
    def with_underlying_price(self, price: Decimal) -> 'OptionContractBuilder':
        """Set the underlying price."""
        self._underlying_price = price
        return self
    
    def build(self) -> OptionContract:
        """Build the OptionContract instance."""
        if not self._option_quote:
            raise ValueError("Option quote is required")
        
        return OptionContract(self._option_quote, self._underlying_price)
    
    @classmethod
    def from_quote(cls, option_quote: OptionQuote, underlying_price: Optional[Decimal] = None) -> OptionContract:
        """Convenience method to create OptionContract from quote."""
        return cls().with_option_quote(option_quote).with_underlying_price(underlying_price).build()