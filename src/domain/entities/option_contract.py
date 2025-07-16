"""
Option contract entity for the Options Trading Engine.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, Any
from enum import Enum

from ..value_objects.greeks import Greeks
from ..value_objects.trade_metrics import TradeMetrics
from ...infrastructure.error_handling import ValidationError


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class OptionAction(Enum):
    """Option action enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OptionContract:
    """
    Option contract entity.
    
    Represents a single option contract with all its properties,
    including pricing, Greeks, and market data.
    """
    
    # Basic contract details
    symbol: str
    option_type: OptionType
    strike: float
    expiration_date: date
    
    # Market data
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    
    # Greeks
    greeks: Greeks
    
    # Volatility
    implied_volatility: float
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    data_source: str = "unknown"
    
    # Calculated properties
    _mid_price: Optional[float] = field(init=False, default=None)
    _bid_ask_spread: Optional[float] = field(init=False, default=None)
    _time_to_expiration: Optional[float] = field(init=False, default=None)
    
    def __post_init__(self):
        """Validate option contract data."""
        self._validate_contract()
    
    def _validate_contract(self):
        """Validate option contract data."""
        if self.strike <= 0:
            raise ValidationError("Strike price must be positive")
        
        if self.expiration_date <= date.today():
            raise ValidationError("Expiration date must be in the future")
        
        if self.bid < 0 or self.ask < 0:
            raise ValidationError("Bid and ask prices must be non-negative")
        
        if self.ask > 0 and self.bid > self.ask:
            raise ValidationError("Bid price cannot be higher than ask price")
        
        if self.volume < 0 or self.open_interest < 0:
            raise ValidationError("Volume and open interest must be non-negative")
        
        if not 0 <= self.implied_volatility <= 10:
            raise ValidationError("Implied volatility must be between 0 and 10")
    
    @property
    def mid_price(self) -> float:
        """Get mid price (bid-ask midpoint)."""
        if self._mid_price is None:
            if self.bid > 0 and self.ask > 0:
                self._mid_price = (self.bid + self.ask) / 2
            elif self.last > 0:
                self._mid_price = self.last
            else:
                self._mid_price = 0.0
        return self._mid_price
    
    @property
    def bid_ask_spread(self) -> float:
        """Get bid-ask spread."""
        if self._bid_ask_spread is None:
            if self.bid > 0 and self.ask > 0:
                self._bid_ask_spread = self.ask - self.bid
            else:
                self._bid_ask_spread = 0.0
        return self._bid_ask_spread
    
    @property
    def spread_percentage(self) -> float:
        """Get bid-ask spread as percentage of mid price."""
        if self.mid_price > 0:
            return (self.bid_ask_spread / self.mid_price) * 100
        return 0.0
    
    @property
    def time_to_expiration(self) -> float:
        """Get time to expiration in years."""
        if self._time_to_expiration is None:
            days_to_exp = (self.expiration_date - date.today()).days
            self._time_to_expiration = days_to_exp / 365.25
        return self._time_to_expiration
    
    @property
    def days_to_expiration(self) -> int:
        """Get days to expiration."""
        return (self.expiration_date - date.today()).days
    
    @property
    def is_liquid(self) -> bool:
        """Check if option is liquid based on volume and open interest."""
        return self.volume >= 10 and self.open_interest >= 50
    
    @property
    def is_tight_spread(self) -> bool:
        """Check if option has tight bid-ask spread."""
        return self.spread_percentage <= 5.0  # 5% spread threshold
    
    @property
    def option_code(self) -> str:
        """Get standardized option code."""
        exp_str = self.expiration_date.strftime("%y%m%d")
        option_char = "C" if self.option_type == OptionType.CALL else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.symbol}{exp_str}{option_char}{strike_str}"
    
    def get_intrinsic_value(self, underlying_price: float) -> float:
        """Calculate intrinsic value."""
        if self.option_type == OptionType.CALL:
            return max(0, underlying_price - self.strike)
        else:  # PUT
            return max(0, self.strike - underlying_price)
    
    def get_time_value(self, underlying_price: float) -> float:
        """Calculate time value."""
        intrinsic = self.get_intrinsic_value(underlying_price)
        return max(0, self.mid_price - intrinsic)
    
    def is_in_the_money(self, underlying_price: float) -> bool:
        """Check if option is in the money."""
        return self.get_intrinsic_value(underlying_price) > 0
    
    def is_out_of_the_money(self, underlying_price: float) -> bool:
        """Check if option is out of the money."""
        return self.get_intrinsic_value(underlying_price) == 0
    
    def is_at_the_money(self, underlying_price: float, threshold: float = 0.02) -> bool:
        """Check if option is at the money (within threshold)."""
        return abs(underlying_price - self.strike) / underlying_price <= threshold
    
    def calculate_profit_loss(self, underlying_price: float, action: OptionAction) -> float:
        """Calculate profit/loss at expiration."""
        intrinsic_value = self.get_intrinsic_value(underlying_price)
        
        if action == OptionAction.BUY:
            return intrinsic_value - self.mid_price
        else:  # SELL
            return self.mid_price - intrinsic_value
    
    def get_breakeven_price(self, action: OptionAction) -> float:
        """Get breakeven price at expiration."""
        if action == OptionAction.BUY:
            if self.option_type == OptionType.CALL:
                return self.strike + self.mid_price
            else:  # PUT
                return self.strike - self.mid_price
        else:  # SELL
            if self.option_type == OptionType.CALL:
                return self.strike + self.mid_price
            else:  # PUT
                return self.strike - self.mid_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'option_type': self.option_type.value,
            'strike': self.strike,
            'expiration_date': self.expiration_date.isoformat(),
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'mid_price': self.mid_price,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'bid_ask_spread': self.bid_ask_spread,
            'spread_percentage': self.spread_percentage,
            'time_to_expiration': self.time_to_expiration,
            'days_to_expiration': self.days_to_expiration,
            'is_liquid': self.is_liquid,
            'is_tight_spread': self.is_tight_spread,
            'option_code': self.option_code,
            'greeks': self.greeks.to_dict(),
            'last_updated': self.last_updated.isoformat(),
            'data_source': self.data_source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionContract':
        """Create from dictionary representation."""
        return cls(
            symbol=data['symbol'],
            option_type=OptionType(data['option_type']),
            strike=data['strike'],
            expiration_date=date.fromisoformat(data['expiration_date']),
            bid=data['bid'],
            ask=data['ask'],
            last=data['last'],
            volume=data['volume'],
            open_interest=data['open_interest'],
            implied_volatility=data['implied_volatility'],
            greeks=Greeks.from_dict(data['greeks']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            data_source=data.get('data_source', 'unknown')
        )
    
    def __eq__(self, other) -> bool:
        """Check equality based on contract identifier."""
        if not isinstance(other, OptionContract):
            return False
        return (
            self.symbol == other.symbol and
            self.option_type == other.option_type and
            self.strike == other.strike and
            self.expiration_date == other.expiration_date
        )
    
    def __hash__(self) -> int:
        """Hash based on contract identifier."""
        return hash((self.symbol, self.option_type, self.strike, self.expiration_date))
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.symbol} {self.expiration_date} {self.strike} {self.option_type.value.upper()}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"OptionContract({self.symbol}, {self.option_type.value}, {self.strike}, {self.expiration_date})"