"""
Core data models for options contracts and related financial instruments.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator


class OptionType(str, Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """Option exercise style."""
    AMERICAN = "american"
    EUROPEAN = "european"


@dataclass(frozen=True)
class OptionSymbol:
    """Standardized option symbol representation."""
    underlying: str
    expiration: date
    strike: Decimal
    option_type: OptionType
    
    def __str__(self) -> str:
        """Generate OCC standard option symbol."""
        exp_str = self.expiration.strftime("%y%m%d")
        type_code = "C" if self.option_type == OptionType.CALL else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.underlying:<6}{exp_str}{type_code}{strike_str}"
    
    @classmethod
    def from_string(cls, symbol: str) -> "OptionSymbol":
        """Parse option symbol from OCC standard format."""
        if len(symbol) < 21:
            raise ValueError(f"Invalid option symbol format: {symbol}")
        
        underlying = symbol[:6].strip()
        exp_date = datetime.strptime(symbol[6:12], "%y%m%d").date()
        option_type = OptionType.CALL if symbol[12] == "C" else OptionType.PUT
        strike = Decimal(symbol[13:21]) / 1000
        
        return cls(
            underlying=underlying,
            expiration=exp_date,
            strike=strike,
            option_type=option_type
        )


class Greeks(BaseModel):
    """Option Greeks container."""
    delta: Optional[float] = Field(None, description="Price sensitivity to underlying")
    gamma: Optional[float] = Field(None, description="Delta sensitivity to underlying")
    theta: Optional[float] = Field(None, description="Time decay")
    vega: Optional[float] = Field(None, description="Volatility sensitivity")
    rho: Optional[float] = Field(None, description="Interest rate sensitivity")
    
    @validator('delta')
    def validate_delta(cls, v):
        if v is not None and not -1 <= v <= 1:
            raise ValueError("Delta must be between -1 and 1")
        return v
    
    @validator('gamma')
    def validate_gamma(cls, v):
        if v is not None and v < 0:
            raise ValueError("Gamma must be non-negative")
        return v


class OptionQuote(BaseModel):
    """Real-time option quote data."""
    symbol: str = Field(..., description="Option symbol")
    underlying: str = Field(..., description="Underlying ticker")
    strike: Decimal = Field(..., description="Strike price")
    expiration: date = Field(..., description="Expiration date")
    option_type: OptionType = Field(..., description="Call or Put")
    
    # Price data
    bid: Optional[Decimal] = Field(None, description="Current bid price")
    ask: Optional[Decimal] = Field(None, description="Current ask price")
    last: Optional[Decimal] = Field(None, description="Last trade price")
    mark: Optional[Decimal] = Field(None, description="Mark price (mid)")
    
    # Volume and interest
    volume: Optional[int] = Field(None, description="Daily volume")
    open_interest: Optional[int] = Field(None, description="Open interest")
    
    # Volatility and Greeks
    implied_volatility: Optional[float] = Field(None, description="Implied volatility")
    greeks: Optional[Greeks] = Field(None, description="Option Greeks")
    
    # Metadata
    quote_time: datetime = Field(default_factory=datetime.utcnow, description="Quote timestamp")
    exchange: Optional[str] = Field(None, description="Exchange")
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.mark
    
    @property
    def bid_ask_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def bid_ask_spread_pct(self) -> Optional[float]:
        """Calculate bid-ask spread as percentage of mid."""
        spread = self.bid_ask_spread
        mid = self.mid_price
        if spread is not None and mid is not None and mid > 0:
            return float(spread / mid)
        return None
    
    @property
    def days_to_expiration(self) -> int:
        """Calculate days until expiration."""
        return (self.expiration - date.today()).days
    
    @property
    def time_to_expiration(self) -> float:
        """Calculate time to expiration in years."""
        return self.days_to_expiration / 365.25
    
    def is_liquid(self, min_volume: int = 10, min_oi: int = 100, max_spread_pct: float = 0.5) -> bool:
        """Check if option meets liquidity requirements."""
        # Volume check
        if self.volume is None or self.volume < min_volume:
            return False
        
        # Open interest check
        if self.open_interest is None or self.open_interest < min_oi:
            return False
        
        # Spread check
        spread_pct = self.bid_ask_spread_pct
        if spread_pct is None or spread_pct > max_spread_pct:
            return False
        
        return True
    
    def is_quote_fresh(self, max_age_minutes: int = 10) -> bool:
        """Check if quote is fresh enough for trading."""
        age = datetime.utcnow() - self.quote_time
        return age.total_seconds() <= max_age_minutes * 60


class OptionsChain(BaseModel):
    """Complete options chain for an underlying."""
    underlying: str = Field(..., description="Underlying ticker")
    underlying_price: Optional[Decimal] = Field(None, description="Current underlying price")
    
    # Options by expiration and strike
    options: Dict[date, Dict[Decimal, Dict[OptionType, OptionQuote]]] = Field(
        default_factory=dict,
        description="Options organized by expiration -> strike -> type"
    )
    
    # Metadata
    chain_timestamp: datetime = Field(default_factory=datetime.utcnow)
    data_source: Optional[str] = Field(None, description="Data provider")
    
    def add_option(self, option: OptionQuote):
        """Add an option to the chain."""
        exp = option.expiration
        strike = option.strike
        opt_type = option.option_type
        
        if exp not in self.options:
            self.options[exp] = {}
        if strike not in self.options[exp]:
            self.options[exp][strike] = {}
        
        self.options[exp][strike][opt_type] = option
    
    def get_option(self, expiration: date, strike: Decimal, option_type: OptionType) -> Optional[OptionQuote]:
        """Retrieve specific option from chain."""
        return self.options.get(expiration, {}).get(strike, {}).get(option_type)
    
    def get_expirations(self) -> list[date]:
        """Get all available expiration dates."""
        return sorted(self.options.keys())
    
    def get_strikes(self, expiration: date) -> list[Decimal]:
        """Get all strikes for a given expiration."""
        return sorted(self.options.get(expiration, {}).keys())
    
    def get_atm_strike(self, expiration: date) -> Optional[Decimal]:
        """Find at-the-money strike for given expiration."""
        if self.underlying_price is None:
            return None
        
        strikes = self.get_strikes(expiration)
        if not strikes:
            return None
        
        # Find closest strike to underlying price
        return min(strikes, key=lambda s: abs(s - self.underlying_price))
    
    def filter_liquid_options(self, **liquidity_params) -> "OptionsChain":
        """Return new chain with only liquid options."""
        filtered_chain = OptionsChain(
            underlying=self.underlying,
            underlying_price=self.underlying_price,
            data_source=self.data_source
        )
        
        for exp_date, strikes in self.options.items():
            for strike, types in strikes.items():
                for opt_type, option in types.items():
                    if option.is_liquid(**liquidity_params):
                        filtered_chain.add_option(option)
        
        return filtered_chain


class OptionContract(BaseModel):
    """Complete option contract specification."""
    symbol: OptionSymbol = Field(..., description="Option symbol")
    quote: Optional[OptionQuote] = Field(None, description="Current quote")
    
    # Contract specifications
    contract_size: int = Field(100, description="Contract multiplier")
    exercise_style: OptionStyle = Field(OptionStyle.AMERICAN, description="Exercise style")
    settlement_type: str = Field("physical", description="Settlement type")
    
    # Risk metrics
    intrinsic_value: Optional[Decimal] = Field(None, description="Intrinsic value")
    time_value: Optional[Decimal] = Field(None, description="Time value")
    
    @property
    def is_itm(self) -> Optional[bool]:
        """Check if option is in-the-money."""
        if self.quote is None or self.quote.underlying is None:
            return None
        # Implementation would need underlying price
        return None
    
    @property
    def moneyness(self) -> Optional[float]:
        """Calculate moneyness (S/K for calls, K/S for puts)."""
        if self.quote is None:
            return None
        # Implementation would need underlying price
        return None