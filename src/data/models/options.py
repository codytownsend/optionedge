"""
Core data models for options contracts and related financial instruments.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import math

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
    """Option Greeks container with validation."""
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
    
    @validator('theta')
    def validate_theta(cls, v):
        if v is not None and v > 0:
            raise ValueError("Theta should typically be negative for long options")
        return v

    def __add__(self, other):
        """Add Greeks together for portfolio calculations."""
        if not isinstance(other, Greeks):
            raise TypeError("Can only add Greeks to Greeks")
        
        return Greeks(
            delta=(self.delta or 0) + (other.delta or 0),
            gamma=(self.gamma or 0) + (other.gamma or 0),
            theta=(self.theta or 0) + (other.theta or 0),
            vega=(self.vega or 0) + (other.vega or 0),
            rho=(self.rho or 0) + (other.rho or 0)
        )


class OptionQuote(BaseModel):
    """Real-time option quote data with all required fields from Tradier API."""
    symbol: str = Field(..., description="Option symbol")
    underlying: str = Field(..., description="Underlying ticker")
    strike: Decimal = Field(..., description="Strike price")
    expiration: date = Field(..., description="Expiration date")
    option_type: OptionType = Field(..., description="Call or Put")
    
    # Price data - all required for trading decisions
    bid: Optional[Decimal] = Field(None, description="Current bid price")
    ask: Optional[Decimal] = Field(None, description="Current ask price")
    last: Optional[Decimal] = Field(None, description="Last trade price")
    mark: Optional[Decimal] = Field(None, description="Mark price (mid)")
    
    # Volume and interest - critical for liquidity assessment
    volume: Optional[int] = Field(None, description="Daily volume")
    open_interest: Optional[int] = Field(None, description="Open interest")
    
    # Volatility and Greeks - essential for strategy construction
    implied_volatility: Optional[float] = Field(None, description="Implied volatility")
    greeks: Optional[Greeks] = Field(None, description="Option Greeks")
    
    # Metadata for quote validation
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
    
    def is_liquid(self, min_volume: int = 10, min_oi: int = 100, max_spread_pct: float = 0.05) -> bool:
        """Check if option meets liquidity requirements per instructions."""
        # Volume check
        if self.volume is None or self.volume < min_volume:
            return False
        
        # Open interest check  
        if self.open_interest is None or self.open_interest < min_oi:
            return False
        
        # Spread check (5% of mid price per instructions)
        spread_pct = self.bid_ask_spread_pct
        if spread_pct is None or spread_pct > max_spread_pct:
            return False
        
        return True
    
    def is_quote_fresh(self, max_age_minutes: int = 10) -> bool:
        """Check if quote is fresh enough per instructions (≤10 minutes)."""
        age = datetime.utcnow() - self.quote_time
        return age.total_seconds() <= max_age_minutes * 60

    def calculate_intrinsic_value(self, underlying_price: Decimal) -> Decimal:
        """Calculate intrinsic value of the option."""
        if self.option_type == OptionType.CALL:
            return max(Decimal('0'), underlying_price - self.strike)
        else:  # PUT
            return max(Decimal('0'), self.strike - underlying_price)

    def calculate_time_value(self, underlying_price: Decimal) -> Optional[Decimal]:
        """Calculate time value (extrinsic value) of the option."""
        intrinsic = self.calculate_intrinsic_value(underlying_price)
        if self.mid_price is not None:
            return max(Decimal('0'), self.mid_price - intrinsic)
        return None


class OptionsChain(BaseModel):
    """Complete options chain for an underlying with enhanced functionality."""
    underlying: str = Field(..., description="Underlying ticker")
    underlying_price: Optional[Decimal] = Field(None, description="Current underlying price")
    
    # Options organized by expiration -> strike -> type for efficient access
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
    
    def get_expirations(self) -> List[date]:
        """Get all available expiration dates sorted."""
        return sorted(self.options.keys())
    
    def get_strikes(self, expiration: date) -> List[Decimal]:
        """Get all strikes for a given expiration sorted."""
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
    
    def get_options_by_expiration(self, expiration: date) -> List[OptionQuote]:
        """Get all options for a specific expiration."""
        options_list = []
        exp_data = self.options.get(expiration, {})
        
        for strike_data in exp_data.values():
            for option in strike_data.values():
                options_list.append(option)
        
        return options_list
    
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
    
    def filter_by_dte_range(self, min_dte: int = 7, max_dte: int = 45) -> "OptionsChain":
        """Filter options by days to expiration range."""
        filtered_chain = OptionsChain(
            underlying=self.underlying,
            underlying_price=self.underlying_price,
            data_source=self.data_source
        )
        
        today = date.today()
        
        for exp_date, strikes in self.options.items():
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                for strike, types in strikes.items():
                    for opt_type, option in types.items():
                        filtered_chain.add_option(option)
        
        return filtered_chain
    
    def get_fresh_quotes_only(self, max_age_minutes: int = 10) -> "OptionsChain":
        """Return chain with only fresh quotes per instructions."""
        filtered_chain = OptionsChain(
            underlying=self.underlying,
            underlying_price=self.underlying_price,
            data_source=self.data_source
        )
        
        for exp_date, strikes in self.options.items():
            for strike, types in strikes.items():
                for opt_type, option in types.items():
                    if option.is_quote_fresh(max_age_minutes):
                        filtered_chain.add_option(option)
        
        return filtered_chain


class OptionContract(BaseModel):
    """Complete option contract specification with enhanced functionality."""
    symbol: OptionSymbol = Field(..., description="Option symbol")
    quote: Optional[OptionQuote] = Field(None, description="Current quote")
    
    # Contract specifications
    contract_size: int = Field(100, description="Contract multiplier")
    exercise_style: OptionStyle = Field(OptionStyle.AMERICAN, description="Exercise style")
    settlement_type: str = Field("physical", description="Settlement type")
    
    # Risk metrics
    intrinsic_value: Optional[Decimal] = Field(None, description="Intrinsic value")
    time_value: Optional[Decimal] = Field(None, description="Time value")
    
    def update_values(self, underlying_price: Decimal):
        """Update intrinsic and time values based on underlying price."""
        if self.quote:
            self.intrinsic_value = self.quote.calculate_intrinsic_value(underlying_price)
            self.time_value = self.quote.calculate_time_value(underlying_price)
    
    @property
    def is_itm(self) -> Optional[bool]:
        """Check if option is in-the-money."""
        if self.intrinsic_value is None:
            return None
        return self.intrinsic_value > 0
    
    @property
    def moneyness(self) -> Optional[float]:
        """Calculate moneyness ratio."""
        if not self.quote or not hasattr(self, 'underlying_price'):
            return None
            
        if self.symbol.option_type == OptionType.CALL:
            return float(self.underlying_price / self.symbol.strike)
        else:
            return float(self.symbol.strike / self.underlying_price)


# Black-Scholes calculation utilities
def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 parameter for Black-Scholes."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def calculate_d2(d1: float, sigma: float, T: float) -> float:
    """Calculate d2 parameter for Black-Scholes."""
    return d1 - sigma * math.sqrt(T)


def normal_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def calculate_black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> Greeks:
    """Calculate Black-Scholes Greeks as specified in instructions."""
    if T <= 0 or sigma <= 0:
        return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
    
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    
    # Standard normal PDF
    phi_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    
    # Delta calculations
    if option_type == OptionType.CALL:
        delta = normal_cdf(d1)
    else:
        delta = normal_cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = phi_d1 / (S * sigma * math.sqrt(T))
    
    # Theta calculations
    common_theta_term = -S * phi_d1 * sigma / (2 * math.sqrt(T))
    if option_type == OptionType.CALL:
        theta = (common_theta_term - r * K * math.exp(-r * T) * normal_cdf(d2)) / 365
    else:
        theta = (common_theta_term + r * K * math.exp(-r * T) * normal_cdf(-d2)) / 365
    
    # Vega (same for calls and puts)
    vega = S * phi_d1 * math.sqrt(T) / 100
    
    # Rho calculations
    if option_type == OptionType.CALL:
        rho = K * T * math.exp(-r * T) * normal_cdf(d2) / 100
    else:
        rho = -K * T * math.exp(-r * T) * normal_cdf(-d2) / 100
    
    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho
    )


def calculate_probability_of_profit_bs(S: float, K: float, T: float, r: float, sigma: float, 
                                      option_type: OptionType, net_credit: float = 0) -> float:
    """Calculate probability of profit using Black-Scholes for options strategies."""
    if T <= 0:
        return 0.0
    
    if option_type == OptionType.PUT:
        # For credit spreads, adjust strike by net credit
        breakeven = K - net_credit
        d2 = (math.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return normal_cdf(-d2)  # Probability of finishing below breakeven
    else:  # CALL
        breakeven = K + net_credit
        d2 = (math.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return normal_cdf(d2)  # Probability of finishing above breakeven