"""
Trade and strategy models for the options trading engine.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator

from .options import OptionQuote, Greeks, OptionType


class StrategyType(str, Enum):
    """Options strategy types."""
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    PROTECTIVE_PUT = "protective_put"


class TradeDirection(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class TradeLeg(BaseModel):
    """Individual leg of an options strategy."""
    
    option: OptionQuote = Field(..., description="Option contract")
    direction: TradeDirection = Field(..., description="Long or short")
    quantity: int = Field(..., description="Number of contracts")
    
    class Config:
        frozen = True
        validate_assignment = True
    
    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate market value of the leg."""
        if not self.option.mark:
            return None
        
        multiplier = 100  # Standard option multiplier
        sign = 1 if self.direction == TradeDirection.LONG else -1
        return self.option.mark * self.quantity * multiplier * sign
    
    @property
    def greeks_contribution(self) -> Optional[Greeks]:
        """Calculate Greeks contribution of this leg."""
        if not self.option.greeks:
            return None
        
        multiplier = 100 * self.quantity
        sign = 1 if self.direction == TradeDirection.LONG else -1
        
        return Greeks(
            delta=(self.option.greeks.delta * multiplier * sign) if self.option.greeks.delta else None,
            gamma=(self.option.greeks.gamma * multiplier * sign) if self.option.greeks.gamma else None,
            theta=(self.option.greeks.theta * multiplier * sign) if self.option.greeks.theta else None,
            vega=(self.option.greeks.vega * multiplier * sign) if self.option.greeks.vega else None,
            rho=(self.option.greeks.rho * multiplier * sign) if self.option.greeks.rho else None
        )


class StrategyDefinition(BaseModel):
    """Definition of an options strategy."""
    
    strategy_type: StrategyType = Field(..., description="Type of strategy")
    underlying: str = Field(..., description="Underlying symbol")
    legs: List[TradeLeg] = Field(..., description="Strategy legs")
    
    # Strategy metrics
    net_premium: Optional[Decimal] = Field(None, description="Net premium received/paid")
    max_profit: Optional[Decimal] = Field(None, description="Maximum profit potential")
    max_loss: Optional[Decimal] = Field(None, description="Maximum loss potential")
    
    # Probability metrics
    probability_of_profit: Optional[float] = Field(None, description="Probability of profit")
    breakeven_points: List[Decimal] = Field(default_factory=list, description="Breakeven prices")
    
    # Greeks
    net_delta: Optional[float] = Field(None, description="Net strategy delta")
    net_gamma: Optional[float] = Field(None, description="Net strategy gamma")
    net_theta: Optional[float] = Field(None, description="Net strategy theta")
    net_vega: Optional[float] = Field(None, description="Net strategy vega")
    net_rho: Optional[float] = Field(None, description="Net strategy rho")
    
    # Metadata
    expiration: Optional[date] = Field(None, description="Primary expiration date")
    days_to_expiration: Optional[int] = Field(None, description="Days to expiration")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Strategy creation time")
    
    class Config:
        validate_assignment = True
    
    @validator('legs')
    def must_have_legs(cls, v):
        if not v:
            raise ValueError('Strategy must have at least one leg')
        return v
    
    @validator('probability_of_profit')
    def pop_must_be_valid(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError('Probability of profit must be between 0 and 1')
        return v
    
    def calculate_net_greeks(self) -> None:
        """Calculate net Greeks from all legs."""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for leg in self.legs:
            leg_greeks = leg.greeks_contribution
            if leg_greeks:
                if leg_greeks.delta:
                    total_delta += leg_greeks.delta
                if leg_greeks.gamma:
                    total_gamma += leg_greeks.gamma
                if leg_greeks.theta:
                    total_theta += leg_greeks.theta
                if leg_greeks.vega:
                    total_vega += leg_greeks.vega
                if leg_greeks.rho:
                    total_rho += leg_greeks.rho
        
        self.net_delta = total_delta
        self.net_gamma = total_gamma
        self.net_theta = total_theta
        self.net_vega = total_vega
        self.net_rho = total_rho
    
    def calculate_net_premium(self) -> None:
        """Calculate net premium from all legs."""
        total_premium = Decimal('0')
        
        for leg in self.legs:
            leg_value = leg.market_value
            if leg_value:
                total_premium += leg_value
        
        self.net_premium = total_premium
    
    @property
    def is_credit_strategy(self) -> bool:
        """Check if strategy receives credit."""
        return self.net_premium is not None and self.net_premium > 0
    
    @property
    def is_debit_strategy(self) -> bool:
        """Check if strategy pays debit."""
        return self.net_premium is not None and self.net_premium < 0
    
    @property
    def credit_to_max_loss_ratio(self) -> Optional[float]:
        """Calculate credit-to-max-loss ratio."""
        if not self.is_credit_strategy or not self.max_loss or self.max_loss == 0:
            return None
        
        return float(self.net_premium / abs(self.max_loss))


class TradeCandidate(BaseModel):
    """A potential trade with scoring and ranking information."""
    
    strategy: StrategyDefinition = Field(..., description="Strategy definition")
    
    # Scoring components
    model_score: Optional[float] = Field(None, description="Composite model score")
    momentum_z_score: Optional[float] = Field(None, description="Momentum Z-score")
    flow_z_score: Optional[float] = Field(None, description="Flow Z-score")
    iv_rank: Optional[float] = Field(None, description="IV rank percentile")
    
    # Market data quality
    quote_age_minutes: Optional[float] = Field(None, description="Quote age in minutes")
    liquidity_score: Optional[float] = Field(None, description="Liquidity score")
    
    # Risk metrics
    capital_required: Optional[Decimal] = Field(None, description="Capital required")
    margin_required: Optional[Decimal] = Field(None, description="Margin required")
    
    # Fundamental context
    sector: Optional[str] = Field(None, description="GICS sector")
    market_cap: Optional[Decimal] = Field(None, description="Market capitalization")
    
    # Trade thesis
    thesis: Optional[str] = Field(None, description="Trade thesis (max 30 words)")
    
    # Ranking
    rank: Optional[int] = Field(None, description="Final rank")
    selected: bool = Field(False, description="Selected for execution")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    
    class Config:
        validate_assignment = True
    
    @validator('thesis')
    def thesis_word_limit(cls, v):
        if v and len(v.split()) > 30:
            raise ValueError('Thesis must be 30 words or fewer')
        return v
    
    @validator('quote_age_minutes')
    def quote_age_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Quote age cannot be negative')
        return v
    
    @property
    def underlying(self) -> str:
        """Get underlying symbol."""
        return self.strategy.underlying
    
    @property
    def strategy_type(self) -> StrategyType:
        """Get strategy type."""
        return self.strategy.strategy_type
    
    @property
    def probability_of_profit(self) -> Optional[float]:
        """Get probability of profit."""
        return self.strategy.probability_of_profit
    
    @property
    def net_premium(self) -> Optional[Decimal]:
        """Get net premium."""
        return self.strategy.net_premium
    
    @property
    def max_loss(self) -> Optional[Decimal]:
        """Get maximum loss."""
        return self.strategy.max_loss
    
    @property
    def days_to_expiration(self) -> Optional[int]:
        """Get days to expiration."""
        return self.strategy.days_to_expiration


@dataclass
class TradeFilterCriteria:
    """Criteria for filtering trades."""
    
    # Risk limits
    min_probability_of_profit: float = 0.65
    min_credit_to_max_loss: float = 0.33
    max_loss_per_trade: Decimal = Decimal('500')
    max_quote_age_minutes: float = 10.0
    
    # Portfolio limits
    max_delta_exposure: float = 0.30
    min_vega_exposure: float = -0.05
    max_trades_per_sector: int = 2
    
    # Liquidity requirements
    min_option_volume: int = 10
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 0.50
    
    # Capital constraints
    available_capital: Optional[Decimal] = None
    nav: Decimal = Decimal('100000')
    
    # Time constraints
    min_days_to_expiration: int = 7
    max_days_to_expiration: int = 45


@dataclass
class TradeExecutionPlan:
    """Plan for executing a selected trade."""
    
    trade_candidate: TradeCandidate
    
    # Execution details
    order_type: str = "LIMIT"
    time_in_force: str = "DAY"
    limit_prices: Dict[str, Decimal] = None
    
    # Risk management
    stop_loss: Optional[Decimal] = None
    profit_target: Optional[Decimal] = None
    
    # Timing
    execution_time: Optional[datetime] = None
    good_till_date: Optional[date] = None
    
    def __post_init__(self):
        if self.limit_prices is None:
            self.limit_prices = {}


class TradeResult(BaseModel):
    """Result of an executed trade."""
    
    trade_candidate: TradeCandidate = Field(..., description="Original trade candidate")
    execution_plan: TradeExecutionPlan = Field(..., description="Execution plan")
    
    # Execution details
    executed_at: datetime = Field(..., description="Execution timestamp")
    fill_prices: Dict[str, Decimal] = Field(default_factory=dict, description="Actual fill prices")
    total_commission: Decimal = Field(Decimal('0'), description="Total commission paid")
    
    # P&L tracking
    unrealized_pnl: Optional[Decimal] = Field(None, description="Current unrealized P&L")
    realized_pnl: Optional[Decimal] = Field(None, description="Realized P&L if closed")
    
    # Status
    status: str = Field("OPEN", description="Trade status")
    closed_at: Optional[datetime] = Field(None, description="Close timestamp")
    
    class Config:
        validate_assignment = True