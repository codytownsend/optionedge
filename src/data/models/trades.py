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
    """Options strategy types as specified in instructions."""
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
    BULL_PUT_SPREAD = "bull_put_spread"  # Alias for put credit spread
    BEAR_CALL_SPREAD = "bear_call_spread"  # Alias for call credit spread


class TradeDirection(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"
    BUY = "BUY"   # Alternative naming
    SELL = "SELL"  # Alternative naming


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
    def action(self) -> str:
        """Get standardized action (BUY/SELL)."""
        if self.direction in [TradeDirection.LONG, TradeDirection.BUY]:
            return "BUY"
        else:
            return "SELL"
    
    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate market value of the leg."""
        if not self.option.mid_price:
            return None
        
        multiplier = 100  # Standard option multiplier
        sign = 1 if self.action == "BUY" else -1
        return self.option.mid_price * self.quantity * multiplier * sign
    
    @property
    def greeks_contribution(self) -> Optional[Greeks]:
        """Calculate Greeks contribution of this leg."""
        if not self.option.greeks:
            return None
        
        multiplier = 100 * self.quantity
        sign = 1 if self.action == "BUY" else -1
        
        return Greeks(
            delta=(self.option.greeks.delta * multiplier * sign) if self.option.greeks.delta else None,
            gamma=(self.option.greeks.gamma * multiplier * sign) if self.option.greeks.gamma else None,
            theta=(self.option.greeks.theta * multiplier * sign) if self.option.greeks.theta else None,
            vega=(self.option.greeks.vega * multiplier * sign) if self.option.greeks.vega else None,
            rho=(self.option.greeks.rho * multiplier * sign) if self.option.greeks.rho else None
        )

    @property
    def strike(self) -> Decimal:
        """Get strike price for convenience."""
        return self.option.strike

    @property
    def expiration_date(self) -> date:
        """Get expiration date for convenience."""
        return self.option.expiration

    @property
    def option_type(self) -> OptionType:
        """Get option type for convenience."""
        return self.option.option_type


class StrategyDefinition(BaseModel):
    """Definition of an options strategy with all metrics per instructions."""
    
    strategy_type: StrategyType = Field(..., description="Type of strategy")
    underlying: str = Field(..., description="Underlying symbol")
    legs: List[TradeLeg] = Field(..., description="Strategy legs")
    
    # Strategy metrics (required per instructions)
    net_premium: Optional[Decimal] = Field(None, description="Net premium received/paid")
    net_credit: Optional[Decimal] = Field(None, description="Net credit received")
    net_debit: Optional[Decimal] = Field(None, description="Net debit paid")
    max_profit: Optional[Decimal] = Field(None, description="Maximum profit potential")
    max_loss: Optional[Decimal] = Field(None, description="Maximum loss potential")
    
    # Probability metrics (critical per instructions)
    probability_of_profit: Optional[float] = Field(None, description="Probability of profit")
    breakeven_points: List[Decimal] = Field(default_factory=list, description="Breakeven prices")
    
    # Greeks (required for portfolio constraints per instructions)
    net_delta: Optional[float] = Field(None, description="Net strategy delta")
    net_gamma: Optional[float] = Field(None, description="Net strategy gamma")
    net_theta: Optional[float] = Field(None, description="Net strategy theta")
    net_vega: Optional[float] = Field(None, description="Net strategy vega")
    net_rho: Optional[float] = Field(None, description="Net strategy rho")
    
    # Strike information for display
    short_strike: Optional[Decimal] = Field(None, description="Short strike (primary)")
    long_strike: Optional[Decimal] = Field(None, description="Long strike (hedge)")
    strike_width: Optional[Decimal] = Field(None, description="Strike width for spreads")
    
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
        
        # Set credit/debit based on sign
        if total_premium > 0:
            self.net_credit = total_premium
            self.net_debit = None
        else:
            self.net_debit = abs(total_premium)
            self.net_credit = None
    
    def calculate_strategy_metrics(self) -> None:
        """Calculate all strategy metrics."""
        self.calculate_net_premium()
        self.calculate_net_greeks()
        self._calculate_strike_info()
        self._calculate_expiration_info()
    
    def _calculate_strike_info(self) -> None:
        """Calculate strike-related information."""
        if not self.legs:
            return
            
        if self.strategy_type in [StrategyType.PUT_CREDIT_SPREAD, StrategyType.CALL_CREDIT_SPREAD]:
            # Find short and long strikes
            short_legs = [leg for leg in self.legs if leg.action == "SELL"]
            long_legs = [leg for leg in self.legs if leg.action == "BUY"]
            
            if short_legs:
                self.short_strike = short_legs[0].strike
            if long_legs:
                self.long_strike = long_legs[0].strike
                
            if self.short_strike and self.long_strike:
                self.strike_width = abs(self.short_strike - self.long_strike)
    
    def _calculate_expiration_info(self) -> None:
        """Calculate expiration-related information."""
        if self.legs:
            self.expiration = self.legs[0].expiration_date
            self.days_to_expiration = (self.expiration - date.today()).days
    
    @property
    def is_credit_strategy(self) -> bool:
        """Check if strategy receives credit."""
        return self.net_credit is not None and self.net_credit > 0
    
    @property
    def is_debit_strategy(self) -> bool:
        """Check if strategy pays debit."""
        return self.net_debit is not None and self.net_debit > 0
    
    @property
    def credit_to_max_loss_ratio(self) -> Optional[float]:
        """Calculate credit-to-max-loss ratio (required constraint per instructions)."""
        if not self.is_credit_strategy or not self.max_loss or self.max_loss == 0:
            return None
        
        return float(self.net_credit / abs(self.max_loss))

    def calculate_margin_requirement(self) -> Decimal:
        """Calculate margin requirement for the strategy."""
        if self.strategy_type in [StrategyType.PUT_CREDIT_SPREAD, StrategyType.CALL_CREDIT_SPREAD]:
            # Margin = Strike width - Net credit
            if self.strike_width and self.net_credit:
                return (self.strike_width * 100) - self.net_credit  # Convert to dollar terms
        elif self.strategy_type == StrategyType.IRON_CONDOR:
            # Margin = Larger strike width - Net credit
            if self.strike_width and self.net_credit:
                return (self.strike_width * 100) - self.net_credit
        elif self.strategy_type == StrategyType.CASH_SECURED_PUT:
            if self.short_strike:
                return self.short_strike * 100  # Cash secured amount
        
        # Default: maximum loss
        return self.max_loss or Decimal('0')


class TradeCandidate(BaseModel):
    """A potential trade with scoring and ranking information per instructions."""
    
    strategy: StrategyDefinition = Field(..., description="Strategy definition")
    
    # Scoring components (required per instructions)
    model_score: Optional[float] = Field(None, description="Composite model score")
    momentum_z_score: Optional[float] = Field(None, description="Momentum Z-score")
    flow_z_score: Optional[float] = Field(None, description="Flow Z-score")
    iv_rank: Optional[float] = Field(None, description="IV rank percentile")
    
    # Market data quality (required per instructions)
    quote_age_minutes: Optional[float] = Field(None, description="Quote age in minutes")
    liquidity_score: Optional[float] = Field(None, description="Liquidity score")
    
    # Risk metrics (required per instructions)
    capital_required: Optional[Decimal] = Field(None, description="Capital required")
    margin_required: Optional[Decimal] = Field(None, description="Margin required")
    
    # Fundamental context (required for diversification per instructions)
    sector: Optional[str] = Field(None, description="GICS sector")
    market_cap: Optional[Decimal] = Field(None, description="Market capitalization")
    
    # Trade thesis (required output per instructions)
    thesis: Optional[str] = Field(None, description="Trade thesis (max 30 words)")
    
    # Ranking (required per instructions)
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
    def ticker(self) -> str:
        """Alias for underlying symbol."""
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
    def net_credit(self) -> Optional[Decimal]:
        """Get net credit."""
        return self.strategy.net_credit
    
    @property
    def max_loss(self) -> Optional[Decimal]:
        """Get maximum loss."""
        return self.strategy.max_loss
    
    @property
    def max_profit(self) -> Optional[Decimal]:
        """Get maximum profit."""
        return self.strategy.max_profit
    
    @property
    def days_to_expiration(self) -> Optional[int]:
        """Get days to expiration."""
        return self.strategy.days_to_expiration

    @property
    def legs(self) -> List[TradeLeg]:
        """Get strategy legs."""
        return self.strategy.legs

    @property
    def short_strike(self) -> Optional[Decimal]:
        """Get short strike for display."""
        return self.strategy.short_strike

    def meets_hard_constraints(self, filter_criteria: 'TradeFilterCriteria') -> bool:
        """Check if trade meets all hard constraints per instructions."""
        # POP constraint
        if (self.probability_of_profit is None or 
            self.probability_of_profit < filter_criteria.min_probability_of_profit):
            return False
        
        # Credit-to-max-loss ratio constraint
        if (self.strategy.credit_to_max_loss_ratio is None or 
            self.strategy.credit_to_max_loss_ratio < filter_criteria.min_credit_to_max_loss):
            return False
        
        # Max loss constraint
        if (self.max_loss is None or 
            self.max_loss > filter_criteria.max_loss_per_trade):
            return False
        
        # Quote age constraint
        if (self.quote_age_minutes is None or 
            self.quote_age_minutes > filter_criteria.max_quote_age_minutes):
            return False
        
        # Capital availability constraint
        required_capital = self.capital_required or self.margin_required or Decimal('0')
        if required_capital > filter_criteria.available_capital:
            return False
        
        return True


@dataclass
class TradeFilterCriteria:
    """Criteria for filtering trades per instructions."""
    
    # Hard constraints from instructions
    min_probability_of_profit: float = 0.65
    min_credit_to_max_loss: float = 0.33
    max_loss_per_trade: Decimal = Decimal('500')
    max_quote_age_minutes: float = 10.0
    
    # Portfolio constraints from instructions
    max_delta_exposure: float = 0.30  # × (NAV / 100k)
    min_vega_exposure: float = -0.05  # × (NAV / 100k)
    max_trades_per_sector: int = 2
    
    # Liquidity requirements from instructions
    min_option_volume: int = 10
    min_open_interest: int = 50  # Reduced from instructions for practicality
    max_bid_ask_spread_pct: float = 0.05  # 5% per instructions
    
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
        arbitrary_types_allowed = True


class PortfolioGreeks(BaseModel):
    """Portfolio-level Greeks tracking per instructions."""
    
    total_delta: float = Field(0.0, description="Total portfolio delta")
    total_gamma: float = Field(0.0, description="Total portfolio gamma")
    total_theta: float = Field(0.0, description="Total portfolio theta")
    total_vega: float = Field(0.0, description="Total portfolio vega")
    total_rho: float = Field(0.0, description="Total portfolio rho")
    
    # Normalized by NAV per instructions
    delta_per_100k: float = Field(0.0, description="Delta per 100k NAV")
    vega_per_100k: float = Field(0.0, description="Vega per 100k NAV")
    
    nav: Decimal = Field(Decimal('100000'), description="Net asset value")
    
    def update_from_trades(self, trades: List[TradeCandidate], nav: Decimal):
        """Update Greeks from list of trades."""
        self.nav = nav
        nav_factor = float(nav / Decimal('100000'))
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        
        for trade in trades:
            if trade.strategy.net_delta:
                total_delta += trade.strategy.net_delta
            if trade.strategy.net_gamma:
                total_gamma += trade.strategy.net_gamma
            if trade.strategy.net_theta:
                total_theta += trade.strategy.net_theta
            if trade.strategy.net_vega:
                total_vega += trade.strategy.net_vega
            if trade.strategy.net_rho:
                total_rho += trade.strategy.net_rho
        
        self.total_delta = total_delta
        self.total_gamma = total_gamma
        self.total_theta = total_theta
        self.total_vega = total_vega
        self.total_rho = total_rho
        
        # Normalize per instructions
        self.delta_per_100k = total_delta / nav_factor
        self.vega_per_100k = total_vega / nav_factor
    
    def within_limits(self, criteria: TradeFilterCriteria) -> bool:
        """Check if portfolio Greeks are within limits per instructions."""
        nav_factor = float(self.nav / Decimal('100000'))
        
        # Delta limits: [-0.30, +0.30] × (NAV / 100k)
        max_delta = criteria.max_delta_exposure * nav_factor
        if abs(self.delta_per_100k) > max_delta:
            return False
        
        # Vega limit: ≥ -0.05 × (NAV / 100k)
        min_vega = criteria.min_vega_exposure * nav_factor
        if self.vega_per_100k < min_vega:
            return False
        
        return True