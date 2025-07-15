"""
Portfolio models for the options trading engine.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator

from .trades import TradeCandidate, TradeResult, StrategyType
from .options import Greeks


class PortfolioStatus(str, Enum):
    """Portfolio status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


@dataclass
class PortfolioGreeks:
    """Portfolio-level Greeks aggregation."""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    net_rho: float = 0.0
    
    def add_trade_greeks(self, trade_greeks: Greeks, multiplier: int = 1):
        """Add Greeks from a trade to portfolio totals."""
        if trade_greeks.delta:
            self.net_delta += trade_greeks.delta * multiplier
        if trade_greeks.gamma:
            self.net_gamma += trade_greeks.gamma * multiplier
        if trade_greeks.theta:
            self.net_theta += trade_greeks.theta * multiplier
        if trade_greeks.vega:
            self.net_vega += trade_greeks.vega * multiplier
        if trade_greeks.rho:
            self.net_rho += trade_greeks.rho * multiplier
    
    def is_within_limits(self, nav: Decimal, max_delta_multiplier: float = 0.30, 
                        min_vega_multiplier: float = -0.05) -> bool:
        """Check if Greeks are within portfolio limits."""
        nav_factor = float(nav) / 100000  # Scale to $100k base
        
        max_delta = max_delta_multiplier * nav_factor
        min_vega = min_vega_multiplier * nav_factor
        
        delta_within_limits = -max_delta <= self.net_delta <= max_delta
        vega_within_limits = self.net_vega >= min_vega
        
        return delta_within_limits and vega_within_limits


class PortfolioMetrics(BaseModel):
    """Portfolio performance and risk metrics."""
    
    # Portfolio value
    nav: Decimal = Field(..., description="Net Asset Value")
    total_cash: Decimal = Field(..., description="Total cash")
    total_equity: Decimal = Field(..., description="Total equity value")
    buying_power: Decimal = Field(..., description="Available buying power")
    
    # P&L metrics
    total_pnl: Decimal = Field(Decimal('0'), description="Total P&L")
    unrealized_pnl: Decimal = Field(Decimal('0'), description="Unrealized P&L")
    realized_pnl: Decimal = Field(Decimal('0'), description="Realized P&L")
    daily_pnl: Decimal = Field(Decimal('0'), description="Daily P&L")
    
    # Performance metrics
    total_return: Optional[float] = Field(None, description="Total return percentage")
    annualized_return: Optional[float] = Field(None, description="Annualized return")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    
    # Risk metrics
    portfolio_beta: Optional[float] = Field(None, description="Portfolio beta")
    volatility: Optional[float] = Field(None, description="Portfolio volatility")
    var_95: Optional[Decimal] = Field(None, description="95% Value at Risk")
    
    # Greeks
    greeks: PortfolioGreeks = Field(default_factory=PortfolioGreeks, description="Portfolio Greeks")
    
    # Trade statistics
    total_trades: int = Field(0, description="Total number of trades")
    winning_trades: int = Field(0, description="Number of winning trades")
    losing_trades: int = Field(0, description="Number of losing trades")
    win_rate: Optional[float] = Field(None, description="Win rate percentage")
    
    # Sector exposure
    sector_exposure: Dict[str, Decimal] = Field(default_factory=dict, description="Exposure by sector")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    class Config:
        validate_assignment = True
    
    @validator('nav', 'total_cash', 'total_equity', 'buying_power')
    def values_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Portfolio values cannot be negative')
        return v
    
    def calculate_win_rate(self):
        """Calculate win rate from trade statistics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        else:
            self.win_rate = None
    
    def update_sector_exposure(self, sector: str, exposure: Decimal):
        """Update sector exposure."""
        self.sector_exposure[sector] = exposure
    
    @property
    def capital_utilization(self) -> float:
        """Calculate capital utilization percentage."""
        if self.nav == 0:
            return 0.0
        return float((self.nav - self.buying_power) / self.nav)


class Portfolio(BaseModel):
    """Main portfolio container."""
    
    # Portfolio identification
    portfolio_id: str = Field(..., description="Unique portfolio identifier")
    name: str = Field(..., description="Portfolio name")
    owner: str = Field(..., description="Portfolio owner")
    
    # Portfolio configuration
    nav: Decimal = Field(..., description="Net Asset Value")
    status: PortfolioStatus = Field(PortfolioStatus.ACTIVE, description="Portfolio status")
    
    # Risk parameters
    max_loss_per_trade: Decimal = Field(Decimal('500'), description="Max loss per trade")
    max_trades: int = Field(5, description="Maximum concurrent trades")
    max_delta_multiplier: float = Field(0.30, description="Max delta multiplier")
    min_vega_multiplier: float = Field(-0.05, description="Min vega multiplier")
    max_trades_per_sector: int = Field(2, description="Max trades per sector")
    
    # Current holdings
    active_trades: List[TradeResult] = Field(default_factory=list, description="Active trades")
    trade_history: List[TradeResult] = Field(default_factory=list, description="Trade history")
    
    # Portfolio metrics
    metrics: PortfolioMetrics = Field(default_factory=PortfolioMetrics, description="Portfolio metrics")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        validate_assignment = True
    
    @validator('max_trades')
    def max_trades_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Max trades must be positive')
        return v
    
    def add_trade(self, trade_result: TradeResult):
        """Add a new trade to the portfolio."""
        self.active_trades.append(trade_result)
        self.update_metrics()
        self.last_updated = datetime.utcnow()
    
    def close_trade(self, trade_id: str, realized_pnl: Decimal):
        """Close an active trade."""
        for i, trade in enumerate(self.active_trades):
            if trade.trade_candidate.strategy.underlying == trade_id:  # Simplified matching
                trade.realized_pnl = realized_pnl
                trade.status = "CLOSED"
                trade.closed_at = datetime.utcnow()
                
                # Move to history
                self.trade_history.append(self.active_trades.pop(i))
                break
        
        self.update_metrics()
        self.last_updated = datetime.utcnow()
    
    def update_metrics(self):
        """Update portfolio metrics based on current positions."""
        # Reset Greeks
        self.metrics.greeks = PortfolioGreeks()
        
        # Update trade counts
        self.metrics.total_trades = len(self.trade_history) + len(self.active_trades)
        
        # Calculate P&L
        self.metrics.unrealized_pnl = sum(
            trade.unrealized_pnl or Decimal('0') for trade in self.active_trades
        )
        self.metrics.realized_pnl = sum(
            trade.realized_pnl or Decimal('0') for trade in self.trade_history
        )
        self.metrics.total_pnl = self.metrics.unrealized_pnl + self.metrics.realized_pnl
        
        # Update Greeks from active trades
        for trade in self.active_trades:
            if trade.trade_candidate.strategy.net_delta:
                self.metrics.greeks.net_delta += trade.trade_candidate.strategy.net_delta
            if trade.trade_candidate.strategy.net_gamma:
                self.metrics.greeks.net_gamma += trade.trade_candidate.strategy.net_gamma
            if trade.trade_candidate.strategy.net_theta:
                self.metrics.greeks.net_theta += trade.trade_candidate.strategy.net_theta
            if trade.trade_candidate.strategy.net_vega:
                self.metrics.greeks.net_vega += trade.trade_candidate.strategy.net_vega
            if trade.trade_candidate.strategy.net_rho:
                self.metrics.greeks.net_rho += trade.trade_candidate.strategy.net_rho
        
        # Update sector exposure
        sector_exposure = {}
        for trade in self.active_trades:
            sector = trade.trade_candidate.sector or "Unknown"
            capital = trade.trade_candidate.capital_required or Decimal('0')
            sector_exposure[sector] = sector_exposure.get(sector, Decimal('0')) + capital
        
        self.metrics.sector_exposure = sector_exposure
        
        # Calculate win/loss statistics
        winning_trades = sum(1 for trade in self.trade_history 
                           if trade.realized_pnl and trade.realized_pnl > 0)
        losing_trades = sum(1 for trade in self.trade_history 
                          if trade.realized_pnl and trade.realized_pnl <= 0)
        
        self.metrics.winning_trades = winning_trades
        self.metrics.losing_trades = losing_trades
        self.metrics.calculate_win_rate()
        
        # Update NAV and buying power
        self.metrics.nav = self.nav + self.metrics.total_pnl
        
        # Calculate buying power (simplified)
        used_capital = sum(
            trade.trade_candidate.capital_required or Decimal('0') 
            for trade in self.active_trades
        )
        self.metrics.buying_power = self.nav - used_capital
        
        self.metrics.last_updated = datetime.utcnow()
    
    def can_add_trade(self, trade_candidate: TradeCandidate) -> tuple[bool, str]:
        """Check if a trade can be added to the portfolio."""
        # Check maximum trades limit
        if len(self.active_trades) >= self.max_trades:
            return False, f"Maximum trades limit ({self.max_trades}) reached"
        
        # Check capital availability
        if trade_candidate.capital_required and trade_candidate.capital_required > self.metrics.buying_power:
            return False, "Insufficient buying power"
        
        # Check sector diversification
        sector = trade_candidate.sector or "Unknown"
        sector_trade_count = sum(1 for trade in self.active_trades 
                               if trade.trade_candidate.sector == sector)
        if sector_trade_count >= self.max_trades_per_sector:
            return False, f"Maximum trades per sector ({self.max_trades_per_sector}) reached for {sector}"
        
        # Check Greeks limits
        temp_greeks = PortfolioGreeks()
        temp_greeks.net_delta = self.metrics.greeks.net_delta
        temp_greeks.net_vega = self.metrics.greeks.net_vega
        
        if trade_candidate.strategy.net_delta:
            temp_greeks.net_delta += trade_candidate.strategy.net_delta
        if trade_candidate.strategy.net_vega:
            temp_greeks.net_vega += trade_candidate.strategy.net_vega
        
        if not temp_greeks.is_within_limits(self.nav, self.max_delta_multiplier, self.min_vega_multiplier):
            return False, "Trade would violate portfolio Greeks limits"
        
        return True, "Trade can be added"
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for reporting."""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'nav': self.metrics.nav,
            'total_pnl': self.metrics.total_pnl,
            'buying_power': self.metrics.buying_power,
            'active_trades': len(self.active_trades),
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.win_rate,
            'net_delta': self.metrics.greeks.net_delta,
            'net_vega': self.metrics.greeks.net_vega,
            'sector_exposure': dict(self.metrics.sector_exposure),
            'last_updated': self.last_updated
        }