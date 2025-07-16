"""
Portfolio data models for the Options Trading Engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ..validators.portfolio_validators import validate_portfolio_allocation
from ...domain.value_objects.greeks import Greeks
from ...infrastructure.error_handling import ValidationError


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"


class AssetType(Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    OPTION = "option"
    ETF = "etf"
    CASH = "cash"


@dataclass
class Position:
    """Individual position within a portfolio."""
    symbol: str
    asset_type: AssetType
    position_type: PositionType
    quantity: int
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    entry_date: datetime
    last_updated: datetime
    
    # Option-specific fields
    strike: Optional[float] = None
    expiration_date: Optional[datetime] = None
    option_type: Optional[str] = None
    
    # Greeks for option positions
    greeks: Optional[Greeks] = None
    
    # Risk metrics
    beta: Optional[float] = None
    correlation: Optional[float] = None
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def allocation_percentage(self) -> float:
        """Get position allocation as percentage."""
        # This would be calculated against total portfolio value
        return 0.0  # Placeholder
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type.value,
            'position_type': self.position_type.value,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'entry_date': self.entry_date.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'strike': self.strike,
            'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
            'option_type': self.option_type,
            'greeks': self.greeks.to_dict() if self.greeks else None,
            'beta': self.beta,
            'correlation': self.correlation
        }


@dataclass
class Portfolio:
    """Portfolio containing multiple positions."""
    portfolio_id: str
    name: str
    positions: List[Position] = field(default_factory=list)
    cash_balance: float = 0.0
    total_nav: float = 0.0
    available_capital: float = 0.0
    margin_used: float = 0.0
    buying_power: float = 0.0
    
    # Aggregate metrics
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Risk metrics
    portfolio_beta: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    
    # Greeks aggregation
    aggregate_greeks: Greeks = field(default_factory=Greeks.zero)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.calculate_aggregate_metrics()
    
    def add_position(self, position: Position):
        """Add a position to the portfolio."""
        self.positions.append(position)
        self.calculate_aggregate_metrics()
        self.last_updated = datetime.now()
    
    def remove_position(self, symbol: str):
        """Remove a position from the portfolio."""
        self.positions = [p for p in self.positions if p.symbol != symbol]
        self.calculate_aggregate_metrics()
        self.last_updated = datetime.now()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position by symbol."""
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def calculate_aggregate_metrics(self):
        """Calculate aggregate portfolio metrics."""
        # Calculate total market value
        total_market_value = sum(pos.market_value for pos in self.positions)
        
        # Calculate total P&L
        self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)
        self.total_realized_pnl = sum(pos.realized_pnl for pos in self.positions)
        
        # Calculate NAV
        self.total_nav = total_market_value + self.cash_balance
        
        # Calculate available capital
        self.available_capital = self.total_nav - self.margin_used
        
        # Aggregate Greeks
        greeks_list = [pos.greeks for pos in self.positions if pos.greeks is not None]
        if greeks_list:
            self.aggregate_greeks = sum(greeks_list, Greeks.zero())
        else:
            self.aggregate_greeks = Greeks.zero()
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get allocation by sector."""
        # This would require sector information for each position
        # Placeholder implementation
        return {}
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Get allocation by asset type."""
        allocations = {}
        total_value = max(self.total_nav, 1.0)  # Avoid division by zero
        
        for asset_type in AssetType:
            positions = [p for p in self.positions if p.asset_type == asset_type]
            total_allocation = sum(p.market_value for p in positions)
            allocations[asset_type.value] = (total_allocation / total_value) * 100
        
        # Add cash allocation
        allocations['cash'] = (self.cash_balance / total_value) * 100
        
        return allocations
    
    def get_top_positions(self, count: int = 10) -> List[Position]:
        """Get top positions by market value."""
        sorted_positions = sorted(self.positions, key=lambda p: abs(p.market_value), reverse=True)
        return sorted_positions[:count]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get portfolio risk metrics."""
        return {
            'portfolio_beta': self.portfolio_beta,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'max_drawdown': self.max_drawdown,
            'aggregate_greeks': self.aggregate_greeks.to_dict(),
            'concentration_risk': self.calculate_concentration_risk(),
            'liquidity_risk': self.calculate_liquidity_risk()
        }
    
    def calculate_concentration_risk(self) -> float:
        """Calculate concentration risk score."""
        if not self.positions or self.total_nav <= 0:
            return 0.0
        
        # Calculate Herfindahl index
        concentrations = []
        for position in self.positions:
            concentration = abs(position.market_value) / self.total_nav
            concentrations.append(concentration ** 2)
        
        herfindahl_index = sum(concentrations)
        return herfindahl_index
    
    def calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk score."""
        # Placeholder implementation
        # Would consider factors like:
        # - Bid-ask spreads
        # - Volume
        # - Market cap
        # - Asset type
        return 0.0
    
    @validate_portfolio_allocation
    def validate_portfolio_constraints(self) -> Dict[str, Any]:
        """Validate portfolio against constraints."""
        constraints = {}
        
        # Maximum position size constraint
        max_position_size = 0.10  # 10% max per position
        for position in self.positions:
            allocation = abs(position.market_value) / max(self.total_nav, 1.0)
            if allocation > max_position_size:
                constraints[f'max_position_{position.symbol}'] = {
                    'violated': True,
                    'current': allocation,
                    'limit': max_position_size
                }
        
        # Greek limits
        max_delta = 500
        max_gamma = 100
        max_theta = -200
        max_vega = 1000
        
        if abs(self.aggregate_greeks.delta) > max_delta:
            constraints['max_delta'] = {
                'violated': True,
                'current': self.aggregate_greeks.delta,
                'limit': max_delta
            }
        
        if self.aggregate_greeks.gamma > max_gamma:
            constraints['max_gamma'] = {
                'violated': True,
                'current': self.aggregate_greeks.gamma,
                'limit': max_gamma
            }
        
        if self.aggregate_greeks.theta < max_theta:
            constraints['max_theta'] = {
                'violated': True,
                'current': self.aggregate_greeks.theta,
                'limit': max_theta
            }
        
        if self.aggregate_greeks.vega > max_vega:
            constraints['max_vega'] = {
                'violated': True,
                'current': self.aggregate_greeks.vega,
                'limit': max_vega
            }
        
        return constraints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'positions': [pos.to_dict() for pos in self.positions],
            'cash_balance': self.cash_balance,
            'total_nav': self.total_nav,
            'available_capital': self.available_capital,
            'margin_used': self.margin_used,
            'buying_power': self.buying_power,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'daily_pnl': self.daily_pnl,
            'portfolio_beta': self.portfolio_beta,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'max_drawdown': self.max_drawdown,
            'aggregate_greeks': self.aggregate_greeks.to_dict(),
            'asset_allocation': self.get_asset_allocation(),
            'sector_allocation': self.get_sector_allocation(),
            'top_positions': [pos.to_dict() for pos in self.get_top_positions()],
            'risk_metrics': self.get_risk_metrics(),
            'constraint_violations': self.validate_portfolio_constraints(),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create from dictionary."""
        positions = [Position(**pos_data) for pos_data in data.get('positions', [])]
        
        portfolio = cls(
            portfolio_id=data['portfolio_id'],
            name=data['name'],
            positions=positions,
            cash_balance=data.get('cash_balance', 0.0),
            total_nav=data.get('total_nav', 0.0),
            available_capital=data.get('available_capital', 0.0),
            margin_used=data.get('margin_used', 0.0),
            buying_power=data.get('buying_power', 0.0),
            total_unrealized_pnl=data.get('total_unrealized_pnl', 0.0),
            total_realized_pnl=data.get('total_realized_pnl', 0.0),
            daily_pnl=data.get('daily_pnl', 0.0),
            portfolio_beta=data.get('portfolio_beta', 0.0),
            var_95=data.get('var_95', 0.0),
            expected_shortfall=data.get('expected_shortfall', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )
        
        return portfolio
    
    def __str__(self) -> str:
        """String representation."""
        return f"Portfolio({self.name}, NAV=${self.total_nav:,.2f}, Positions={len(self.positions)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Portfolio(id={self.portfolio_id}, name={self.name}, nav={self.total_nav}, positions={len(self.positions)})"