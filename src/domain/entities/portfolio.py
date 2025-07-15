"""
Portfolio domain entities for position management and risk control.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass

from .strategy import Strategy
from ...data.models.portfolios import Portfolio as PortfolioModel, PortfolioGreeks, PortfolioMetrics
from ...data.models.trades import TradeCandidate, TradeResult, StrategyType


class PortfolioConstraintType(str, Enum):
    """Types of portfolio constraints."""
    MAX_TRADES = "max_trades"
    MAX_LOSS_PER_TRADE = "max_loss_per_trade"
    MAX_DELTA_EXPOSURE = "max_delta_exposure"
    MIN_VEGA_EXPOSURE = "min_vega_exposure"
    MAX_SECTOR_EXPOSURE = "max_sector_exposure"
    MIN_LIQUIDITY = "min_liquidity"
    MAX_QUOTE_AGE = "max_quote_age"
    CAPITAL_LIMITS = "capital_limits"


class ConstraintViolationType(str, Enum):
    """Types of constraint violations."""
    HARD_VIOLATION = "hard_violation"    # Must be rejected
    SOFT_VIOLATION = "soft_violation"    # Warning but can proceed
    INFO_VIOLATION = "info_violation"    # Informational only


@dataclass
class ConstraintViolation:
    """Represents a portfolio constraint violation."""
    constraint_type: PortfolioConstraintType
    violation_type: ConstraintViolationType
    message: str
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    trade_candidate: Optional[TradeCandidate] = None


class PortfolioConstraints:
    """Portfolio constraints configuration and validation."""
    
    def __init__(self,
                 nav: Decimal,
                 max_trades: int = 5,
                 max_loss_per_trade: Decimal = Decimal('500'),
                 max_delta_multiplier: float = 0.30,
                 min_vega_multiplier: float = -0.05,
                 max_trades_per_sector: int = 2,
                 min_option_volume: int = 10,
                 min_open_interest: int = 100,
                 max_bid_ask_spread_pct: float = 0.50,
                 max_quote_age_minutes: float = 10.0):
        """
        Initialize portfolio constraints.
        
        Args:
            nav: Net Asset Value for scaling calculations
            max_trades: Maximum number of concurrent trades
            max_loss_per_trade: Maximum loss per individual trade
            max_delta_multiplier: Maximum delta exposure multiplier
            min_vega_multiplier: Minimum vega exposure multiplier
            max_trades_per_sector: Maximum trades per GICS sector
            min_option_volume: Minimum option volume requirement
            min_open_interest: Minimum open interest requirement
            max_bid_ask_spread_pct: Maximum bid-ask spread percentage
            max_quote_age_minutes: Maximum quote age in minutes
        """
        self.nav = nav
        self.max_trades = max_trades
        self.max_loss_per_trade = max_loss_per_trade
        self.max_delta_multiplier = max_delta_multiplier
        self.min_vega_multiplier = min_vega_multiplier
        self.max_trades_per_sector = max_trades_per_sector
        self.min_option_volume = min_option_volume
        self.min_open_interest = min_open_interest
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.max_quote_age_minutes = max_quote_age_minutes
    
    @property
    def max_delta_exposure(self) -> float:
        """Calculate maximum delta exposure based on NAV."""
        return self.max_delta_multiplier * (float(self.nav) / 100000)
    
    @property
    def min_vega_exposure(self) -> float:
        """Calculate minimum vega exposure based on NAV."""
        return self.min_vega_multiplier * (float(self.nav) / 100000)
    
    def validate_trade(self, 
                      trade_candidate: TradeCandidate,
                      current_portfolio: 'Position') -> List[ConstraintViolation]:
        """
        Validate a trade candidate against portfolio constraints.
        
        Args:
            trade_candidate: Trade to validate
            current_portfolio: Current portfolio state
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Max trades constraint
        if len(current_portfolio.active_trades) >= self.max_trades:
            violations.append(ConstraintViolation(
                constraint_type=PortfolioConstraintType.MAX_TRADES,
                violation_type=ConstraintViolationType.HARD_VIOLATION,
                message=f"Maximum trades limit ({self.max_trades}) would be exceeded",
                current_value=len(current_portfolio.active_trades) + 1,
                limit_value=self.max_trades,
                trade_candidate=trade_candidate
            ))
        
        # Max loss per trade
        if trade_candidate.max_loss and abs(trade_candidate.max_loss) > self.max_loss_per_trade:
            violations.append(ConstraintViolation(
                constraint_type=PortfolioConstraintType.MAX_LOSS_PER_TRADE,
                violation_type=ConstraintViolationType.HARD_VIOLATION,
                message=f"Trade max loss {abs(trade_candidate.max_loss)} exceeds limit {self.max_loss_per_trade}",
                current_value=float(abs(trade_candidate.max_loss)),
                limit_value=float(self.max_loss_per_trade),
                trade_candidate=trade_candidate
            ))
        
        # Greeks constraints
        violations.extend(self._validate_greeks_constraints(trade_candidate, current_portfolio))
        
        # Sector diversification
        violations.extend(self._validate_sector_constraints(trade_candidate, current_portfolio))
        
        # Liquidity constraints
        violations.extend(self._validate_liquidity_constraints(trade_candidate))
        
        # Capital constraints
        violations.extend(self._validate_capital_constraints(trade_candidate, current_portfolio))
        
        return violations
    
    def _validate_greeks_constraints(self, 
                                   trade_candidate: TradeCandidate,
                                   current_portfolio: 'Position') -> List[ConstraintViolation]:
        """Validate Greeks constraints."""
        violations = []
        
        # Calculate new Greeks after adding trade
        current_delta = current_portfolio.get_net_delta()
        current_vega = current_portfolio.get_net_vega()
        
        trade_delta = trade_candidate.strategy.net_delta or 0
        trade_vega = trade_candidate.strategy.net_vega or 0
        
        new_delta = current_delta + trade_delta
        new_vega = current_vega + trade_vega
        
        # Delta constraint
        max_delta = self.max_delta_exposure
        if abs(new_delta) > max_delta:
            violations.append(ConstraintViolation(
                constraint_type=PortfolioConstraintType.MAX_DELTA_EXPOSURE,
                violation_type=ConstraintViolationType.HARD_VIOLATION,
                message=f"Delta exposure {new_delta:.2f} would exceed limit ±{max_delta:.2f}",
                current_value=abs(new_delta),
                limit_value=max_delta,
                trade_candidate=trade_candidate
            ))
        
        # Vega constraint
        min_vega = self.min_vega_exposure
        if new_vega < min_vega:
            violations.append(ConstraintViolation(
                constraint_type=PortfolioConstraintType.MIN_VEGA_EXPOSURE,
                violation_type=ConstraintViolationType.HARD_VIOLATION,
                message=f"Vega exposure {new_vega:.2f} would be below limit {min_vega:.2f}",
                current_value=new_vega,
                limit_value=min_vega,
                trade_candidate=trade_candidate
            ))
        
        return violations
    
    def _validate_sector_constraints(self,
                                   trade_candidate: TradeCandidate,
                                   current_portfolio: 'Position') -> List[ConstraintViolation]:
        """Validate sector diversification constraints."""
        violations = []
        
        trade_sector = trade_candidate.sector or "Unknown"
        sector_count = current_portfolio.get_trades_by_sector().get(trade_sector, 0)
        
        if sector_count >= self.max_trades_per_sector:
            violations.append(ConstraintViolation(
                constraint_type=PortfolioConstraintType.MAX_SECTOR_EXPOSURE,
                violation_type=ConstraintViolationType.HARD_VIOLATION,
                message=f"Maximum trades per sector ({self.max_trades_per_sector}) exceeded for {trade_sector}",
                current_value=sector_count + 1,
                limit_value=self.max_trades_per_sector,
                trade_candidate=trade_candidate
            ))
        
        return violations
    
    def _validate_liquidity_constraints(self, trade_candidate: TradeCandidate) -> List[ConstraintViolation]:
        """Validate liquidity constraints."""
        violations = []
        
        # Check each leg for liquidity
        for leg in trade_candidate.strategy.legs:
            option = leg.option
            
            # Volume check
            if option.volume is not None and option.volume < self.min_option_volume:
                violations.append(ConstraintViolation(
                    constraint_type=PortfolioConstraintType.MIN_LIQUIDITY,
                    violation_type=ConstraintViolationType.SOFT_VIOLATION,
                    message=f"Option {option.symbol} volume {option.volume} below minimum {self.min_option_volume}",
                    current_value=option.volume,
                    limit_value=self.min_option_volume,
                    trade_candidate=trade_candidate
                ))
            
            # Open interest check
            if option.open_interest is not None and option.open_interest < self.min_open_interest:
                violations.append(ConstraintViolation(
                    constraint_type=PortfolioConstraintType.MIN_LIQUIDITY,
                    violation_type=ConstraintViolationType.SOFT_VIOLATION,
                    message=f"Option {option.symbol} open interest {option.open_interest} below minimum {self.min_open_interest}",
                    current_value=option.open_interest,
                    limit_value=self.min_open_interest,
                    trade_candidate=trade_candidate
                ))
            
            # Spread check
            spread_pct = option.bid_ask_spread_percent
            if spread_pct is not None and spread_pct > self.max_bid_ask_spread_pct:
                violations.append(ConstraintViolation(
                    constraint_type=PortfolioConstraintType.MIN_LIQUIDITY,
                    violation_type=ConstraintViolationType.SOFT_VIOLATION,
                    message=f"Option {option.symbol} spread {spread_pct:.1%} exceeds maximum {self.max_bid_ask_spread_pct:.1%}",
                    current_value=spread_pct * 100,
                    limit_value=self.max_bid_ask_spread_pct * 100,
                    trade_candidate=trade_candidate
                ))
        
        # Quote age check
        if trade_candidate.quote_age_minutes is not None:
            if trade_candidate.quote_age_minutes > self.max_quote_age_minutes:
                violations.append(ConstraintViolation(
                    constraint_type=PortfolioConstraintType.MAX_QUOTE_AGE,
                    violation_type=ConstraintViolationType.HARD_VIOLATION,
                    message=f"Quote age {trade_candidate.quote_age_minutes:.1f} minutes exceeds maximum {self.max_quote_age_minutes}",
                    current_value=trade_candidate.quote_age_minutes,
                    limit_value=self.max_quote_age_minutes,
                    trade_candidate=trade_candidate
                ))
        
        return violations
    
    def _validate_capital_constraints(self,
                                    trade_candidate: TradeCandidate,
                                    current_portfolio: 'Position') -> List[ConstraintViolation]:
        """Validate capital and margin constraints."""
        violations = []
        
        # Check available capital
        available_capital = current_portfolio.get_available_capital()
        required_capital = trade_candidate.capital_required or Decimal('0')
        
        if required_capital > available_capital:
            violations.append(ConstraintViolation(
                constraint_type=PortfolioConstraintType.CAPITAL_LIMITS,
                violation_type=ConstraintViolationType.HARD_VIOLATION,
                message=f"Required capital {required_capital} exceeds available {available_capital}",
                current_value=float(required_capital),
                limit_value=float(available_capital),
                trade_candidate=trade_candidate
            ))
        
        return violations


class Position:
    """
    Domain entity representing the current portfolio position.
    
    This class manages the current state of the portfolio including
    active trades, risk metrics, and constraint validation.
    """
    
    def __init__(self, 
                 portfolio_model: PortfolioModel,
                 constraints: PortfolioConstraints):
        """
        Initialize position.
        
        Args:
            portfolio_model: Underlying portfolio data model
            constraints: Portfolio constraints configuration
        """
        self._portfolio_model = portfolio_model
        self.constraints = constraints
        self._cached_calculations: Dict[str, any] = {}
    
    @property
    def portfolio_id(self) -> str:
        """Get portfolio ID."""
        return self._portfolio_model.portfolio_id
    
    @property
    def nav(self) -> Decimal:
        """Get current NAV."""
        return self._portfolio_model.metrics.nav
    
    @property
    def active_trades(self) -> List[TradeResult]:
        """Get active trades."""
        return self._portfolio_model.active_trades
    
    def can_add_trade(self, trade_candidate: TradeCandidate) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Check if a trade can be added to the portfolio.
        
        Args:
            trade_candidate: Trade to evaluate
            
        Returns:
            Tuple of (can_add, violations_list)
        """
        violations = self.constraints.validate_trade(trade_candidate, self)
        
        # Check if there are any hard violations
        hard_violations = [v for v in violations if v.violation_type == ConstraintViolationType.HARD_VIOLATION]
        can_add = len(hard_violations) == 0
        
        return can_add, violations
    
    def add_trade(self, trade_result: TradeResult) -> None:
        """Add a trade to the portfolio."""
        self._portfolio_model.add_trade(trade_result)
        self._clear_cache()
    
    def close_trade(self, trade_id: str, realized_pnl: Decimal) -> None:
        """Close a trade in the portfolio."""
        self._portfolio_model.close_trade(trade_id, realized_pnl)
        self._clear_cache()
    
    def get_net_delta(self) -> float:
        """Get net delta exposure."""
        cache_key = "net_delta"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        total_delta = 0.0
        for trade in self.active_trades:
            if trade.trade_candidate.strategy.net_delta:
                total_delta += trade.trade_candidate.strategy.net_delta
        
        self._cached_calculations[cache_key] = total_delta
        return total_delta
    
    def get_net_vega(self) -> float:
        """Get net vega exposure."""
        cache_key = "net_vega"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        total_vega = 0.0
        for trade in self.active_trades:
            if trade.trade_candidate.strategy.net_vega:
                total_vega += trade.trade_candidate.strategy.net_vega
        
        self._cached_calculations[cache_key] = total_vega
        return total_vega
    
    def get_net_theta(self) -> float:
        """Get net theta exposure."""
        cache_key = "net_theta"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        total_theta = 0.0
        for trade in self.active_trades:
            if trade.trade_candidate.strategy.net_theta:
                total_theta += trade.trade_candidate.strategy.net_theta
        
        self._cached_calculations[cache_key] = total_theta
        return total_theta
    
    def get_trades_by_sector(self) -> Dict[str, int]:
        """Get count of trades by sector."""
        cache_key = "trades_by_sector"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        sector_counts = {}
        for trade in self.active_trades:
            sector = trade.trade_candidate.sector or "Unknown"
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        self._cached_calculations[cache_key] = sector_counts
        return sector_counts
    
    def get_trades_by_strategy_type(self) -> Dict[StrategyType, int]:
        """Get count of trades by strategy type."""
        cache_key = "trades_by_strategy"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        strategy_counts = {}
        for trade in self.active_trades:
            strategy_type = trade.trade_candidate.strategy_type
            strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
        
        self._cached_calculations[cache_key] = strategy_counts
        return strategy_counts
    
    def get_available_capital(self) -> Decimal:
        """Get available capital for new trades."""
        return self._portfolio_model.metrics.buying_power
    
    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L."""
        return self._portfolio_model.metrics.unrealized_pnl
    
    def get_total_realized_pnl(self) -> Decimal:
        """Get total realized P&L."""
        return self._portfolio_model.metrics.realized_pnl
    
    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get portfolio-level Greeks."""
        return self._portfolio_model.metrics.greeks
    
    def calculate_utilization_metrics(self) -> Dict[str, float]:
        """Calculate portfolio utilization metrics."""
        cache_key = "utilization_metrics"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        max_trades = self.constraints.max_trades
        current_trades = len(self.active_trades)
        
        # Trade count utilization
        trade_utilization = current_trades / max_trades if max_trades > 0 else 0
        
        # Capital utilization
        total_nav = float(self.nav)
        available_capital = float(self.get_available_capital())
        capital_utilization = (total_nav - available_capital) / total_nav if total_nav > 0 else 0
        
        # Greeks utilization
        max_delta = self.constraints.max_delta_exposure
        current_delta = abs(self.get_net_delta())
        delta_utilization = current_delta / max_delta if max_delta > 0 else 0
        
        metrics = {
            'trade_utilization': trade_utilization,
            'capital_utilization': capital_utilization,
            'delta_utilization': delta_utilization,
        }
        
        self._cached_calculations[cache_key] = metrics
        return metrics
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get comprehensive risk summary."""
        utilization = self.calculate_utilization_metrics()
        
        return {
            'portfolio_id': self.portfolio_id,
            'nav': self.nav,
            'active_trades': len(self.active_trades),
            'available_capital': self.get_available_capital(),
            'unrealized_pnl': self.get_total_unrealized_pnl(),
            'realized_pnl': self.get_total_realized_pnl(),
            'net_delta': self.get_net_delta(),
            'net_vega': self.get_net_vega(),
            'net_theta': self.get_net_theta(),
            'trades_by_sector': self.get_trades_by_sector(),
            'trades_by_strategy': self.get_trades_by_strategy_type(),
            'utilization_metrics': utilization,
            'constraints': {
                'max_trades': self.constraints.max_trades,
                'max_delta_exposure': self.constraints.max_delta_exposure,
                'min_vega_exposure': self.constraints.min_vega_exposure,
                'max_trades_per_sector': self.constraints.max_trades_per_sector,
            }
        }
    
    def _clear_cache(self) -> None:
        """Clear cached calculations."""
        self._cached_calculations.clear()


class PortfolioManager:
    """
    Portfolio manager for handling portfolio operations and risk management.
    
    This class provides high-level portfolio management operations
    including trade selection, risk monitoring, and constraint enforcement.
    """
    
    def __init__(self, position: Position):
        """
        Initialize portfolio manager.
        
        Args:
            position: Current portfolio position
        """
        self.position = position
    
    def evaluate_trade_candidates(self, 
                                candidates: List[TradeCandidate]) -> List[Tuple[TradeCandidate, bool, List[ConstraintViolation]]]:
        """
        Evaluate multiple trade candidates against portfolio constraints.
        
        Args:
            candidates: List of trade candidates to evaluate
            
        Returns:
            List of tuples (candidate, can_add, violations)
        """
        results = []
        
        for candidate in candidates:
            can_add, violations = self.position.can_add_trade(candidate)
            results.append((candidate, can_add, violations))
        
        return results
    
    def select_optimal_trades(self, 
                            candidates: List[TradeCandidate],
                            max_trades: Optional[int] = None) -> List[TradeCandidate]:
        """
        Select optimal trades from candidates considering portfolio constraints.
        
        Args:
            candidates: List of trade candidates
            max_trades: Maximum trades to select (uses portfolio constraint if None)
            
        Returns:
            List of selected trade candidates
        """
        if max_trades is None:
            max_trades = self.position.constraints.max_trades - len(self.position.active_trades)
        
        # Evaluate all candidates
        evaluations = self.evaluate_trade_candidates(candidates)
        
        # Filter to only valid candidates
        valid_candidates = [(candidate, violations) for candidate, can_add, violations in evaluations if can_add]
        
        if not valid_candidates:
            return []
        
        # Sort by model score (assuming higher is better)
        valid_candidates.sort(key=lambda x: x[0].model_score or 0, reverse=True)
        
        # Select trades while respecting constraints
        selected_trades = []
        temp_position = self._create_temp_position()
        
        for candidate, violations in valid_candidates:
            if len(selected_trades) >= max_trades:
                break
            
            # Check if we can still add this trade
            can_add, _ = temp_position.can_add_trade(candidate)
            if can_add:
                selected_trades.append(candidate)
                # Simulate adding the trade to temp position
                self._simulate_trade_addition(temp_position, candidate)
        
        return selected_trades
    
    def monitor_portfolio_health(self) -> Dict[str, any]:
        """Monitor portfolio health and risk metrics."""
        utilization = self.position.calculate_utilization_metrics()
        
        # Health score calculation (0-100)
        health_score = 100
        
        # Penalize high utilization
        if utilization['trade_utilization'] > 0.9:
            health_score -= 20
        elif utilization['trade_utilization'] > 0.8:
            health_score -= 10
        
        if utilization['capital_utilization'] > 0.9:
            health_score -= 15
        elif utilization['capital_utilization'] > 0.8:
            health_score -= 8
        
        if utilization['delta_utilization'] > 0.9:
            health_score -= 15
        elif utilization['delta_utilization'] > 0.8:
            health_score -= 8
        
        # Check for loss concentration
        total_unrealized = float(self.position.get_total_unrealized_pnl())
        nav = float(self.position.nav)
        
        if total_unrealized < -nav * 0.1:  # More than 10% loss
            health_score -= 25
        elif total_unrealized < -nav * 0.05:  # More than 5% loss
            health_score -= 15
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'utilization_metrics': utilization,
            'risk_warnings': self._generate_risk_warnings(),
            'portfolio_summary': self.position.get_risk_summary()
        }
    
    def _create_temp_position(self) -> Position:
        """Create a temporary position for simulation."""
        # For simplicity, return the current position
        # In a real implementation, this would create a deep copy
        return self.position
    
    def _simulate_trade_addition(self, temp_position: Position, candidate: TradeCandidate) -> None:
        """Simulate adding a trade to temporary position."""
        # This would update the temporary position's state
        # For now, we'll skip the implementation
        pass
    
    def _generate_risk_warnings(self) -> List[str]:
        """Generate risk warnings based on current portfolio state."""
        warnings = []
        utilization = self.position.calculate_utilization_metrics()
        
        if utilization['trade_utilization'] > 0.8:
            warnings.append("High trade count utilization")
        
        if utilization['capital_utilization'] > 0.9:
            warnings.append("High capital utilization")
        
        if utilization['delta_utilization'] > 0.8:
            warnings.append("High delta exposure")
        
        # Check sector concentration
        sector_trades = self.position.get_trades_by_sector()
        max_sector_count = max(sector_trades.values()) if sector_trades else 0
        
        if max_sector_count >= self.position.constraints.max_trades_per_sector:
            warnings.append("Maximum sector concentration reached")
        
        return warnings