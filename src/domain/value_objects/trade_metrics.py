"""
Trade metrics value object for the Options Trading Engine.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

from ...infrastructure.error_handling import ValidationError


@dataclass(frozen=True)
class TradeMetrics:
    """
    Trade metrics value object representing key trade performance metrics.
    
    This is an immutable value object that encapsulates all the key
    metrics for evaluating a trade opportunity.
    """
    
    # Profitability metrics
    max_profit: float
    max_loss: float
    probability_of_profit: float
    expected_value: float
    
    # Risk metrics
    profit_loss_ratio: float
    breakeven_points: list
    capital_required: float
    
    # Efficiency metrics
    return_on_capital: float
    annualized_return: float
    days_to_expiration: int
    
    # Greeks exposure
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    
    # Market metrics
    iv_rank: Optional[float] = None
    liquidity_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate trade metrics."""
        self._validate_metrics()
    
    def _validate_metrics(self):
        """Validate trade metrics are within reasonable ranges."""
        # Probability of profit validation
        if not 0.0 <= self.probability_of_profit <= 1.0:
            raise ValidationError(f"Probability of profit must be between 0.0 and 1.0, got {self.probability_of_profit}")
        
        # Capital required validation
        if self.capital_required < 0:
            raise ValidationError(f"Capital required must be non-negative, got {self.capital_required}")
        
        # Days to expiration validation
        if self.days_to_expiration <= 0:
            raise ValidationError(f"Days to expiration must be positive, got {self.days_to_expiration}")
        
        # Profit/Loss ratio validation
        if self.profit_loss_ratio < 0:
            raise ValidationError(f"Profit/Loss ratio must be non-negative, got {self.profit_loss_ratio}")
        
        # IV rank validation
        if self.iv_rank is not None and not 0.0 <= self.iv_rank <= 100.0:
            raise ValidationError(f"IV rank must be between 0.0 and 100.0, got {self.iv_rank}")
        
        # Liquidity score validation
        if self.liquidity_score is not None and not 0.0 <= self.liquidity_score <= 1.0:
            raise ValidationError(f"Liquidity score must be between 0.0 and 1.0, got {self.liquidity_score}")
    
    @property
    def max_profit_percentage(self) -> float:
        """Get max profit as percentage of capital required."""
        if self.capital_required > 0:
            return (self.max_profit / self.capital_required) * 100
        return 0.0
    
    @property
    def max_loss_percentage(self) -> float:
        """Get max loss as percentage of capital required."""
        if self.capital_required > 0:
            return (abs(self.max_loss) / self.capital_required) * 100
        return 0.0
    
    @property
    def win_rate_threshold(self) -> float:
        """Calculate minimum win rate needed to be profitable."""
        if self.max_profit > 0 and self.max_loss < 0:
            return abs(self.max_loss) / (self.max_profit + abs(self.max_loss))
        return 0.0
    
    @property
    def edge(self) -> float:
        """Calculate edge (expected value relative to capital at risk)."""
        if self.capital_required > 0:
            return self.expected_value / self.capital_required
        return 0.0
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if self.max_profit > 0:
            return abs(self.max_loss) / self.max_profit
        return float('inf')
    
    @property
    def kelly_fraction(self) -> float:
        """Calculate Kelly fraction for position sizing."""
        if self.max_loss < 0 and self.max_profit > 0:
            p = self.probability_of_profit
            b = self.max_profit / abs(self.max_loss)  # Odds received
            kelly = (p * b - (1 - p)) / b
            return max(0, min(kelly, 0.25))  # Cap at 25% for safety
        return 0.0
    
    @property
    def annualized_profit_potential(self) -> float:
        """Calculate annualized profit potential."""
        if self.days_to_expiration > 0:
            return (self.max_profit_percentage / self.days_to_expiration) * 365
        return 0.0
    
    @property
    def theta_efficiency(self) -> float:
        """Calculate theta efficiency (theta per day per dollar at risk)."""
        if self.capital_required > 0 and self.days_to_expiration > 0:
            return (self.net_theta / self.capital_required) * 365
        return 0.0
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment."""
        return {
            'risk_level': self._calculate_risk_level(),
            'risk_factors': self._identify_risk_factors(),
            'position_sizing_recommendation': self._get_position_sizing_recommendation(),
            'exit_criteria': self._get_exit_criteria()
        }
    
    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level."""
        risk_score = 0
        
        # Probability of profit factor
        if self.probability_of_profit < 0.5:
            risk_score += 2
        elif self.probability_of_profit < 0.7:
            risk_score += 1
        
        # Risk/reward ratio factor
        if self.risk_reward_ratio > 2.0:
            risk_score += 2
        elif self.risk_reward_ratio > 1.0:
            risk_score += 1
        
        # Days to expiration factor
        if self.days_to_expiration < 15:
            risk_score += 2
        elif self.days_to_expiration < 30:
            risk_score += 1
        
        # Greeks exposure factor
        if abs(self.net_delta) > 0.5:
            risk_score += 1
        if self.net_gamma > 0.1:
            risk_score += 1
        if self.net_vega > 0.5:
            risk_score += 1
        
        # IV rank factor
        if self.iv_rank is not None:
            if self.iv_rank < 30 and self.net_vega > 0:
                risk_score += 1
            elif self.iv_rank > 70 and self.net_vega < 0:
                risk_score += 1
        
        # Liquidity factor
        if self.liquidity_score is not None and self.liquidity_score < 0.5:
            risk_score += 1
        
        if risk_score <= 2:
            return "LOW"
        elif risk_score <= 4:
            return "MEDIUM"
        elif risk_score <= 6:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _identify_risk_factors(self) -> list:
        """Identify specific risk factors."""
        factors = []
        
        if self.probability_of_profit < 0.5:
            factors.append("Low probability of profit")
        
        if self.risk_reward_ratio > 2.0:
            factors.append("High risk/reward ratio")
        
        if self.days_to_expiration < 15:
            factors.append("Short time to expiration")
        
        if abs(self.net_delta) > 0.5:
            factors.append("High delta exposure")
        
        if self.net_gamma > 0.1:
            factors.append("High gamma exposure")
        
        if self.net_vega > 0.5:
            factors.append("High vega exposure")
        
        if self.iv_rank is not None and self.iv_rank < 30 and self.net_vega > 0:
            factors.append("Long volatility in low IV environment")
        
        if self.iv_rank is not None and self.iv_rank > 70 and self.net_vega < 0:
            factors.append("Short volatility in high IV environment")
        
        if self.liquidity_score is not None and self.liquidity_score < 0.5:
            factors.append("Poor liquidity")
        
        return factors
    
    def _get_position_sizing_recommendation(self) -> Dict[str, Any]:
        """Get position sizing recommendation."""
        kelly = self.kelly_fraction
        risk_level = self._calculate_risk_level()
        
        # Adjust Kelly based on risk level
        if risk_level == "LOW":
            recommended_size = min(kelly, 0.10)  # Max 10% for low risk
        elif risk_level == "MEDIUM":
            recommended_size = min(kelly, 0.05)  # Max 5% for medium risk
        elif risk_level == "HIGH":
            recommended_size = min(kelly, 0.02)  # Max 2% for high risk
        else:  # VERY_HIGH
            recommended_size = min(kelly, 0.01)  # Max 1% for very high risk
        
        return {
            'kelly_fraction': kelly,
            'recommended_size': recommended_size,
            'max_position_size': recommended_size * 2,  # Conservative max
            'reasoning': f"Based on {risk_level} risk level and Kelly criterion"
        }
    
    def _get_exit_criteria(self) -> Dict[str, Any]:
        """Get exit criteria recommendations."""
        return {
            'profit_target': min(self.max_profit * 0.5, self.max_profit * 0.8),  # 50-80% of max profit
            'stop_loss': self.max_loss * 0.5,  # 50% of max loss
            'time_exit': max(7, self.days_to_expiration // 3),  # Exit with 1/3 time remaining
            'delta_threshold': 0.15,  # Close if delta gets too high
            'iv_change_threshold': 0.1  # Close if IV changes by 10%
        }
    
    def compare_to(self, other: 'TradeMetrics') -> Dict[str, Any]:
        """Compare this trade to another trade."""
        comparison = {}
        
        # Compare key metrics
        comparison['probability_of_profit'] = {
            'this': self.probability_of_profit,
            'other': other.probability_of_profit,
            'better': self.probability_of_profit > other.probability_of_profit
        }
        
        comparison['expected_value'] = {
            'this': self.expected_value,
            'other': other.expected_value,
            'better': self.expected_value > other.expected_value
        }
        
        comparison['return_on_capital'] = {
            'this': self.return_on_capital,
            'other': other.return_on_capital,
            'better': self.return_on_capital > other.return_on_capital
        }
        
        comparison['risk_reward_ratio'] = {
            'this': self.risk_reward_ratio,
            'other': other.risk_reward_ratio,
            'better': self.risk_reward_ratio < other.risk_reward_ratio  # Lower is better
        }
        
        comparison['annualized_return'] = {
            'this': self.annualized_return,
            'other': other.annualized_return,
            'better': self.annualized_return > other.annualized_return
        }
        
        # Overall recommendation
        better_count = sum(1 for metric in comparison.values() if metric.get('better', False))
        total_count = len(comparison)
        
        comparison['overall_better'] = better_count > total_count / 2
        comparison['score'] = better_count / total_count
        
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'probability_of_profit': self.probability_of_profit,
            'expected_value': self.expected_value,
            'profit_loss_ratio': self.profit_loss_ratio,
            'breakeven_points': self.breakeven_points,
            'capital_required': self.capital_required,
            'return_on_capital': self.return_on_capital,
            'annualized_return': self.annualized_return,
            'days_to_expiration': self.days_to_expiration,
            'net_delta': self.net_delta,
            'net_gamma': self.net_gamma,
            'net_theta': self.net_theta,
            'net_vega': self.net_vega,
            'net_rho': self.net_rho,
            'iv_rank': self.iv_rank,
            'liquidity_score': self.liquidity_score,
            'max_profit_percentage': self.max_profit_percentage,
            'max_loss_percentage': self.max_loss_percentage,
            'win_rate_threshold': self.win_rate_threshold,
            'edge': self.edge,
            'risk_reward_ratio': self.risk_reward_ratio,
            'kelly_fraction': self.kelly_fraction,
            'annualized_profit_potential': self.annualized_profit_potential,
            'theta_efficiency': self.theta_efficiency,
            'risk_assessment': self.get_risk_assessment()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeMetrics':
        """Create from dictionary representation."""
        return cls(
            max_profit=data['max_profit'],
            max_loss=data['max_loss'],
            probability_of_profit=data['probability_of_profit'],
            expected_value=data['expected_value'],
            profit_loss_ratio=data['profit_loss_ratio'],
            breakeven_points=data['breakeven_points'],
            capital_required=data['capital_required'],
            return_on_capital=data['return_on_capital'],
            annualized_return=data['annualized_return'],
            days_to_expiration=data['days_to_expiration'],
            net_delta=data['net_delta'],
            net_gamma=data['net_gamma'],
            net_theta=data['net_theta'],
            net_vega=data['net_vega'],
            net_rho=data['net_rho'],
            iv_rank=data.get('iv_rank'),
            liquidity_score=data.get('liquidity_score')
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"TradeMetrics(POP={self.probability_of_profit:.2%}, EV={self.expected_value:.2f}, ROC={self.return_on_capital:.2%})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"TradeMetrics(profit={self.max_profit}, loss={self.max_loss}, pop={self.probability_of_profit}, ev={self.expected_value})"