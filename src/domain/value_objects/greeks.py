"""
Domain value objects for options Greeks calculations and analysis.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from ...data.models.options import Greeks as BaseGreeks, OptionType


class GreeksRiskLevel(str, Enum):
    """Risk levels for Greeks exposure."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(frozen=True)
class DomainGreeks:
    """
    Enhanced Greeks value object with rich domain behavior.
    
    Immutable value object that extends basic Greeks with
    domain-specific calculations and risk analysis.
    """
    
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    
    # Enhanced metrics
    delta_dollars: Optional[float] = None  # Delta in dollar terms
    gamma_dollars: Optional[float] = None  # Gamma in dollar terms
    theta_dollars: Optional[float] = None  # Theta in dollar terms
    vega_dollars: Optional[float] = None   # Vega in dollar terms
    
    # Risk metrics
    delta_risk_level: Optional[GreeksRiskLevel] = None
    gamma_risk_level: Optional[GreeksRiskLevel] = None
    vega_risk_level: Optional[GreeksRiskLevel] = None
    
    def __post_init__(self):
        """Validate Greeks values upon creation."""
        if self.delta is not None and not (-2.0 <= self.delta <= 2.0):
            raise ValueError(f"Delta {self.delta} out of reasonable range [-2, 2]")
        
        if self.gamma is not None and self.gamma < 0:
            raise ValueError(f"Gamma {self.gamma} cannot be negative")
        
        if self.vega is not None and self.vega < 0:
            raise ValueError(f"Vega {self.vega} cannot be negative")
    
    @classmethod
    def from_base_greeks(cls, 
                        base_greeks: BaseGreeks,
                        underlying_price: Optional[Decimal] = None,
                        position_size: int = 1) -> 'DomainGreeks':
        """
        Create DomainGreeks from base Greeks with enhanced calculations.
        
        Args:
            base_greeks: Basic Greeks from data layer
            underlying_price: Current underlying price for dollar calculations
            position_size: Number of contracts (positive for long, negative for short)
            
        Returns:
            Enhanced DomainGreeks instance
        """
        # Calculate dollar Greeks if underlying price is available
        delta_dollars = None
        gamma_dollars = None
        theta_dollars = None
        vega_dollars = None
        
        if underlying_price:
            multiplier = position_size * 100  # Standard option multiplier
            underlying_float = float(underlying_price)
            
            if base_greeks.delta:
                delta_dollars = base_greeks.delta * underlying_float * multiplier
            
            if base_greeks.gamma:
                gamma_dollars = base_greeks.gamma * underlying_float * multiplier
            
            if base_greeks.theta:
                theta_dollars = base_greeks.theta * multiplier
            
            if base_greeks.vega:
                vega_dollars = base_greeks.vega * multiplier
        
        # Assess risk levels
        delta_risk = cls._assess_delta_risk(base_greeks.delta, position_size)
        gamma_risk = cls._assess_gamma_risk(base_greeks.gamma, position_size)
        vega_risk = cls._assess_vega_risk(base_greeks.vega, position_size)
        
        return cls(
            delta=base_greeks.delta,
            gamma=base_greeks.gamma,
            theta=base_greeks.theta,
            vega=base_greeks.vega,
            rho=base_greeks.rho,
            delta_dollars=delta_dollars,
            gamma_dollars=gamma_dollars,
            theta_dollars=theta_dollars,
            vega_dollars=vega_dollars,
            delta_risk_level=delta_risk,
            gamma_risk_level=gamma_risk,
            vega_risk_level=vega_risk
        )
    
    @staticmethod
    def _assess_delta_risk(delta: Optional[float], position_size: int) -> Optional[GreeksRiskLevel]:
        """Assess delta risk level."""
        if delta is None:
            return None
        
        # Adjust for position size
        effective_delta = abs(delta * position_size)
        
        if effective_delta <= 0.25:
            return GreeksRiskLevel.LOW
        elif effective_delta <= 0.50:
            return GreeksRiskLevel.MODERATE
        elif effective_delta <= 0.75:
            return GreeksRiskLevel.HIGH
        else:
            return GreeksRiskLevel.EXTREME
    
    @staticmethod
    def _assess_gamma_risk(gamma: Optional[float], position_size: int) -> Optional[GreeksRiskLevel]:
        """Assess gamma risk level."""
        if gamma is None:
            return None
        
        # Adjust for position size
        effective_gamma = abs(gamma * position_size)
        
        if effective_gamma <= 0.05:
            return GreeksRiskLevel.LOW
        elif effective_gamma <= 0.15:
            return GreeksRiskLevel.MODERATE
        elif effective_gamma <= 0.30:
            return GreeksRiskLevel.HIGH
        else:
            return GreeksRiskLevel.EXTREME
    
    @staticmethod
    def _assess_vega_risk(vega: Optional[float], position_size: int) -> Optional[GreeksRiskLevel]:
        """Assess vega risk level."""
        if vega is None:
            return None
        
        # Adjust for position size
        effective_vega = abs(vega * position_size)
        
        if effective_vega <= 10:
            return GreeksRiskLevel.LOW
        elif effective_vega <= 25:
            return GreeksRiskLevel.MODERATE
        elif effective_vega <= 50:
            return GreeksRiskLevel.HIGH
        else:
            return GreeksRiskLevel.EXTREME
    
    def add(self, other: 'DomainGreeks') -> 'DomainGreeks':
        """Add two Greeks objects together."""
        return DomainGreeks(
            delta=self._safe_add(self.delta, other.delta),
            gamma=self._safe_add(self.gamma, other.gamma),
            theta=self._safe_add(self.theta, other.theta),
            vega=self._safe_add(self.vega, other.vega),
            rho=self._safe_add(self.rho, other.rho),
            delta_dollars=self._safe_add(self.delta_dollars, other.delta_dollars),
            gamma_dollars=self._safe_add(self.gamma_dollars, other.gamma_dollars),
            theta_dollars=self._safe_add(self.theta_dollars, other.theta_dollars),
            vega_dollars=self._safe_add(self.vega_dollars, other.vega_dollars)
        )
    
    def scale(self, factor: float) -> 'DomainGreeks':
        """Scale Greeks by a factor."""
        return DomainGreeks(
            delta=self._safe_multiply(self.delta, factor),
            gamma=self._safe_multiply(self.gamma, factor),
            theta=self._safe_multiply(self.theta, factor),
            vega=self._safe_multiply(self.vega, factor),
            rho=self._safe_multiply(self.rho, factor),
            delta_dollars=self._safe_multiply(self.delta_dollars, factor),
            gamma_dollars=self._safe_multiply(self.gamma_dollars, factor),
            theta_dollars=self._safe_multiply(self.theta_dollars, factor),
            vega_dollars=self._safe_multiply(self.vega_dollars, factor)
        )
    
    def get_directional_exposure(self) -> Optional[float]:
        """Get overall directional exposure magnitude."""
        if self.delta is None:
            return None
        return abs(self.delta)
    
    def get_convexity_exposure(self) -> Optional[float]:
        """Get convexity exposure from gamma."""
        return self.gamma
    
    def get_time_decay_income(self) -> Optional[float]:
        """Get daily time decay income (positive means income)."""
        if self.theta is None:
            return None
        return -self.theta  # Theta is typically negative, so negate for income
    
    def get_volatility_exposure(self) -> Optional[float]:
        """Get volatility exposure from vega."""
        return self.vega
    
    def is_delta_neutral(self, tolerance: float = 0.05) -> bool:
        """Check if position is delta neutral within tolerance."""
        if self.delta is None:
            return False
        return abs(self.delta) <= tolerance
    
    def is_gamma_neutral(self, tolerance: float = 0.02) -> bool:
        """Check if position is gamma neutral within tolerance."""
        if self.gamma is None:
            return False
        return abs(self.gamma) <= tolerance
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'delta_dollars': self.delta_dollars,
            'theta_dollars': self.theta_dollars,
            'vega_dollars': self.vega_dollars,
            'directional_exposure': self.get_directional_exposure(),
            'convexity_exposure': self.get_convexity_exposure(),
            'time_decay_income': self.get_time_decay_income(),
            'volatility_exposure': self.get_volatility_exposure(),
            'delta_risk_level': self.delta_risk_level.value if self.delta_risk_level else None,
            'gamma_risk_level': self.gamma_risk_level.value if self.gamma_risk_level else None,
            'vega_risk_level': self.vega_risk_level.value if self.vega_risk_level else None,
            'is_delta_neutral': self.is_delta_neutral(),
            'is_gamma_neutral': self.is_gamma_neutral()
        }
    
    @staticmethod
    def _safe_add(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Safely add two optional float values."""
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return a + b
    
    @staticmethod
    def _safe_multiply(value: Optional[float], factor: float) -> Optional[float]:
        """Safely multiply optional float by factor."""
        if value is None:
            return None
        return value * factor


@dataclass(frozen=True)
class GreeksSnapshot:
    """
    Snapshot of Greeks at a specific point in time.
    
    Immutable value object representing Greeks state with timestamp
    and market conditions for historical analysis.
    """
    
    greeks: DomainGreeks
    timestamp: datetime
    underlying_price: Optional[Decimal] = None
    underlying_symbol: Optional[str] = None
    market_conditions: Optional[Dict[str, Any]] = None
    
    def age_in_minutes(self) -> float:
        """Get age of snapshot in minutes."""
        age = datetime.utcnow() - self.timestamp
        return age.total_seconds() / 60
    
    def is_stale(self, max_age_minutes: float = 15.0) -> bool:
        """Check if snapshot is stale."""
        return self.age_in_minutes() > max_age_minutes
    
    def compare_to(self, other: 'GreeksSnapshot') -> Dict[str, Optional[float]]:
        """Compare Greeks changes between snapshots."""
        return {
            'delta_change': self._safe_subtract(self.greeks.delta, other.greeks.delta),
            'gamma_change': self._safe_subtract(self.greeks.gamma, other.greeks.gamma),
            'theta_change': self._safe_subtract(self.greeks.theta, other.greeks.theta),
            'vega_change': self._safe_subtract(self.greeks.vega, other.greeks.vega),
            'rho_change': self._safe_subtract(self.greeks.rho, other.greeks.rho),
            'time_elapsed_minutes': (self.timestamp - other.timestamp).total_seconds() / 60
        }
    
    @staticmethod
    def _safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Safely subtract two optional float values."""
        if a is None or b is None:
            return None
        return a - b


@dataclass(frozen=True)
class GreeksAnalysis:
    """
    Comprehensive Greeks analysis value object.
    
    Provides detailed analysis of Greeks behavior, risk assessment,
    and hedging recommendations.
    """
    
    current_greeks: DomainGreeks
    target_greeks: Optional[DomainGreeks] = None
    risk_limits: Optional[Dict[str, float]] = None
    hedging_requirements: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate derived analysis fields."""
        object.__setattr__(self, 'analysis_timestamp', datetime.utcnow())
        object.__setattr__(self, '_risk_violations', self._calculate_risk_violations())
        object.__setattr__(self, '_hedging_needs', self._calculate_hedging_needs())
    
    def _calculate_risk_violations(self) -> List[Dict[str, Any]]:
        """Calculate any risk limit violations."""
        violations = []
        
        if not self.risk_limits:
            return violations
        
        # Check delta limits
        delta_limit = self.risk_limits.get('max_delta')
        if delta_limit and self.current_greeks.delta:
            if abs(self.current_greeks.delta) > delta_limit:
                violations.append({
                    'greek': 'delta',
                    'current': self.current_greeks.delta,
                    'limit': delta_limit,
                    'violation_amount': abs(self.current_greeks.delta) - delta_limit,
                    'severity': 'high' if abs(self.current_greeks.delta) > delta_limit * 1.2 else 'medium'
                })
        
        # Check gamma limits
        gamma_limit = self.risk_limits.get('max_gamma')
        if gamma_limit and self.current_greeks.gamma:
            if self.current_greeks.gamma > gamma_limit:
                violations.append({
                    'greek': 'gamma',
                    'current': self.current_greeks.gamma,
                    'limit': gamma_limit,
                    'violation_amount': self.current_greeks.gamma - gamma_limit,
                    'severity': 'high' if self.current_greeks.gamma > gamma_limit * 1.5 else 'medium'
                })
        
        # Check vega limits
        vega_limit = self.risk_limits.get('max_vega')
        if vega_limit and self.current_greeks.vega:
            if abs(self.current_greeks.vega) > vega_limit:
                violations.append({
                    'greek': 'vega',
                    'current': self.current_greeks.vega,
                    'limit': vega_limit,
                    'violation_amount': abs(self.current_greeks.vega) - vega_limit,
                    'severity': 'medium'
                })
        
        return violations
    
    def _calculate_hedging_needs(self) -> Dict[str, Any]:
        """Calculate hedging requirements."""
        needs = {
            'delta_hedge_shares': None,
            'gamma_hedge_options': None,
            'vega_hedge_options': None,
            'hedging_priority': []
        }
        
        # Delta hedging
        if self.current_greeks.delta:
            # Number of shares needed to hedge delta
            needs['delta_hedge_shares'] = -int(self.current_greeks.delta * 100)
            
            if abs(self.current_greeks.delta) > 0.25:
                needs['hedging_priority'].append('delta')
        
        # Gamma hedging assessment
        if self.current_greeks.gamma and self.current_greeks.gamma > 0.15:
            needs['hedging_priority'].append('gamma')
        
        # Vega hedging assessment
        if self.current_greeks.vega and abs(self.current_greeks.vega) > 25:
            needs['hedging_priority'].append('vega')
        
        return needs
    
    def has_risk_violations(self) -> bool:
        """Check if there are any risk violations."""
        return len(self._risk_violations) > 0
    
    def get_risk_violations(self) -> List[Dict[str, Any]]:
        """Get list of risk violations."""
        return self._risk_violations.copy()
    
    def needs_hedging(self) -> bool:
        """Check if position needs hedging."""
        return len(self._hedging_needs['hedging_priority']) > 0
    
    def get_hedging_recommendations(self) -> Dict[str, Any]:
        """Get hedging recommendations."""
        return self._hedging_needs.copy()
    
    def calculate_hedge_effectiveness(self, hedge_greeks: DomainGreeks) -> Dict[str, float]:
        """Calculate effectiveness of proposed hedge."""
        hedged_greeks = self.current_greeks.add(hedge_greeks)
        
        # Calculate reduction in exposure
        effectiveness = {}
        
        if self.current_greeks.delta and hedged_greeks.delta:
            delta_reduction = (abs(self.current_greeks.delta) - abs(hedged_greeks.delta)) / abs(self.current_greeks.delta)
            effectiveness['delta_hedge_effectiveness'] = max(0, delta_reduction)
        
        if self.current_greeks.gamma and hedged_greeks.gamma:
            gamma_reduction = (self.current_greeks.gamma - hedged_greeks.gamma) / self.current_greeks.gamma
            effectiveness['gamma_hedge_effectiveness'] = max(0, gamma_reduction)
        
        if self.current_greeks.vega and hedged_greeks.vega:
            vega_reduction = (abs(self.current_greeks.vega) - abs(hedged_greeks.vega)) / abs(self.current_greeks.vega)
            effectiveness['vega_hedge_effectiveness'] = max(0, vega_reduction)
        
        return effectiveness
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive Greeks analysis."""
        return {
            'current_greeks': self.current_greeks.get_risk_summary(),
            'target_greeks': self.target_greeks.get_risk_summary() if self.target_greeks else None,
            'risk_violations': self.get_risk_violations(),
            'has_violations': self.has_risk_violations(),
            'needs_hedging': self.needs_hedging(),
            'hedging_recommendations': self.get_hedging_recommendations(),
            'analysis_timestamp': self.analysis_timestamp,
            'overall_risk_level': self._assess_overall_risk_level()
        }
    
    def _assess_overall_risk_level(self) -> GreeksRiskLevel:
        """Assess overall risk level across all Greeks."""
        risk_levels = []
        
        if self.current_greeks.delta_risk_level:
            risk_levels.append(self.current_greeks.delta_risk_level)
        if self.current_greeks.gamma_risk_level:
            risk_levels.append(self.current_greeks.gamma_risk_level)
        if self.current_greeks.vega_risk_level:
            risk_levels.append(self.current_greeks.vega_risk_level)
        
        if not risk_levels:
            return GreeksRiskLevel.LOW
        
        # Return highest risk level
        if GreeksRiskLevel.EXTREME in risk_levels:
            return GreeksRiskLevel.EXTREME
        elif GreeksRiskLevel.HIGH in risk_levels:
            return GreeksRiskLevel.HIGH
        elif GreeksRiskLevel.MODERATE in risk_levels:
            return GreeksRiskLevel.MODERATE
        else:
            return GreeksRiskLevel.LOW


class GreeksCalculator:
    """
    Utility class for advanced Greeks calculations and transformations.
    
    Provides static methods for complex Greeks operations that don't
    belong in individual value objects.
    """
    
    @staticmethod
    def aggregate_portfolio_greeks(greeks_list: List[DomainGreeks]) -> DomainGreeks:
        """Aggregate Greeks from multiple positions."""
        if not greeks_list:
            return DomainGreeks()
        
        result = greeks_list[0]
        for greeks in greeks_list[1:]:
            result = result.add(greeks)
        
        return result
    
    @staticmethod
    def calculate_correlation_adjusted_greeks(
        greeks_list: List[DomainGreeks],
        correlation_matrix: Optional[List[List[float]]] = None
    ) -> DomainGreeks:
        """
        Calculate correlation-adjusted portfolio Greeks.
        
        Args:
            greeks_list: List of individual position Greeks
            correlation_matrix: Correlation matrix between positions
            
        Returns:
            Correlation-adjusted portfolio Greeks
        """
        if not correlation_matrix:
            # If no correlation matrix, assume zero correlation
            return GreeksCalculator.aggregate_portfolio_greeks(greeks_list)
        
        # For now, implement simple aggregation
        # Real implementation would apply correlation adjustments
        return GreeksCalculator.aggregate_portfolio_greeks(greeks_list)
    
    @staticmethod
    def estimate_greeks_decay(
        current_greeks: DomainGreeks,
        days_forward: int,
        underlying_price: Decimal,
        volatility: float
    ) -> DomainGreeks:
        """
        Estimate Greeks values after time decay.
        
        Args:
            current_greeks: Current Greeks values
            days_forward: Number of days to project forward
            underlying_price: Current underlying price
            volatility: Current volatility
            
        Returns:
            Estimated Greeks after time decay
        """
        # Simplified time decay estimation
        # Real implementation would use proper options pricing models
        
        time_factor = max(0, 1 - (days_forward / 365))  # Rough time decay factor
        
        return DomainGreeks(
            delta=current_greeks.delta,  # Delta relatively stable
            gamma=current_greeks.gamma * time_factor if current_greeks.gamma else None,
            theta=current_greeks.theta * time_factor if current_greeks.theta else None,
            vega=current_greeks.vega * time_factor if current_greeks.vega else None,
            rho=current_greeks.rho  # Rho relatively stable
        )
    
    @staticmethod
    def calculate_greeks_sensitivity(
        base_greeks: DomainGreeks,
        price_change_pct: float,
        vol_change_pct: float
    ) -> Dict[str, DomainGreeks]:
        """
        Calculate Greeks sensitivity to market moves.
        
        Args:
            base_greeks: Base Greeks values
            price_change_pct: Percentage change in underlying price
            vol_change_pct: Percentage change in volatility
            
        Returns:
            Dictionary of scenario Greeks
        """
        scenarios = {}
        
        # Price up scenario
        price_up_factor = 1 + (price_change_pct / 100)
        scenarios['price_up'] = DomainGreeks(
            delta=base_greeks.delta,
            gamma=base_greeks.gamma * 0.9 if base_greeks.gamma else None,  # Gamma decreases as we move ITM/OTM
            theta=base_greeks.theta,
            vega=base_greeks.vega * 0.95 if base_greeks.vega else None,  # Vega decreases slightly
            rho=base_greeks.rho
        )
        
        # Price down scenario
        scenarios['price_down'] = DomainGreeks(
            delta=base_greeks.delta,
            gamma=base_greeks.gamma * 0.9 if base_greeks.gamma else None,
            theta=base_greeks.theta,
            vega=base_greeks.vega * 0.95 if base_greeks.vega else None,
            rho=base_greeks.rho
        )
        
        # Volatility up scenario
        vol_up_factor = 1 + (vol_change_pct / 100)
        scenarios['vol_up'] = DomainGreeks(
            delta=base_greeks.delta,
            gamma=base_greeks.gamma * vol_up_factor if base_greeks.gamma else None,
            theta=base_greeks.theta,
            vega=base_greeks.vega * vol_up_factor if base_greeks.vega else None,
            rho=base_greeks.rho
        )
        
        # Volatility down scenario
        vol_down_factor = 1 + (vol_change_pct / 100)
        scenarios['vol_down'] = DomainGreeks(
            delta=base_greeks.delta,
            gamma=base_greeks.gamma * vol_down_factor if base_greeks.gamma else None,
            theta=base_greeks.theta,
            vega=base_greeks.vega * vol_down_factor if base_greeks.vega else None,
            rho=base_greeks.rho
        )
        
        return scenarios