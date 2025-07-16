"""
Greeks value object for the Options Trading Engine.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from ...infrastructure.error_handling import ValidationError


@dataclass(frozen=True)
class Greeks:
    """
    Greeks value object representing option sensitivities.
    
    This is an immutable value object that encapsulates all the Greek
    sensitivities for an option contract.
    """
    
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def __post_init__(self):
        """Validate Greeks values."""
        self._validate_greeks()
    
    def _validate_greeks(self):
        """Validate Greeks values are within reasonable ranges."""
        # Delta validation
        if not -1.0 <= self.delta <= 1.0:
            raise ValidationError(f"Delta must be between -1.0 and 1.0, got {self.delta}")
        
        # Gamma validation (always positive)
        if self.gamma < 0:
            raise ValidationError(f"Gamma must be non-negative, got {self.gamma}")
        
        # Gamma should not be extremely large
        if self.gamma > 1.0:
            raise ValidationError(f"Gamma seems unusually large: {self.gamma}")
        
        # Theta validation (usually negative for long options)
        if abs(self.theta) > 1.0:
            raise ValidationError(f"Theta seems unusually large: {self.theta}")
        
        # Vega validation (always positive)
        if self.vega < 0:
            raise ValidationError(f"Vega must be non-negative, got {self.vega}")
        
        # Vega should not be extremely large
        if self.vega > 10.0:
            raise ValidationError(f"Vega seems unusually large: {self.vega}")
        
        # Rho validation
        if abs(self.rho) > 10.0:
            raise ValidationError(f"Rho seems unusually large: {self.rho}")
    
    @property
    def delta_percentage(self) -> float:
        """Get delta as percentage."""
        return self.delta * 100
    
    @property
    def gamma_percentage(self) -> float:
        """Get gamma as percentage."""
        return self.gamma * 100
    
    @property
    def theta_daily(self) -> float:
        """Get theta as daily decay (theta is typically per day)."""
        return self.theta
    
    @property
    def vega_percentage(self) -> float:
        """Get vega as percentage (per 1% IV change)."""
        return self.vega
    
    @property
    def rho_percentage(self) -> float:
        """Get rho as percentage (per 1% interest rate change)."""
        return self.rho
    
    def scale(self, multiplier: float) -> 'Greeks':
        """Scale all Greeks by a multiplier (e.g., for position sizing)."""
        return Greeks(
            delta=self.delta * multiplier,
            gamma=self.gamma * multiplier,
            theta=self.theta * multiplier,
            vega=self.vega * multiplier,
            rho=self.rho * multiplier
        )
    
    def add(self, other: 'Greeks') -> 'Greeks':
        """Add another Greeks object (for portfolio aggregation)."""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho
        )
    
    def subtract(self, other: 'Greeks') -> 'Greeks':
        """Subtract another Greeks object."""
        return Greeks(
            delta=self.delta - other.delta,
            gamma=self.gamma - other.gamma,
            theta=self.theta - other.theta,
            vega=self.vega - other.vega,
            rho=self.rho - other.rho
        )
    
    def is_delta_neutral(self, threshold: float = 0.05) -> bool:
        """Check if position is delta neutral within threshold."""
        return abs(self.delta) <= threshold
    
    def is_gamma_heavy(self, threshold: float = 0.1) -> bool:
        """Check if position has high gamma exposure."""
        return self.gamma >= threshold
    
    def is_theta_positive(self) -> bool:
        """Check if position has positive theta (time decay working for us)."""
        return self.theta > 0
    
    def is_vega_positive(self) -> bool:
        """Check if position benefits from volatility increase."""
        return self.vega > 0
    
    def get_portfolio_impact(self, underlying_move: float = 0.01, 
                           vol_change: float = 0.01, 
                           time_decay: float = 1.0,
                           rate_change: float = 0.01) -> float:
        """
        Calculate approximate portfolio impact from various factors.
        
        Args:
            underlying_move: Underlying price change (as decimal, e.g., 0.01 for 1%)
            vol_change: Volatility change (as decimal, e.g., 0.01 for 1%)
            time_decay: Time decay in days (default 1 day)
            rate_change: Interest rate change (as decimal, e.g., 0.01 for 1%)
            
        Returns:
            Approximate portfolio impact
        """
        delta_impact = self.delta * underlying_move * 100  # Delta impact
        gamma_impact = 0.5 * self.gamma * (underlying_move * 100) ** 2  # Gamma impact
        theta_impact = self.theta * time_decay  # Theta impact
        vega_impact = self.vega * vol_change * 100  # Vega impact
        rho_impact = self.rho * rate_change * 100  # Rho impact
        
        return delta_impact + gamma_impact + theta_impact + vega_impact + rho_impact
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get risk metrics based on Greeks."""
        return {
            'delta_risk': abs(self.delta),
            'gamma_risk': self.gamma,
            'theta_decay': abs(self.theta),
            'vega_risk': self.vega,
            'rho_risk': abs(self.rho),
            'total_risk_score': self._calculate_total_risk_score()
        }
    
    def _calculate_total_risk_score(self) -> float:
        """Calculate total risk score based on Greeks."""
        # Weighted risk score
        delta_weight = 0.3
        gamma_weight = 0.25
        theta_weight = 0.2
        vega_weight = 0.15
        rho_weight = 0.1
        
        risk_score = (
            abs(self.delta) * delta_weight +
            self.gamma * gamma_weight +
            abs(self.theta) * theta_weight +
            self.vega * vega_weight +
            abs(self.rho) * rho_weight
        )
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'delta_percentage': self.delta_percentage,
            'gamma_percentage': self.gamma_percentage,
            'theta_daily': self.theta_daily,
            'vega_percentage': self.vega_percentage,
            'rho_percentage': self.rho_percentage,
            'risk_metrics': self.get_risk_metrics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Greeks':
        """Create from dictionary representation."""
        return cls(
            delta=data['delta'],
            gamma=data['gamma'],
            theta=data['theta'],
            vega=data['vega'],
            rho=data['rho']
        )
    
    @classmethod
    def zero(cls) -> 'Greeks':
        """Create zero Greeks object."""
        return cls(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0
        )
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        """Add operator overload."""
        return self.add(other)
    
    def __sub__(self, other: 'Greeks') -> 'Greeks':
        """Subtract operator overload."""
        return self.subtract(other)
    
    def __mul__(self, multiplier: float) -> 'Greeks':
        """Multiply operator overload."""
        return self.scale(multiplier)
    
    def __rmul__(self, multiplier: float) -> 'Greeks':
        """Reverse multiply operator overload."""
        return self.scale(multiplier)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Greeks(Δ={self.delta:.3f}, Γ={self.gamma:.3f}, Θ={self.theta:.3f}, ν={self.vega:.3f}, ρ={self.rho:.3f})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Greeks(delta={self.delta}, gamma={self.gamma}, theta={self.theta}, vega={self.vega}, rho={self.rho})"