"""
Portfolio Greeks management system for risk control and hedging.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
import logging

from ...data.models.options import Greeks, OptionQuote, OptionType
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, PortfolioGreeks, 
    TradeFilterCriteria
)
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, ConstraintViolationError
)


class GreeksLimitType(Enum):
    """Types of Greeks limits."""
    ABSOLUTE = "absolute"
    PERCENTAGE_OF_NAV = "percentage_of_nav"
    NORMALIZED_PER_100K = "normalized_per_100k"


class RiskLevel(Enum):
    """Portfolio risk levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class GreeksLimits:
    """Portfolio Greeks limits configuration."""
    # Delta limits (directional exposure)
    max_net_delta: Optional[float] = None
    max_positive_delta: Optional[float] = None
    max_negative_delta: Optional[float] = None
    
    # Gamma limits (convexity risk)
    max_net_gamma: Optional[float] = None
    max_positive_gamma: Optional[float] = None
    
    # Theta limits (time decay)
    min_net_theta: Optional[float] = None  # Minimum (most negative)
    max_net_theta: Optional[float] = None  # Maximum (least negative)
    
    # Vega limits (volatility risk)
    max_net_vega: Optional[float] = None
    min_net_vega: Optional[float] = None
    
    # Rho limits (interest rate risk)
    max_net_rho: Optional[float] = None
    min_net_rho: Optional[float] = None
    
    # Limit type
    limit_type: GreeksLimitType = GreeksLimitType.NORMALIZED_PER_100K


@dataclass
class GreeksAnalysis:
    """Portfolio Greeks analysis result."""
    current_greeks: PortfolioGreeks
    limits: GreeksLimits
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    utilization: Dict[str, float] = field(default_factory=dict)  # Percentage of limit used
    risk_score: float = 0.0  # 0-100
    hedge_recommendations: List[str] = field(default_factory=list)


@dataclass
class GreeksImpact:
    """Impact of adding a trade to portfolio Greeks."""
    trade_candidate: TradeCandidate
    current_greeks: PortfolioGreeks
    projected_greeks: PortfolioGreeks
    delta_change: float
    gamma_change: float
    theta_change: float
    vega_change: float
    rho_change: float
    within_limits: bool
    violations: List[str] = field(default_factory=list)


class PortfolioGreeksManager:
    """
    Portfolio-level Greeks management and risk control system.
    
    Features:
    - Individual Trade Greeks Analysis
    - Delta exposure calculation with hedge ratio implications
    - Gamma risk assessment for directional exposure changes
    - Theta decay analysis with time-to-expiration considerations
    - Vega exposure evaluation for volatility sensitivity
    - Rho interest rate sensitivity for longer-dated strategies
    - Portfolio-Level Greeks Aggregation
    - Net delta calculation with position weighting
    - Aggregate gamma exposure with convexity risk assessment
    - Combined theta decay with income generation analysis
    - Total vega exposure with volatility clustering considerations
    - Portfolio correlation effects on individual position Greeks
    """
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_level = risk_level
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Set default limits based on risk level
        self.default_limits = self._get_default_limits(risk_level)
        
        # Greeks correlation matrix (simplified)
        self.greeks_correlations = {
            ("delta", "gamma"): 0.3,
            ("delta", "theta"): -0.2,
            ("delta", "vega"): 0.1,
            ("gamma", "vega"): 0.4,
            ("theta", "vega"): -0.3
        }
    
    @handle_errors(operation_name="analyze_portfolio_greeks")
    def analyze_portfolio_greeks(
        self,
        current_trades: List[TradeCandidate],
        nav: Decimal,
        custom_limits: Optional[GreeksLimits] = None
    ) -> GreeksAnalysis:
        """
        Analyze current portfolio Greeks and identify risks.
        
        Args:
            current_trades: Current portfolio trades
            nav: Portfolio net asset value
            custom_limits: Custom Greeks limits (optional)
            
        Returns:
            Comprehensive Greeks analysis
        """
        self.logger.info(f"Analyzing portfolio Greeks for {len(current_trades)} trades")
        
        # Calculate current portfolio Greeks
        portfolio_greeks = PortfolioGreeks(nav=nav)
        portfolio_greeks.update_from_trades(current_trades, nav)
        
        # Use custom limits or defaults
        limits = custom_limits or self.default_limits
        
        # Analyze Greeks vs limits
        violations = []
        warnings = []
        utilization = {}
        
        # Check delta limits
        if limits.max_net_delta is not None:
            delta_util = abs(portfolio_greeks.delta_per_100k) / limits.max_net_delta
            utilization["delta"] = delta_util
            
            if abs(portfolio_greeks.delta_per_100k) > limits.max_net_delta:
                violations.append(
                    f"Net delta {portfolio_greeks.delta_per_100k:.2f} exceeds limit {limits.max_net_delta:.2f}"
                )
            elif delta_util > 0.8:
                warnings.append(f"Delta utilization at {delta_util:.1%}")
        
        # Check vega limits
        if limits.min_net_vega is not None:
            if portfolio_greeks.vega_per_100k < limits.min_net_vega:
                violations.append(
                    f"Net vega {portfolio_greeks.vega_per_100k:.2f} below limit {limits.min_net_vega:.2f}"
                )
        
        if limits.max_net_vega is not None:
            vega_util = abs(portfolio_greeks.vega_per_100k) / abs(limits.max_net_vega)
            utilization["vega"] = vega_util
            
            if abs(portfolio_greeks.vega_per_100k) > abs(limits.max_net_vega):
                violations.append(
                    f"Net vega {portfolio_greeks.vega_per_100k:.2f} exceeds limit {limits.max_net_vega:.2f}"
                )
            elif vega_util > 0.8:
                warnings.append(f"Vega utilization at {vega_util:.1%}")
        
        # Check gamma limits
        if limits.max_net_gamma is not None:
            gamma_util = abs(portfolio_greeks.total_gamma) / limits.max_net_gamma
            utilization["gamma"] = gamma_util
            
            if abs(portfolio_greeks.total_gamma) > limits.max_net_gamma:
                violations.append(
                    f"Net gamma {portfolio_greeks.total_gamma:.2f} exceeds limit {limits.max_net_gamma:.2f}"
                )
            elif gamma_util > 0.8:
                warnings.append(f"Gamma utilization at {gamma_util:.1%}")
        
        # Check theta limits
        if limits.min_net_theta is not None and portfolio_greeks.total_theta < limits.min_net_theta:
            violations.append(
                f"Net theta {portfolio_greeks.total_theta:.2f} below limit {limits.min_net_theta:.2f}"
            )
        
        # Calculate risk score
        risk_score = self._calculate_portfolio_risk_score(portfolio_greeks, limits, utilization)
        
        # Generate hedge recommendations
        hedge_recommendations = self._generate_hedge_recommendations(
            portfolio_greeks, limits, violations
        )
        
        return GreeksAnalysis(
            current_greeks=portfolio_greeks,
            limits=limits,
            violations=violations,
            warnings=warnings,
            utilization=utilization,
            risk_score=risk_score,
            hedge_recommendations=hedge_recommendations
        )
    
    def assess_trade_impact(
        self,
        trade_candidate: TradeCandidate,
        current_trades: List[TradeCandidate],
        nav: Decimal,
        limits: Optional[GreeksLimits] = None
    ) -> GreeksImpact:
        """
        Assess the impact of adding a trade to the portfolio Greeks.
        
        Args:
            trade_candidate: Trade to assess
            current_trades: Current portfolio trades
            nav: Portfolio NAV
            limits: Greeks limits to check against
            
        Returns:
            Greeks impact analysis
        """
        # Calculate current Greeks
        current_greeks = PortfolioGreeks(nav=nav)
        current_greeks.update_from_trades(current_trades, nav)
        
        # Calculate projected Greeks with new trade
        projected_trades = current_trades + [trade_candidate]
        projected_greeks = PortfolioGreeks(nav=nav)
        projected_greeks.update_from_trades(projected_trades, nav)
        
        # Calculate changes
        delta_change = projected_greeks.total_delta - current_greeks.total_delta
        gamma_change = projected_greeks.total_gamma - current_greeks.total_gamma
        theta_change = projected_greeks.total_theta - current_greeks.total_theta
        vega_change = projected_greeks.total_vega - current_greeks.total_vega
        rho_change = projected_greeks.total_rho - current_greeks.total_rho
        
        # Check if within limits
        limits = limits or self.default_limits
        violations = self._check_greeks_violations(projected_greeks, limits)
        within_limits = len(violations) == 0
        
        return GreeksImpact(
            trade_candidate=trade_candidate,
            current_greeks=current_greeks,
            projected_greeks=projected_greeks,
            delta_change=delta_change,
            gamma_change=gamma_change,
            theta_change=theta_change,
            vega_change=vega_change,
            rho_change=rho_change,
            within_limits=within_limits,
            violations=violations
        )
    
    def filter_trades_by_greeks_constraints(
        self,
        trade_candidates: List[TradeCandidate],
        current_trades: List[TradeCandidate],
        nav: Decimal,
        limits: Optional[GreeksLimits] = None
    ) -> List[TradeCandidate]:
        """
        Filter trade candidates to only those that don't violate Greeks limits.
        
        Args:
            trade_candidates: Candidates to filter
            current_trades: Current portfolio trades
            nav: Portfolio NAV
            limits: Greeks limits
            
        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []
        limits = limits or self.default_limits
        
        for candidate in trade_candidates:
            impact = self.assess_trade_impact(candidate, current_trades, nav, limits)
            
            if impact.within_limits:
                filtered_candidates.append(candidate)
            else:
                self.logger.debug(
                    f"Filtered out {candidate.strategy.strategy_type.value} due to Greeks violations: "
                    f"{impact.violations}"
                )
        
        return filtered_candidates
    
    def suggest_greeks_neutral_positions(
        self,
        current_trades: List[TradeCandidate],
        target_greek: str,
        target_value: float = 0.0,
        nav: Decimal = Decimal('100000')
    ) -> List[Dict[str, Any]]:
        """
        Suggest positions to achieve Greeks neutrality.
        
        Args:
            current_trades: Current portfolio trades
            target_greek: Greek to neutralize ('delta', 'gamma', 'vega', etc.)
            target_value: Target value for the Greek
            nav: Portfolio NAV
            
        Returns:
            List of suggested hedge positions
        """
        # Calculate current Greeks
        portfolio_greeks = PortfolioGreeks(nav=nav)
        portfolio_greeks.update_from_trades(current_trades, nav)
        
        suggestions = []
        
        if target_greek == "delta":
            current_delta = portfolio_greeks.total_delta
            delta_to_hedge = current_delta - target_value
            
            if abs(delta_to_hedge) > 10:  # Only hedge significant exposures
                # Suggest underlying stock hedge
                shares_to_hedge = int(-delta_to_hedge / 100)  # Convert to shares
                suggestions.append({
                    "type": "stock_hedge",
                    "action": "buy" if shares_to_hedge > 0 else "sell",
                    "quantity": abs(shares_to_hedge),
                    "rationale": f"Hedge {delta_to_hedge:.0f} delta exposure"
                })
                
                # Suggest options hedge alternatives
                suggestions.extend(self._suggest_delta_hedge_options(delta_to_hedge))
        
        elif target_greek == "vega":
            current_vega = portfolio_greeks.total_vega
            vega_to_hedge = current_vega - target_value
            
            if abs(vega_to_hedge) > 50:  # Only hedge significant vega
                suggestions.extend(self._suggest_vega_hedge_options(vega_to_hedge))
        
        elif target_greek == "gamma":
            current_gamma = portfolio_greeks.total_gamma
            gamma_to_hedge = current_gamma - target_value
            
            if abs(gamma_to_hedge) > 5:  # Only hedge significant gamma
                suggestions.extend(self._suggest_gamma_hedge_options(gamma_to_hedge))
        
        return suggestions
    
    def calculate_greeks_correlation_risk(
        self,
        trades: List[TradeCandidate],
        nav: Decimal
    ) -> Dict[str, float]:
        """
        Calculate correlation risk between different Greeks exposures.
        
        Args:
            trades: Portfolio trades
            nav: Portfolio NAV
            
        Returns:
            Dictionary of correlation risk metrics
        """
        portfolio_greeks = PortfolioGreeks(nav=nav)
        portfolio_greeks.update_from_trades(trades, nav)
        
        # Calculate correlation-adjusted risk
        correlation_risks = {}
        
        # Delta-Gamma correlation risk
        delta_gamma_risk = abs(portfolio_greeks.total_delta * portfolio_greeks.total_gamma) * 0.3
        correlation_risks["delta_gamma"] = delta_gamma_risk
        
        # Vega-Gamma correlation risk
        vega_gamma_risk = abs(portfolio_greeks.total_vega * portfolio_greeks.total_gamma) * 0.4
        correlation_risks["vega_gamma"] = vega_gamma_risk
        
        # Theta-Vega correlation risk
        theta_vega_risk = abs(portfolio_greeks.total_theta * portfolio_greeks.total_vega) * 0.3
        correlation_risks["theta_vega"] = theta_vega_risk
        
        # Overall correlation risk score
        total_risk = sum(correlation_risks.values())
        correlation_risks["total_correlation_risk"] = total_risk
        
        return correlation_risks
    
    def _get_default_limits(self, risk_level: RiskLevel) -> GreeksLimits:
        """Get default Greeks limits based on risk level."""
        
        if risk_level == RiskLevel.CONSERVATIVE:
            return GreeksLimits(
                max_net_delta=0.20,  # ±20% per 100k NAV
                max_net_gamma=5.0,
                min_net_theta=-20.0,
                max_net_theta=5.0,
                max_net_vega=30.0,
                min_net_vega=-10.0,
                max_net_rho=20.0,
                min_net_rho=-20.0
            )
        elif risk_level == RiskLevel.MODERATE:
            return GreeksLimits(
                max_net_delta=0.30,  # ±30% per 100k NAV (per instructions)
                max_net_gamma=8.0,
                min_net_theta=-40.0,
                max_net_theta=10.0,
                max_net_vega=50.0,
                min_net_vega=-20.0,  # -0.05 × (NAV / 100k) per instructions
                max_net_rho=40.0,
                min_net_rho=-40.0
            )
        else:  # AGGRESSIVE
            return GreeksLimits(
                max_net_delta=0.50,  # ±50% per 100k NAV
                max_net_gamma=15.0,
                min_net_theta=-80.0,
                max_net_theta=20.0,
                max_net_vega=100.0,
                min_net_vega=-40.0,
                max_net_rho=80.0,
                min_net_rho=-80.0
            )
    
    def _check_greeks_violations(
        self,
        portfolio_greeks: PortfolioGreeks,
        limits: GreeksLimits
    ) -> List[str]:
        """Check for Greeks limit violations."""
        
        violations = []
        
        # Delta violations
        if limits.max_net_delta is not None:
            if abs(portfolio_greeks.delta_per_100k) > limits.max_net_delta:
                violations.append(f"Delta limit exceeded: {portfolio_greeks.delta_per_100k:.3f}")
        
        # Vega violations
        if limits.min_net_vega is not None:
            if portfolio_greeks.vega_per_100k < limits.min_net_vega:
                violations.append(f"Vega below minimum: {portfolio_greeks.vega_per_100k:.3f}")
        
        if limits.max_net_vega is not None:
            if portfolio_greeks.vega_per_100k > limits.max_net_vega:
                violations.append(f"Vega above maximum: {portfolio_greeks.vega_per_100k:.3f}")
        
        # Gamma violations
        if limits.max_net_gamma is not None:
            if abs(portfolio_greeks.total_gamma) > limits.max_net_gamma:
                violations.append(f"Gamma limit exceeded: {portfolio_greeks.total_gamma:.3f}")
        
        # Theta violations
        if limits.min_net_theta is not None:
            if portfolio_greeks.total_theta < limits.min_net_theta:
                violations.append(f"Theta below minimum: {portfolio_greeks.total_theta:.3f}")
        
        if limits.max_net_theta is not None:
            if portfolio_greeks.total_theta > limits.max_net_theta:
                violations.append(f"Theta above maximum: {portfolio_greeks.total_theta:.3f}")
        
        return violations
    
    def _calculate_portfolio_risk_score(
        self,
        portfolio_greeks: PortfolioGreeks,
        limits: GreeksLimits,
        utilization: Dict[str, float]
    ) -> float:
        """Calculate overall portfolio risk score (0-100)."""
        
        risk_components = []
        
        # Greeks utilization risk
        for greek, util in utilization.items():
            if util > 1.0:  # Over limit
                risk_components.append(100)
            elif util > 0.8:  # High utilization
                risk_components.append(80)
            elif util > 0.6:  # Moderate utilization
                risk_components.append(60)
            else:  # Low utilization
                risk_components.append(util * 50)
        
        # Concentration risk (simplified)
        total_delta = abs(portfolio_greeks.total_delta)
        total_vega = abs(portfolio_greeks.total_vega)
        
        if total_delta > 100 or total_vega > 100:
            risk_components.append(90)  # High concentration
        elif total_delta > 50 or total_vega > 50:
            risk_components.append(70)  # Moderate concentration
        
        # Correlation risk
        correlation_risks = self.calculate_greeks_correlation_risk([], portfolio_greeks.nav)
        if correlation_risks.get("total_correlation_risk", 0) > 100:
            risk_components.append(80)
        
        return statistics.mean(risk_components) if risk_components else 0
    
    def _generate_hedge_recommendations(
        self,
        portfolio_greeks: PortfolioGreeks,
        limits: GreeksLimits,
        violations: List[str]
    ) -> List[str]:
        """Generate hedge recommendations based on violations."""
        
        recommendations = []
        
        # Delta hedge recommendations
        if limits.max_net_delta and abs(portfolio_greeks.delta_per_100k) > limits.max_net_delta:
            if portfolio_greeks.delta_per_100k > 0:
                recommendations.append("Consider selling calls or buying puts to reduce positive delta")
            else:
                recommendations.append("Consider buying calls or selling puts to reduce negative delta")
        
        # Vega hedge recommendations
        if limits.min_net_vega and portfolio_greeks.vega_per_100k < limits.min_net_vega:
            recommendations.append("Consider buying options to increase vega exposure")
        elif limits.max_net_vega and portfolio_greeks.vega_per_100k > limits.max_net_vega:
            recommendations.append("Consider closing long option positions to reduce vega")
        
        # Gamma hedge recommendations
        if limits.max_net_gamma and abs(portfolio_greeks.total_gamma) > limits.max_net_gamma:
            recommendations.append("Consider reducing options positions near ATM to lower gamma")
        
        # Theta recommendations
        if limits.min_net_theta and portfolio_greeks.total_theta < limits.min_net_theta:
            recommendations.append("Portfolio theta too negative - consider profit-taking on short options")
        
        return recommendations
    
    def _suggest_delta_hedge_options(self, delta_to_hedge: float) -> List[Dict[str, Any]]:
        """Suggest options-based delta hedges."""
        
        suggestions = []
        
        if delta_to_hedge > 0:  # Need to reduce positive delta
            suggestions.append({
                "type": "options_hedge",
                "strategy": "short_calls",
                "delta_effect": "negative",
                "rationale": f"Sell calls to reduce {delta_to_hedge:.0f} positive delta"
            })
            suggestions.append({
                "type": "options_hedge", 
                "strategy": "long_puts",
                "delta_effect": "negative",
                "rationale": f"Buy puts to hedge {delta_to_hedge:.0f} positive delta"
            })
        else:  # Need to reduce negative delta
            suggestions.append({
                "type": "options_hedge",
                "strategy": "long_calls",
                "delta_effect": "positive",
                "rationale": f"Buy calls to offset {abs(delta_to_hedge):.0f} negative delta"
            })
            suggestions.append({
                "type": "options_hedge",
                "strategy": "short_puts",
                "delta_effect": "positive", 
                "rationale": f"Sell puts to reduce {abs(delta_to_hedge):.0f} negative delta"
            })
        
        return suggestions
    
    def _suggest_vega_hedge_options(self, vega_to_hedge: float) -> List[Dict[str, Any]]:
        """Suggest vega hedge options."""
        
        suggestions = []
        
        if vega_to_hedge > 0:  # Need to reduce positive vega
            suggestions.append({
                "type": "vega_hedge",
                "strategy": "sell_options",
                "vega_effect": "negative",
                "rationale": f"Sell options to reduce {vega_to_hedge:.0f} vega exposure"
            })
        else:  # Need to increase vega
            suggestions.append({
                "type": "vega_hedge",
                "strategy": "buy_options",
                "vega_effect": "positive",
                "rationale": f"Buy options to increase vega by {abs(vega_to_hedge):.0f}"
            })
        
        return suggestions
    
    def _suggest_gamma_hedge_options(self, gamma_to_hedge: float) -> List[Dict[str, Any]]:
        """Suggest gamma hedge options."""
        
        suggestions = []
        
        if gamma_to_hedge > 0:  # Need to reduce positive gamma
            suggestions.append({
                "type": "gamma_hedge",
                "strategy": "move_away_from_atm",
                "gamma_effect": "negative",
                "rationale": f"Move strikes away from ATM to reduce {gamma_to_hedge:.2f} gamma"
            })
        else:  # Need to increase gamma
            suggestions.append({
                "type": "gamma_hedge",
                "strategy": "move_toward_atm",
                "gamma_effect": "positive",
                "rationale": f"Add ATM options to increase gamma by {abs(gamma_to_hedge):.2f}"
            })
        
        return suggestions