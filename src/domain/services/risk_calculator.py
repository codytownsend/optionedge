"""
Risk calculation service for comprehensive risk analysis and scenario testing.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import math
import logging

from ..entities.strategy import Strategy
from ..entities.portfolio import Position, PortfolioConstraints
from ...data.models.trades import TradeCandidate
from ...data.models.options import Greeks

logger = logging.getLogger(__name__)


class RiskScenarioType(str, Enum):
    """Types of risk scenarios."""
    MARKET_CRASH = "market_crash"           # -20% to -50% market move
    MARKET_RALLY = "market_rally"           # +20% to +50% market move  
    VOLATILITY_SPIKE = "volatility_spike"   # 2x to 5x volatility increase
    VOLATILITY_CRUSH = "volatility_crush"   # 50% to 80% volatility decrease
    TIME_DECAY = "time_decay"               # Fast forward in time
    INTEREST_RATE_CHANGE = "rate_change"    # Interest rate shifts
    LIQUIDITY_CRISIS = "liquidity_crisis"   # Wide spreads, low volume


@dataclass
class RiskScenario:
    """Definition of a risk scenario for stress testing."""
    
    scenario_type: RiskScenarioType
    name: str
    description: str
    
    # Market parameters
    underlying_price_change: Optional[float] = None  # Percentage change
    volatility_multiplier: Optional[float] = None     # Volatility scaling factor
    time_forward_days: Optional[int] = None           # Days to fast-forward
    interest_rate_change: Optional[float] = None      # Rate change in basis points
    
    # Liquidity parameters
    spread_multiplier: Optional[float] = None         # Bid-ask spread scaling
    volume_multiplier: Optional[float] = None         # Volume scaling


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position or portfolio."""
    
    # Basic metrics
    current_value: Decimal
    max_profit: Optional[Decimal]
    max_loss: Optional[Decimal]
    probability_of_profit: Optional[float]
    
    # Greeks
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    
    # Risk measures
    value_at_risk_95: Optional[Decimal]
    conditional_var_95: Optional[Decimal]
    maximum_drawdown: Optional[Decimal]
    sharpe_ratio: Optional[float]
    
    # Stress test results
    stress_test_results: Dict[str, Decimal] = None
    
    # Time-based metrics
    break_even_time: Optional[int] = None  # Days to break even
    profit_target_time: Optional[int] = None  # Days to reach profit target
    
    def __post_init__(self):
        if self.stress_test_results is None:
            self.stress_test_results = {}


class RiskCalculator:
    """
    Service for calculating comprehensive risk metrics and performing scenario analysis.
    
    This service provides advanced risk calculations including Greeks analysis,
    Value at Risk, stress testing, and portfolio-level risk aggregation.
    """
    
    def __init__(self):
        self.default_scenarios = self._create_default_scenarios()
    
    def calculate_strategy_risk(self, 
                              strategy: Strategy,
                              confidence_level: float = 0.95,
                              time_horizon_days: int = 30) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a strategy.
        
        Args:
            strategy: Strategy to analyze
            confidence_level: Confidence level for VaR calculation
            time_horizon_days: Time horizon for risk calculations
            
        Returns:
            Comprehensive risk metrics
        """
        logger.debug(f"Calculating risk metrics for {strategy.strategy_type}")
        
        # Basic metrics
        current_value = strategy.calculate_net_premium()
        max_profit = strategy.calculate_max_profit()
        max_loss = strategy.calculate_max_loss()
        prob_profit = strategy.calculate_probability_of_profit()
        
        # Greeks
        net_greeks = strategy.calculate_net_greeks()
        
        # Value at Risk calculation
        var_95 = self._calculate_value_at_risk(strategy, confidence_level, time_horizon_days)
        cvar_95 = self._calculate_conditional_var(strategy, confidence_level, time_horizon_days)
        
        # Stress testing
        stress_results = self._perform_stress_tests(strategy, self.default_scenarios)
        
        # Time-based analysis
        break_even_time = self._calculate_break_even_time(strategy)
        
        return RiskMetrics(
            current_value=current_value,
            max_profit=max_profit,
            max_loss=max_loss,
            probability_of_profit=prob_profit,
            delta=net_greeks.delta,
            gamma=net_greeks.gamma,
            theta=net_greeks.theta,
            vega=net_greeks.vega,
            rho=net_greeks.rho,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            stress_test_results=stress_results,
            break_even_time=break_even_time
        )
    
    def calculate_portfolio_risk(self, 
                               position: Position,
                               confidence_level: float = 0.95) -> RiskMetrics:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            position: Current portfolio position
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Portfolio risk metrics
        """
        logger.debug(f"Calculating portfolio risk for {position.portfolio_id}")
        
        # Aggregate portfolio values
        current_value = position.nav
        total_unrealized = position.get_total_unrealized_pnl()
        
        # Aggregate Greeks
        net_delta = position.get_net_delta()
        net_vega = position.get_net_vega()
        net_theta = position.get_net_theta()
        
        # Portfolio-level VaR (simplified aggregation)
        portfolio_var = self._calculate_portfolio_var(position, confidence_level)
        
        # Stress test portfolio
        portfolio_stress_results = self._perform_portfolio_stress_tests(position)
        
        return RiskMetrics(
            current_value=current_value,
            max_profit=None,  # Not applicable at portfolio level
            max_loss=None,    # Not applicable at portfolio level
            probability_of_profit=None,
            delta=net_delta,
            gamma=0.0,  # Would need to aggregate from individual strategies
            theta=net_theta,
            vega=net_vega,
            rho=0.0,
            value_at_risk_95=portfolio_var,
            stress_test_results=portfolio_stress_results
        )
    
    def scenario_analysis(self, 
                         strategy: Strategy,
                         scenarios: List[RiskScenario]) -> Dict[str, Decimal]:
        """
        Perform scenario analysis on a strategy.
        
        Args:
            strategy: Strategy to analyze
            scenarios: List of scenarios to test
            
        Returns:
            Dictionary mapping scenario names to P&L impacts
        """
        results = {}
        
        for scenario in scenarios:
            try:
                pnl_impact = self._calculate_scenario_impact(strategy, scenario)
                results[scenario.name] = pnl_impact
            except Exception as e:
                logger.warning(f"Failed to calculate scenario {scenario.name}: {e}")
                results[scenario.name] = Decimal('0')
        
        return results
    
    def calculate_option_sensitivities(self, 
                                     strategy: Strategy,
                                     price_range_pct: float = 0.20,
                                     vol_range_pct: float = 0.50,
                                     time_steps: int = 10) -> Dict[str, List[Tuple[float, Decimal]]]:
        """
        Calculate option sensitivities across ranges of parameters.
        
        Args:
            strategy: Strategy to analyze
            price_range_pct: Price range as percentage of current price
            vol_range_pct: Volatility range as percentage of current vol
            time_steps: Number of steps in each dimension
            
        Returns:
            Dictionary of sensitivity analyses
        """
        if not strategy.underlying_price:
            return {}
        
        sensitivities = {}
        base_price = float(strategy.underlying_price)
        
        # Price sensitivity
        price_points = []
        price_min = base_price * (1 - price_range_pct)
        price_max = base_price * (1 + price_range_pct)
        
        for i in range(time_steps + 1):
            price = price_min + (price_max - price_min) * i / time_steps
            pnl = strategy._calculate_pnl_at_price(Decimal(str(price)))
            price_points.append((price, pnl))
        
        sensitivities['price_sensitivity'] = price_points
        
        # Time decay sensitivity
        time_points = []
        days_to_exp = strategy.get_days_to_expiration()
        
        for i in range(min(days_to_exp, time_steps) + 1):
            # Simplified time decay calculation
            time_factor = 1 - (i / days_to_exp) if days_to_exp > 0 else 1
            # This is a simplified approach - real implementation would use Greeks
            estimated_pnl = strategy.calculate_net_premium() * Decimal(str(time_factor))
            time_points.append((i, estimated_pnl))
        
        sensitivities['time_decay'] = time_points
        
        return sensitivities
    
    def _calculate_value_at_risk(self, 
                               strategy: Strategy,
                               confidence_level: float,
                               time_horizon_days: int) -> Optional[Decimal]:
        """Calculate Value at Risk using Monte Carlo simulation."""
        if not strategy.underlying_price:
            return None
        
        # Get average implied volatility
        ivs = []
        for leg in strategy.legs:
            if leg.option.implied_volatility:
                ivs.append(leg.option.implied_volatility)
        
        if not ivs:
            return None
        
        avg_vol = sum(ivs) / len(ivs)
        
        # Monte Carlo simulation
        num_simulations = 1000
        pnl_outcomes = []
        
        current_price = float(strategy.underlying_price)
        time_to_horizon = time_horizon_days / 365.0
        
        for _ in range(num_simulations):
            # Simulate price at time horizon
            final_price = self._simulate_price_gbm(current_price, avg_vol, time_to_horizon)
            
            # Calculate P&L at final price
            pnl = strategy._calculate_pnl_at_price(Decimal(str(final_price)))
            pnl_outcomes.append(float(pnl))
        
        # Calculate VaR
        pnl_outcomes.sort()
        var_index = int((1 - confidence_level) * num_simulations)
        var_value = pnl_outcomes[var_index]
        
        return Decimal(str(var_value))
    
    def _calculate_conditional_var(self,
                                 strategy: Strategy,
                                 confidence_level: float,
                                 time_horizon_days: int) -> Optional[Decimal]:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self._calculate_value_at_risk(strategy, confidence_level, time_horizon_days)
        
        if not var:
            return None
        
        # Simplified CVaR calculation - would need full simulation data
        # CVaR is typically 20-50% worse than VaR
        cvar_multiplier = 1.3
        return var * Decimal(str(cvar_multiplier))
    
    def _perform_stress_tests(self, 
                            strategy: Strategy,
                            scenarios: List[RiskScenario]) -> Dict[str, Decimal]:
        """Perform stress tests on strategy."""
        results = {}
        
        for scenario in scenarios:
            try:
                impact = self._calculate_scenario_impact(strategy, scenario)
                results[scenario.name] = impact
            except Exception as e:
                logger.warning(f"Stress test failed for {scenario.name}: {e}")
                results[scenario.name] = Decimal('0')
        
        return results
    
    def _calculate_scenario_impact(self, 
                                 strategy: Strategy,
                                 scenario: RiskScenario) -> Decimal:
        """Calculate P&L impact of a specific scenario."""
        if not strategy.underlying_price:
            return Decimal('0')
        
        current_price = strategy.underlying_price
        
        # Apply scenario to underlying price
        if scenario.underlying_price_change:
            scenario_price = current_price * (1 + scenario.underlying_price_change)
        else:
            scenario_price = current_price
        
        # Calculate P&L at scenario price
        scenario_pnl = strategy._calculate_pnl_at_price(scenario_price)
        base_pnl = strategy.calculate_net_premium()
        
        # Return the impact (difference from base case)
        return scenario_pnl - base_pnl
    
    def _calculate_portfolio_var(self, 
                               position: Position,
                               confidence_level: float) -> Optional[Decimal]:
        """Calculate portfolio-level Value at Risk."""
        # Simplified portfolio VaR - would need correlation matrix in real implementation
        total_var_squared = Decimal('0')
        
        for trade in position.active_trades:
            strategy = trade.trade_candidate.strategy
            individual_var = self._calculate_value_at_risk(strategy, confidence_level, 30)
            
            if individual_var:
                total_var_squared += individual_var ** 2
        
        if total_var_squared > 0:
            # Simple sum (assumes no correlation)
            return total_var_squared.sqrt()
        
        return None
    
    def _perform_portfolio_stress_tests(self, position: Position) -> Dict[str, Decimal]:
        """Perform stress tests on entire portfolio."""
        results = {}
        
        for scenario in self.default_scenarios:
            total_impact = Decimal('0')
            
            for trade in position.active_trades:
                strategy = trade.trade_candidate.strategy
                impact = self._calculate_scenario_impact(strategy, scenario)
                total_impact += impact
            
            results[scenario.name] = total_impact
        
        return results
    
    def _calculate_break_even_time(self, strategy: Strategy) -> Optional[int]:
        """Calculate days to break even (simplified)."""
        net_greeks = strategy.calculate_net_greeks()
        
        if not net_greeks.theta or net_greeks.theta >= 0:
            return None
        
        current_pnl = strategy.calculate_net_premium()
        daily_theta = net_greeks.theta  # Simplified - theta is already daily
        
        if current_pnl >= 0:
            return 0  # Already profitable
        
        # Days to break even = -current_loss / daily_theta_gain
        days_to_break_even = -float(current_pnl) / daily_theta
        
        return max(0, int(days_to_break_even))
    
    def _simulate_price_gbm(self, 
                          current_price: float,
                          volatility: float,
                          time_years: float) -> float:
        """Simulate price using Geometric Brownian Motion."""
        import random
        
        # Risk-free rate (simplified)
        r = 0.05
        
        # Generate random normal variable
        z = random.gauss(0, 1)
        
        # GBM formula
        drift = (r - 0.5 * volatility**2) * time_years
        diffusion = volatility * math.sqrt(time_years) * z
        
        final_price = current_price * math.exp(drift + diffusion)
        return final_price
    
    def _create_default_scenarios(self) -> List[RiskScenario]:
        """Create default stress testing scenarios."""
        return [
            RiskScenario(
                scenario_type=RiskScenarioType.MARKET_CRASH,
                name="Market Crash -20%",
                description="20% market decline",
                underlying_price_change=-0.20
            ),
            RiskScenario(
                scenario_type=RiskScenarioType.MARKET_CRASH,
                name="Severe Market Crash -35%",
                description="35% market decline",
                underlying_price_change=-0.35
            ),
            RiskScenario(
                scenario_type=RiskScenarioType.MARKET_RALLY,
                name="Market Rally +20%",
                description="20% market rally",
                underlying_price_change=0.20
            ),
            RiskScenario(
                scenario_type=RiskScenarioType.VOLATILITY_SPIKE,
                name="Volatility Spike 3x",
                description="Volatility increases 3x",
                volatility_multiplier=3.0
            ),
            RiskScenario(
                scenario_type=RiskScenarioType.VOLATILITY_CRUSH,
                name="Volatility Crush 50%",
                description="Volatility drops 50%",
                volatility_multiplier=0.5
            ),
            RiskScenario(
                scenario_type=RiskScenarioType.TIME_DECAY,
                name="Time Decay 1 Week",
                description="Fast forward 1 week",
                time_forward_days=7
            ),
            RiskScenario(
                scenario_type=RiskScenarioType.TIME_DECAY,
                name="Time Decay 2 Weeks",
                description="Fast forward 2 weeks",
                time_forward_days=14
            )
        ]


class PortfolioRiskAnalyzer:
    """
    Specialized risk analyzer for portfolio-level risk management.
    
    This class focuses on portfolio-level risk aggregation, correlation
    analysis, and portfolio optimization from a risk perspective.
    """
    
    def __init__(self, risk_calculator: RiskCalculator):
        self.risk_calculator = risk_calculator
    
    def analyze_portfolio_risk(self, position: Position) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk analysis.
        
        Args:
            position: Current portfolio position
            
        Returns:
            Detailed risk analysis results
        """
        # Basic risk metrics
        risk_metrics = self.risk_calculator.calculate_portfolio_risk(position)
        
        # Concentration analysis
        concentration_analysis = self._analyze_concentration(position)
        
        # Liquidity analysis
        liquidity_analysis = self._analyze_liquidity(position)
        
        # Greeks analysis
        greeks_analysis = self._analyze_greeks_exposure(position)
        
        return {
            'risk_metrics': risk_metrics,
            'concentration_analysis': concentration_analysis,
            'liquidity_analysis': liquidity_analysis,
            'greeks_analysis': greeks_analysis,
            'overall_risk_score': self._calculate_risk_score(position)
        }
    
    def _analyze_concentration(self, position: Position) -> Dict[str, Any]:
        """Analyze portfolio concentration risks."""
        sector_exposure = position.get_trades_by_sector()
        strategy_exposure = position.get_trades_by_strategy_type()
        
        # Calculate concentration metrics
        total_trades = len(position.active_trades)
        max_sector_concentration = max(sector_exposure.values()) if sector_exposure else 0
        max_strategy_concentration = max(strategy_exposure.values()) if strategy_exposure else 0
        
        return {
            'sector_exposure': dict(sector_exposure),
            'strategy_exposure': {k.value: v for k, v in strategy_exposure.items()},
            'max_sector_concentration_pct': max_sector_concentration / total_trades if total_trades > 0 else 0,
            'max_strategy_concentration_pct': max_strategy_concentration / total_trades if total_trades > 0 else 0,
            'diversification_score': self._calculate_diversification_score(sector_exposure, strategy_exposure)
        }
    
    def _analyze_liquidity(self, position: Position) -> Dict[str, Any]:
        """Analyze portfolio liquidity profile."""
        total_trades = len(position.active_trades)
        liquid_trades = 0
        total_volume = 0
        total_open_interest = 0
        
        for trade in position.active_trades:
            strategy = trade.trade_candidate.strategy
            
            # Check if all legs are liquid
            all_legs_liquid = True
            for leg in strategy.legs:
                volume = leg.option.volume or 0
                open_interest = leg.option.open_interest or 0
                
                total_volume += volume
                total_open_interest += open_interest
                
                if volume < 10 or open_interest < 100:
                    all_legs_liquid = False
            
            if all_legs_liquid:
                liquid_trades += 1
        
        return {
            'total_trades': total_trades,
            'liquid_trades': liquid_trades,
            'liquidity_ratio': liquid_trades / total_trades if total_trades > 0 else 1.0,
            'avg_volume_per_trade': total_volume / total_trades if total_trades > 0 else 0,
            'avg_open_interest_per_trade': total_open_interest / total_trades if total_trades > 0 else 0
        }
    
    def _analyze_greeks_exposure(self, position: Position) -> Dict[str, Any]:
        """Analyze portfolio Greeks exposure."""
        constraints = position.constraints
        
        net_delta = position.get_net_delta()
        net_vega = position.get_net_vega()
        net_theta = position.get_net_theta()
        
        max_delta = constraints.max_delta_exposure
        min_vega = constraints.min_vega_exposure
        
        return {
            'net_delta': net_delta,
            'net_vega': net_vega,
            'net_theta': net_theta,
            'delta_utilization': abs(net_delta) / max_delta if max_delta > 0 else 0,
            'vega_headroom': net_vega - min_vega,
            'theta_income_daily': net_theta,  # Daily theta income
            'greeks_within_limits': (abs(net_delta) <= max_delta and net_vega >= min_vega)
        }
    
    def _calculate_diversification_score(self, 
                                       sector_exposure: Dict[str, int],
                                       strategy_exposure: Dict[Any, int]) -> float:
        """Calculate diversification score (0-100)."""
        total_trades = sum(sector_exposure.values())
        
        if total_trades <= 1:
            return 100.0  # Single trade is fully "diversified" in its own category
        
        # Calculate Herfindahl-Hirschman Index for concentration
        sector_hhi = sum((count / total_trades) ** 2 for count in sector_exposure.values())
        strategy_hhi = sum((count / total_trades) ** 2 for count in strategy_exposure.values())
        
        # Average HHI (lower is more diversified)
        avg_hhi = (sector_hhi + strategy_hhi) / 2
        
        # Convert to diversification score (higher is better)
        # HHI ranges from 1/n to 1, where n is number of categories
        diversification_score = (1 - avg_hhi) * 100
        
        return max(0, min(100, diversification_score))
    
    def _calculate_risk_score(self, position: Position) -> float:
        """Calculate overall portfolio risk score (0-100, lower is riskier)."""
        score = 100.0
        
        # Utilization penalties
        utilization = position.calculate_utilization_metrics()
        
        if utilization['trade_utilization'] > 0.9:
            score -= 15
        elif utilization['trade_utilization'] > 0.8:
            score -= 8
        
        if utilization['capital_utilization'] > 0.9:
            score -= 20
        elif utilization['capital_utilization'] > 0.8:
            score -= 10
        
        if utilization['delta_utilization'] > 0.9:
            score -= 15
        elif utilization['delta_utilization'] > 0.8:
            score -= 8
        
        # P&L impact
        unrealized_pnl = float(position.get_total_unrealized_pnl())
        nav = float(position.nav)
        
        if unrealized_pnl < -nav * 0.1:  # > 10% loss
            score -= 25
        elif unrealized_pnl < -nav * 0.05:  # > 5% loss
            score -= 15
        elif unrealized_pnl < 0:  # Any loss
            score -= 5
        
        return max(0, score)