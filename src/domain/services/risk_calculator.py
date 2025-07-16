"""
Risk metric calculations for portfolio and strategy analysis.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
import numpy as np
import logging

from ...data.models.options import OptionQuote, OptionType, Greeks
from ...data.models.market_data import OHLCVData, TechnicalIndicators
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, PortfolioGreeks, 
    TradeFilterCriteria
)
from ...infrastructure.error_handling import (
    handle_errors, CalculationError, RiskCalculationError
)

from .probability_calculator import ProbabilityCalculator, ProbabilityParams, ProbabilityModel


class RiskMeasure(Enum):
    """Types of risk measures."""
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TAIL_RISK = "tail_risk"


class ConfidenceLevel(Enum):
    """Standard confidence levels for risk calculations."""
    NINETY_PERCENT = 0.90
    NINETY_FIVE_PERCENT = 0.95
    NINETY_NINE_PERCENT = 0.99


@dataclass
class StressScenario:
    """Market stress scenario definition."""
    name: str
    price_change: float  # Percentage change in underlying
    volatility_change: float  # Change in implied volatility
    time_decay_days: int = 0  # Days of time decay
    interest_rate_change: float = 0.0  # Change in interest rates
    description: str = ""


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for strategies and portfolios."""
    # Basic risk measures
    value_at_risk_95: float
    conditional_var_95: float
    maximum_drawdown: float
    maximum_loss_scenario: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    # Stress test results
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    
    # Time-based metrics
    holding_period_var: Optional[float] = None
    time_to_min_profit: Optional[int] = None  # Days to minimum profit
    
    # Correlation risks
    correlation_risk_score: float = 0.0
    concentration_risk_score: float = 0.0


@dataclass
class PortfolioRiskProfile:
    """Portfolio-level risk assessment."""
    total_var_95: float
    total_cvar_95: float
    portfolio_beta: float
    diversification_ratio: float
    concentration_risk: float
    
    # Component risks
    individual_strategy_vars: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Risk attribution
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    marginal_vars: Dict[str, float] = field(default_factory=dict)


class RiskCalculator:
    """
    Advanced risk calculation engine for options strategies and portfolios.
    
    Features:
    - Value at Risk (VaR) calculations using multiple methodologies
    - Conditional VaR (Expected Shortfall) for tail risk assessment
    - Maximum loss scenarios under various market stress conditions
    - Sharpe ratio and risk-adjusted return estimations
    - Monte Carlo simulation for complex risk scenarios
    - Greeks-based risk decomposition
    - Portfolio-level risk aggregation with correlation effects
    - Stress testing against historical market events
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize probability calculator for simulations
        self.probability_calculator = ProbabilityCalculator()
        
        # Default stress scenarios
        self.default_stress_scenarios = [
            StressScenario("Black Monday 1987", -0.22, 2.0, 1, description="Historical crash scenario"),
            StressScenario("Flash Crash 2010", -0.09, 1.5, 1, description="Intraday crash scenario"),
            StressScenario("COVID Crash 2020", -0.34, 3.0, 14, description="Pandemic crash scenario"),
            StressScenario("Dot-com Crash 2000", -0.15, 1.8, 30, description="Tech bubble burst"),
            StressScenario("Financial Crisis 2008", -0.20, 2.5, 21, description="Credit crisis scenario"),
            StressScenario("Moderate Correction", -0.10, 1.2, 7, description="Standard correction"),
            StressScenario("Volatility Spike", 0.0, 2.0, 1, description="Pure volatility increase"),
            StressScenario("Extended Decline", -0.05, 1.1, 60, description="Slow grind down")
        ]
    
    @handle_errors(operation_name="calculate_strategy_risk")
    def calculate_strategy_risk_metrics(
        self,
        strategy: StrategyDefinition,
        current_price: Decimal,
        historical_data: Optional[List[OHLCVData]] = None,
        confidence_level: ConfidenceLevel = ConfidenceLevel.NINETY_FIVE_PERCENT,
        stress_scenarios: Optional[List[StressScenario]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a single strategy.
        
        Args:
            strategy: Strategy definition
            current_price: Current underlying price
            historical_data: Historical price data
            confidence_level: Confidence level for VaR calculations
            stress_scenarios: Custom stress scenarios
            
        Returns:
            Comprehensive risk metrics
        """
        self.logger.info(f"Calculating risk metrics for {strategy.strategy_type.value}")
        
        if stress_scenarios is None:
            stress_scenarios = self.default_stress_scenarios
        
        # Run Monte Carlo simulation for PnL distribution
        pnl_distribution = self._simulate_strategy_pnl_distribution(
            strategy, current_price, historical_data
        )
        
        # Calculate basic risk measures
        var_95 = self._calculate_value_at_risk(pnl_distribution, confidence_level.value)
        cvar_95 = self._calculate_conditional_var(pnl_distribution, confidence_level.value)
        max_drawdown = self._calculate_maximum_drawdown(pnl_distribution)
        
        # Calculate risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_distribution)
        sortino_ratio = self._calculate_sortino_ratio(pnl_distribution)
        calmar_ratio = self._calculate_calmar_ratio(pnl_distribution, max_drawdown)
        
        # Calculate distribution metrics
        skewness = self._calculate_skewness(pnl_distribution)
        kurtosis = self._calculate_kurtosis(pnl_distribution)
        tail_ratio = self._calculate_tail_ratio(pnl_distribution)
        
        # Run stress tests
        stress_results = self._run_stress_tests(strategy, current_price, stress_scenarios)
        max_loss_scenario = min(stress_results.values()) if stress_results else var_95
        
        # Calculate correlation and concentration risks
        correlation_risk = self._calculate_strategy_correlation_risk(strategy)
        concentration_risk = self._calculate_strategy_concentration_risk(strategy)
        
        return RiskMetrics(
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            maximum_drawdown=max_drawdown,
            maximum_loss_scenario=max_loss_scenario,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            stress_test_results=stress_results,
            correlation_risk_score=correlation_risk,
            concentration_risk_score=concentration_risk
        )
    
    @handle_errors(operation_name="calculate_portfolio_risk")
    def calculate_portfolio_risk_profile(
        self,
        strategies: List[TradeCandidate],
        nav: Decimal,
        historical_data: Optional[Dict[str, List[OHLCVData]]] = None,
        confidence_level: ConfidenceLevel = ConfidenceLevel.NINETY_FIVE_PERCENT
    ) -> PortfolioRiskProfile:
        """
        Calculate portfolio-level risk metrics with correlation effects.
        
        Args:
            strategies: List of portfolio strategies
            nav: Portfolio net asset value
            historical_data: Historical data by symbol
            confidence_level: Confidence level for calculations
            
        Returns:
            Portfolio risk profile
        """
        self.logger.info(f"Calculating portfolio risk for {len(strategies)} strategies")
        
        if not strategies:
            raise CalculationError("No strategies provided for portfolio risk calculation")
        
        # Calculate individual strategy VaRs
        individual_vars = {}
        strategy_pnl_distributions = {}
        
        for i, trade_candidate in enumerate(strategies):
            strategy = trade_candidate.strategy
            symbol = strategy.underlying
            
            # Get historical data for this symbol
            symbol_data = historical_data.get(symbol) if historical_data else None
            
            # Use last known price or default
            current_price = Decimal('100')  # Would get from market data in practice
            
            # Calculate strategy PnL distribution
            pnl_dist = self._simulate_strategy_pnl_distribution(
                strategy, current_price, symbol_data
            )
            strategy_pnl_distributions[f"strategy_{i}"] = pnl_dist
            
            # Calculate individual VaR
            var_95 = self._calculate_value_at_risk(pnl_dist, confidence_level.value)
            individual_vars[f"strategy_{i}"] = var_95
        
        # Calculate portfolio correlation matrix
        correlation_matrix = self._calculate_portfolio_correlations(strategy_pnl_distributions)
        
        # Calculate portfolio VaR with correlations
        portfolio_var = self._calculate_portfolio_var(individual_vars, correlation_matrix)
        portfolio_cvar = self._calculate_portfolio_cvar(strategy_pnl_distributions, confidence_level.value)
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(individual_vars, portfolio_var)
        
        # Calculate portfolio beta (simplified)
        portfolio_beta = self._calculate_portfolio_beta(strategies)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_portfolio_concentration_risk(strategies, nav)
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(individual_vars, correlation_matrix)
        
        # Calculate marginal VaRs
        marginal_vars = self._calculate_marginal_vars(individual_vars, correlation_matrix)
        
        return PortfolioRiskProfile(
            total_var_95=portfolio_var,
            total_cvar_95=portfolio_cvar,
            portfolio_beta=portfolio_beta,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            individual_strategy_vars=individual_vars,
            correlation_matrix=correlation_matrix,
            risk_contributions=risk_contributions,
            marginal_vars=marginal_vars
        )
    
    def _simulate_strategy_pnl_distribution(
        self,
        strategy: StrategyDefinition,
        current_price: Decimal,
        historical_data: Optional[List[OHLCVData]] = None,
        num_simulations: int = 10000
    ) -> List[float]:
        """Simulate PnL distribution for strategy using Monte Carlo."""
        
        if not strategy.expiration:
            raise CalculationError("Strategy missing expiration date")
        
        time_to_expiration = (strategy.expiration - date.today()).days / 365.25
        if time_to_expiration <= 0:
            raise CalculationError("Strategy already expired")
        
        # Estimate volatility
        volatility = self._estimate_volatility(historical_data) if historical_data else 0.25
        
        pnl_outcomes = []
        current_price_float = float(current_price)
        
        for _ in range(num_simulations):
            # Simulate final price using geometric Brownian motion
            drift = 0.05 - 0.5 * volatility ** 2  # Risk-free rate minus dividend yield
            random_shock = np.random.normal(0, 1) * volatility * math.sqrt(time_to_expiration)
            final_price = current_price_float * math.exp(drift * time_to_expiration + random_shock)
            
            # Calculate strategy PnL at this final price
            pnl = self._calculate_strategy_pnl_at_expiration(strategy, Decimal(str(final_price)), current_price)
            pnl_outcomes.append(float(pnl))
        
        return pnl_outcomes
    
    def _calculate_strategy_pnl_at_expiration(
        self,
        strategy: StrategyDefinition,
        final_price: Decimal,
        initial_price: Decimal
    ) -> Decimal:
        """Calculate strategy P&L at expiration for given final price."""
        
        total_pnl = Decimal('0')
        
        for leg in strategy.legs:
            # Calculate intrinsic value at expiration
            intrinsic_value = leg.option.calculate_intrinsic_value(final_price)
            
            # Calculate P&L for this leg
            if leg.direction.value == "BUY":
                # Paid premium, receive intrinsic value
                leg_pnl = intrinsic_value - (leg.option.mid_price or Decimal('0'))
            else:  # SELL
                # Received premium, pay intrinsic value
                leg_pnl = (leg.option.mid_price or Decimal('0')) - intrinsic_value
            
            # Multiply by quantity and contract size
            leg_pnl *= leg.quantity * 100  # 100 shares per contract
            
            total_pnl += leg_pnl
        
        return total_pnl
    
    def _calculate_value_at_risk(self, pnl_distribution: List[float], confidence_level: float) -> float:
        """Calculate Value at Risk at specified confidence level."""
        
        if not pnl_distribution:
            return 0.0
        
        sorted_pnls = sorted(pnl_distribution)
        var_index = int((1 - confidence_level) * len(sorted_pnls))
        return sorted_pnls[var_index]
    
    def _calculate_conditional_var(self, pnl_distribution: List[float], confidence_level: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        
        if not pnl_distribution:
            return 0.0
        
        sorted_pnls = sorted(pnl_distribution)
        var_index = int((1 - confidence_level) * len(sorted_pnls))
        
        # Average of all losses worse than VaR
        tail_losses = sorted_pnls[:var_index]
        return statistics.mean(tail_losses) if tail_losses else 0.0
    
    def _calculate_maximum_drawdown(self, pnl_distribution: List[float]) -> float:
        """Calculate maximum drawdown from PnL distribution."""
        
        if not pnl_distribution:
            return 0.0
        
        # Simulate cumulative returns
        cumulative_pnl = 0
        max_pnl = 0
        max_drawdown = 0
        
        for pnl in pnl_distribution:
            cumulative_pnl += pnl
            max_pnl = max(max_pnl, cumulative_pnl)
            drawdown = max_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, pnl_distribution: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio from PnL distribution."""
        
        if not pnl_distribution or len(pnl_distribution) < 2:
            return 0.0
        
        try:
            mean_return = statistics.mean(pnl_distribution)
            std_return = statistics.stdev(pnl_distribution)
            
            if std_return == 0:
                return 0.0
            
            # Annualize assuming daily returns
            annual_return = mean_return * 252
            annual_std = std_return * math.sqrt(252)
            
            return (annual_return - risk_free_rate) / annual_std
            
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, pnl_distribution: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        
        if not pnl_distribution:
            return 0.0
        
        try:
            mean_return = statistics.mean(pnl_distribution)
            negative_returns = [r for r in pnl_distribution if r < 0]
            
            if not negative_returns:
                return float('inf') if mean_return > 0 else 0.0
            
            downside_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0.0
            
            if downside_std == 0:
                return 0.0
            
            # Annualize
            annual_return = mean_return * 252
            annual_downside_std = downside_std * math.sqrt(252)
            
            return (annual_return - risk_free_rate) / annual_downside_std
            
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, pnl_distribution: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        
        if not pnl_distribution or max_drawdown == 0:
            return 0.0
        
        try:
            annual_return = statistics.mean(pnl_distribution) * 252
            return annual_return / max_drawdown
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, pnl_distribution: List[float]) -> float:
        """Calculate skewness of PnL distribution."""
        
        if len(pnl_distribution) < 3:
            return 0.0
        
        try:
            mean_pnl = statistics.mean(pnl_distribution)
            std_pnl = statistics.stdev(pnl_distribution)
            
            if std_pnl == 0:
                return 0.0
            
            skewness = sum((x - mean_pnl) ** 3 for x in pnl_distribution) / len(pnl_distribution)
            return skewness / (std_pnl ** 3)
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, pnl_distribution: List[float]) -> float:
        """Calculate kurtosis of PnL distribution."""
        
        if len(pnl_distribution) < 4:
            return 0.0
        
        try:
            mean_pnl = statistics.mean(pnl_distribution)
            std_pnl = statistics.stdev(pnl_distribution)
            
            if std_pnl == 0:
                return 0.0
            
            kurtosis = sum((x - mean_pnl) ** 4 for x in pnl_distribution) / len(pnl_distribution)
            return kurtosis / (std_pnl ** 4) - 3  # Excess kurtosis
            
        except Exception:
            return 0.0
    
    def _calculate_tail_ratio(self, pnl_distribution: List[float]) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        
        if len(pnl_distribution) < 20:
            return 1.0
        
        try:
            sorted_pnls = sorted(pnl_distribution)
            p95 = sorted_pnls[int(0.95 * len(sorted_pnls))]
            p5 = sorted_pnls[int(0.05 * len(sorted_pnls))]
            
            if p5 == 0:
                return float('inf') if p95 > 0 else 1.0
            
            return abs(p95 / p5)
            
        except Exception:
            return 1.0
    
    def _run_stress_tests(
        self,
        strategy: StrategyDefinition,
        current_price: Decimal,
        stress_scenarios: List[StressScenario]
    ) -> Dict[str, float]:
        """Run stress tests against predefined scenarios."""
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            try:
                # Calculate stressed price
                stressed_price = current_price * (1 + Decimal(str(scenario.price_change)))
                
                # Calculate PnL under stress
                stress_pnl = self._calculate_strategy_pnl_at_expiration(
                    strategy, stressed_price, current_price
                )
                
                # Apply time decay if specified
                if scenario.time_decay_days > 0:
                    theta_decay = self._estimate_theta_decay(strategy, scenario.time_decay_days)
                    stress_pnl += Decimal(str(theta_decay))
                
                stress_results[scenario.name] = float(stress_pnl)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate stress scenario {scenario.name}: {str(e)}")
                stress_results[scenario.name] = 0.0
        
        return stress_results
    
    def _calculate_strategy_correlation_risk(self, strategy: StrategyDefinition) -> float:
        """Calculate correlation risk score for strategy."""
        
        # Simplified correlation risk based on strategy type
        if strategy.strategy_type.value in ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"]:
            return 0.3  # Moderate correlation with underlying
        elif strategy.strategy_type.value == "IRON_CONDOR":
            return 0.1  # Low correlation risk (market neutral)
        else:
            return 0.5  # Default moderate risk
    
    def _calculate_strategy_concentration_risk(self, strategy: StrategyDefinition) -> float:
        """Calculate concentration risk score for strategy."""
        
        # Risk based on number of legs and strike concentration
        num_legs = len(strategy.legs)
        
        if num_legs == 1:
            return 1.0  # High concentration
        elif num_legs == 2:
            return 0.6  # Moderate concentration
        elif num_legs >= 4:
            return 0.2  # Low concentration (iron condor/butterfly)
        else:
            return 0.4  # Default
    
    def _calculate_portfolio_correlations(
        self,
        strategy_distributions: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate correlation matrix between strategies."""
        
        correlation_matrix = {}
        strategy_names = list(strategy_distributions.keys())
        
        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names):
                if i <= j:  # Only calculate upper triangle + diagonal
                    if strategy1 == strategy2:
                        correlation = 1.0
                    else:
                        correlation = self._calculate_correlation(
                            strategy_distributions[strategy1],
                            strategy_distributions[strategy2]
                        )
                    correlation_matrix[(strategy1, strategy2)] = correlation
                    if i != j:  # Also set symmetric entry
                        correlation_matrix[(strategy2, strategy1)] = correlation
        
        return correlation_matrix
    
    def _calculate_correlation(self, series1: List[float], series2: List[float]) -> float:
        """Calculate correlation coefficient between two series."""
        
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        
        try:
            mean1 = statistics.mean(series1)
            mean2 = statistics.mean(series2)
            
            numerator = sum((x1 - mean1) * (x2 - mean2) for x1, x2 in zip(series1, series2))
            
            sum_sq1 = sum((x1 - mean1) ** 2 for x1 in series1)
            sum_sq2 = sum((x2 - mean2) ** 2 for x2 in series2)
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception:
            return 0.0
    
    def _calculate_portfolio_var(
        self,
        individual_vars: Dict[str, float],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate portfolio VaR with correlation effects."""
        
        strategy_names = list(individual_vars.keys())
        
        if len(strategy_names) == 1:
            return list(individual_vars.values())[0]
        
        # Portfolio variance calculation with correlations
        portfolio_variance = 0.0
        
        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names):
                var1 = individual_vars[strategy1]
                var2 = individual_vars[strategy2]
                
                correlation = correlation_matrix.get((strategy1, strategy2), 0.0)
                
                # Assume equal weights for simplicity
                weight1 = weight2 = 1.0 / len(strategy_names)
                
                portfolio_variance += weight1 * weight2 * var1 * var2 * correlation
        
        return math.sqrt(max(0, portfolio_variance))
    
    def _calculate_portfolio_cvar(
        self,
        strategy_distributions: Dict[str, List[float]],
        confidence_level: float
    ) -> float:
        """Calculate portfolio Conditional VaR."""
        
        # Combine all strategy PnLs (assuming equal weights)
        if not strategy_distributions:
            return 0.0
        
        # Get minimum length to ensure all series have same length
        min_length = min(len(dist) for dist in strategy_distributions.values())
        
        # Calculate portfolio PnL for each simulation
        portfolio_pnls = []
        weight = 1.0 / len(strategy_distributions)
        
        for i in range(min_length):
            portfolio_pnl = sum(weight * dist[i] for dist in strategy_distributions.values())
            portfolio_pnls.append(portfolio_pnl)
        
        return self._calculate_conditional_var(portfolio_pnls, confidence_level)
    
    def _calculate_diversification_ratio(
        self,
        individual_vars: Dict[str, float],
        portfolio_var: float
    ) -> float:
        """Calculate diversification ratio."""
        
        if not individual_vars or portfolio_var == 0:
            return 1.0
        
        # Weighted average of individual VaRs
        weight = 1.0 / len(individual_vars)
        weighted_avg_var = sum(weight * var for var in individual_vars.values())
        
        if weighted_avg_var == 0:
            return 1.0
        
        return weighted_avg_var / portfolio_var
    
    def _calculate_portfolio_beta(self, strategies: List[TradeCandidate]) -> float:
        """Calculate portfolio beta (simplified)."""
        
        if not strategies:
            return 1.0
        
        # Simplified beta calculation based on strategy types
        total_beta = 0.0
        weight = 1.0 / len(strategies)
        
        for trade_candidate in strategies:
            strategy = trade_candidate.strategy
            
            # Assign beta based on strategy type
            if strategy.strategy_type.value in ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"]:
                strategy_beta = 0.3  # Moderate market exposure
            elif strategy.strategy_type.value == "IRON_CONDOR":
                strategy_beta = 0.1  # Low market exposure
            elif strategy.strategy_type.value == "COVERED_CALL":
                strategy_beta = 0.8  # High market exposure
            else:
                strategy_beta = 0.5  # Default
            
            total_beta += weight * strategy_beta
        
        return total_beta
    
    def _calculate_portfolio_concentration_risk(
        self,
        strategies: List[TradeCandidate],
        nav: Decimal
    ) -> float:
        """Calculate portfolio concentration risk."""
        
        if not strategies:
            return 0.0
        
        # Calculate position sizes as percentage of NAV
        position_weights = []
        
        for trade_candidate in strategies:
            strategy = trade_candidate.strategy
            
            # Estimate position size based on max loss
            if strategy.max_loss:
                position_size = float(strategy.max_loss) / float(nav)
            else:
                position_size = 0.01  # Default 1%
            
            position_weights.append(position_size)
        
        if not position_weights:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        hhi = sum(weight ** 2 for weight in position_weights)
        
        # Normalize to 0-1 scale (1 = maximum concentration)
        max_hhi = 1.0  # If all capital in one position
        return hhi / max_hhi
    
    def _calculate_risk_contributions(
        self,
        individual_vars: Dict[str, float],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """Calculate risk contribution of each strategy to portfolio risk."""
        
        risk_contributions = {}
        total_var = sum(individual_vars.values())
        
        if total_var == 0:
            return {strategy: 0.0 for strategy in individual_vars.keys()}
        
        # Simplified risk contribution calculation
        for strategy in individual_vars.keys():
            # Weight by individual VaR and average correlation
            avg_correlation = self._get_average_correlation(strategy, correlation_matrix)
            contribution = individual_vars[strategy] * avg_correlation / total_var
            risk_contributions[strategy] = contribution
        
        return risk_contributions
    
    def _calculate_marginal_vars(
        self,
        individual_vars: Dict[str, float],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """Calculate marginal VaR of each strategy."""
        
        marginal_vars = {}
        
        for strategy in individual_vars.keys():
            # Simplified marginal VaR calculation
            avg_correlation = self._get_average_correlation(strategy, correlation_matrix)
            marginal_var = individual_vars[strategy] * avg_correlation
            marginal_vars[strategy] = marginal_var
        
        return marginal_vars
    
    def _get_average_correlation(
        self,
        strategy: str,
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """Get average correlation of strategy with all others."""
        
        correlations = []
        
        for key, correlation in correlation_matrix.items():
            if strategy in key and key[0] != key[1]:  # Exclude self-correlation
                correlations.append(correlation)
        
        return statistics.mean(correlations) if correlations else 0.0
    
    def _estimate_volatility(self, historical_data: List[OHLCVData]) -> float:
        """Estimate historical volatility from price data."""
        
        if len(historical_data) < 20:
            return 0.25  # Default 25%
        
        # Calculate daily returns
        prices = [float(candle.close) for candle in historical_data[-60:]]  # Last 60 days
        returns = []
        
        for i in range(1, len(prices)):
            daily_return = math.log(prices[i] / prices[i-1])
            returns.append(daily_return)
        
        if len(returns) < 2:
            return 0.25
        
        # Annualized volatility
        daily_vol = statistics.stdev(returns)
        return daily_vol * math.sqrt(252)
    
    def _estimate_theta_decay(self, strategy: StrategyDefinition, days: int) -> float:
        """Estimate theta decay over specified days."""
        
        # Simplified theta estimation
        if not strategy.legs:
            return 0.0
        
        total_theta = 0.0
        
        for leg in strategy.legs:
            if leg.option.greeks and leg.option.greeks.theta:
                leg_theta = float(leg.option.greeks.theta)
                
                # Apply direction (buy = negative theta, sell = positive theta)
                if leg.direction.value == "BUY":
                    leg_theta = -abs(leg_theta)
                else:
                    leg_theta = abs(leg_theta)
                
                total_theta += leg_theta * leg.quantity
        
        return total_theta * days