"""
Portfolio-level risk controls and position management system.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
import logging

from ...data.models.options import OptionQuote, OptionType, Greeks
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, TradeLeg, TradeDirection,
    PortfolioGreeks, TradeFilterCriteria, StrategyType
)
from ...infrastructure.error_handling import (
    handle_errors, RiskManagementError, BusinessLogicError
)

from .constraint_engine import ConstraintViolation, ConstraintType, ConstraintSeverity, GICS_SECTORS


class RiskLevel(Enum):
    """Portfolio risk level classifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class PositionSizeMethod(Enum):
    """Position sizing methodologies."""
    KELLY_CRITERION = "kelly_criterion"
    FIXED_DOLLAR = "fixed_dollar"
    PERCENT_OF_NAV = "percent_of_nav"
    RISK_PARITY = "risk_parity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class RiskBudget:
    """Portfolio risk budget allocation."""
    total_risk_budget: Decimal  # Total $ amount at risk
    per_trade_limit: Decimal    # Maximum $ risk per trade
    sector_limit: Decimal       # Maximum $ risk per sector
    strategy_type_limit: Decimal # Maximum $ risk per strategy type
    correlation_limit: float    # Maximum correlation between positions
    concentration_limit: float  # Maximum % of portfolio in single position


@dataclass
class PortfolioRiskMetrics:
    """Current portfolio risk assessment."""
    total_value_at_risk: Decimal
    total_exposure: Decimal
    leverage_ratio: float
    diversification_score: float
    concentration_risk: float
    correlation_risk: float
    
    # Greeks exposure
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    
    # Sector allocation
    sector_exposure: Dict[str, Decimal] = field(default_factory=dict)
    sector_risk_scores: Dict[str, float] = field(default_factory=dict)
    
    # Strategy type allocation
    strategy_allocation: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation."""
    recommended_contracts: int
    max_contracts: int
    sizing_method: PositionSizeMethod
    risk_amount: Decimal
    justification: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioAction:
    """Recommended portfolio action."""
    action_type: str  # "reduce", "hedge", "rebalance", "hold"
    priority: str     # "high", "medium", "low"
    description: str
    affected_positions: List[str] = field(default_factory=list)
    recommended_adjustments: Dict[str, Any] = field(default_factory=dict)


class PortfolioRiskController:
    """
    Portfolio-level risk management and position control system.
    
    Features:
    - Real-time portfolio risk monitoring with Greeks aggregation
    - Position sizing using Kelly criterion and risk parity methods
    - Sector and strategy type diversification enforcement
    - Correlation-based risk management
    - Dynamic hedge recommendations
    - Emergency risk reduction protocols
    - Capital allocation optimization
    """
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.risk_level = risk_level
        
        # Initialize risk budgets based on risk level
        self.risk_budgets = self._initialize_risk_budgets(risk_level)
        
        # Portfolio limits by risk level
        self.portfolio_limits = {
            RiskLevel.CONSERVATIVE: {
                'max_portfolio_delta': 0.20,
                'max_portfolio_vega': 40.0,
                'min_portfolio_vega': -10.0,
                'max_correlation': 0.6,
                'max_sector_allocation': 0.25,
                'max_single_position': 0.15
            },
            RiskLevel.MODERATE: {
                'max_portfolio_delta': 0.30,
                'max_portfolio_vega': 60.0,
                'min_portfolio_vega': -20.0,
                'max_correlation': 0.7,
                'max_sector_allocation': 0.30,
                'max_single_position': 0.20
            },
            RiskLevel.AGGRESSIVE: {
                'max_portfolio_delta': 0.50,
                'max_portfolio_vega': 100.0,
                'min_portfolio_vega': -40.0,
                'max_correlation': 0.8,
                'max_sector_allocation': 0.40,
                'max_single_position': 0.25
            }
        }
    
    @handle_errors(operation_name="assess_portfolio_risk")
    def assess_current_portfolio_risk(
        self,
        trades: List[TradeCandidate],
        nav: Decimal,
        market_data: Optional[Dict[str, Any]] = None
    ) -> PortfolioRiskMetrics:
        """
        Assess current portfolio risk across all dimensions.
        
        Args:
            trades: Current portfolio trades
            nav: Portfolio net asset value
            market_data: Current market data for calculations
            
        Returns:
            Comprehensive portfolio risk assessment
        """
        self.logger.info(f"Assessing portfolio risk for {len(trades)} positions")
        
        # Calculate portfolio Greeks
        portfolio_greeks = self._calculate_portfolio_greeks(trades, nav)
        
        # Calculate Value at Risk
        total_var = self._calculate_portfolio_var(trades, nav)
        
        # Calculate total exposure
        total_exposure = sum(
            trade.strategy.max_loss or Decimal('0') for trade in trades
        )
        
        # Calculate leverage ratio
        leverage_ratio = float(total_exposure / nav) if nav > 0 else 0.0
        
        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(trades)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk(trades, nav)
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(trades)
        
        # Calculate sector exposure
        sector_exposure = self._calculate_sector_exposure(trades)
        sector_risk_scores = self._calculate_sector_risk_scores(sector_exposure, nav)
        
        # Calculate strategy allocation
        strategy_allocation = self._calculate_strategy_allocation(trades)
        
        return PortfolioRiskMetrics(
            total_value_at_risk=total_var,
            total_exposure=total_exposure,
            leverage_ratio=leverage_ratio,
            diversification_score=diversification_score,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            net_delta=portfolio_greeks['delta'],
            net_gamma=portfolio_greeks['gamma'],
            net_theta=portfolio_greeks['theta'],
            net_vega=portfolio_greeks['vega'],
            net_rho=portfolio_greeks['rho'],
            sector_exposure=sector_exposure,
            sector_risk_scores=sector_risk_scores,
            strategy_allocation=strategy_allocation
        )
    
    @handle_errors(operation_name="calculate_position_size")
    def calculate_optimal_position_size(
        self,
        trade_candidate: TradeCandidate,
        current_trades: List[TradeCandidate],
        nav: Decimal,
        available_capital: Decimal,
        sizing_method: PositionSizeMethod = PositionSizeMethod.KELLY_CRITERION
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size based on risk management principles.
        
        Args:
            trade_candidate: Trade to size
            current_trades: Current portfolio positions
            nav: Portfolio NAV
            available_capital: Available capital
            sizing_method: Sizing methodology to use
            
        Returns:
            Position sizing recommendation
        """
        strategy = trade_candidate.strategy
        
        if sizing_method == PositionSizeMethod.KELLY_CRITERION:
            return self._kelly_criterion_sizing(trade_candidate, nav)
        elif sizing_method == PositionSizeMethod.FIXED_DOLLAR:
            return self._fixed_dollar_sizing(trade_candidate, nav)
        elif sizing_method == PositionSizeMethod.PERCENT_OF_NAV:
            return self._percent_nav_sizing(trade_candidate, nav)
        elif sizing_method == PositionSizeMethod.RISK_PARITY:
            return self._risk_parity_sizing(trade_candidate, current_trades, nav)
        else:  # VOLATILITY_ADJUSTED
            return self._volatility_adjusted_sizing(trade_candidate, nav)
    
    @handle_errors(operation_name="check_portfolio_limits")
    def check_portfolio_limits_compliance(
        self,
        new_trade: TradeCandidate,
        current_trades: List[TradeCandidate],
        nav: Decimal
    ) -> List[ConstraintViolation]:
        """
        Check if adding new trade would violate portfolio limits.
        
        Args:
            new_trade: Proposed new trade
            current_trades: Current portfolio trades
            nav: Portfolio NAV
            
        Returns:
            List of constraint violations
        """
        violations = []
        limits = self.portfolio_limits[self.risk_level]
        
        # Project portfolio state with new trade
        projected_trades = current_trades + [new_trade]
        projected_greeks = self._calculate_portfolio_greeks(projected_trades, nav)
        
        # Check delta limits
        nav_factor = float(nav) / 100000
        max_delta = limits['max_portfolio_delta'] * nav_factor
        
        if abs(projected_greeks['delta']) > max_delta:
            violations.append(ConstraintViolation(
                constraint_name="portfolio_delta_limit",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Portfolio delta limit exceeded: {projected_greeks['delta']:.2f}",
                current_value=projected_greeks['delta'],
                required_value=f"≤{max_delta:.2f}",
                recommendation="Hedge delta exposure or reduce position size"
            ))
        
        # Check vega limits
        max_vega = limits['max_portfolio_vega'] * nav_factor
        min_vega = limits['min_portfolio_vega'] * nav_factor
        
        if projected_greeks['vega'] > max_vega:
            violations.append(ConstraintViolation(
                constraint_name="portfolio_vega_limit",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Portfolio vega too high: {projected_greeks['vega']:.2f}",
                current_value=projected_greeks['vega'],
                required_value=f"≤{max_vega:.2f}",
                recommendation="Reduce long vega exposure"
            ))
        elif projected_greeks['vega'] < min_vega:
            violations.append(ConstraintViolation(
                constraint_name="portfolio_vega_limit",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Portfolio vega too low: {projected_greeks['vega']:.2f}",
                current_value=projected_greeks['vega'],
                required_value=f"≥{min_vega:.2f}",
                recommendation="Add long vega positions"
            ))
        
        # Check sector concentration
        sector_exposure = self._calculate_sector_exposure(projected_trades)
        max_sector_allocation = limits['max_sector_allocation']
        
        for sector, exposure in sector_exposure.items():
            sector_percentage = float(exposure) / float(nav)
            if sector_percentage > max_sector_allocation:
                violations.append(ConstraintViolation(
                    constraint_name="sector_concentration",
                    violation_type=ConstraintType.PORTFOLIO_LIMIT,
                    severity=ConstraintSeverity.CRITICAL,
                    message=f"Sector concentration too high: {sector} at {sector_percentage:.1%}",
                    current_value=f"{sector_percentage:.1%}",
                    required_value=f"≤{max_sector_allocation:.1%}",
                    recommendation=f"Reduce {sector} exposure or diversify"
                ))
        
        # Check single position concentration
        max_position_size = limits['max_single_position']
        new_position_size = float(new_trade.strategy.max_loss or 0) / float(nav)
        
        if new_position_size > max_position_size:
            violations.append(ConstraintViolation(
                constraint_name="position_concentration",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Position too large: {new_position_size:.1%} of NAV",
                current_value=f"{new_position_size:.1%}",
                required_value=f"≤{max_position_size:.1%}",
                recommendation="Reduce position size"
            ))
        
        return violations
    
    @handle_errors(operation_name="generate_portfolio_actions")
    def generate_portfolio_rebalancing_actions(
        self,
        current_trades: List[TradeCandidate],
        portfolio_metrics: PortfolioRiskMetrics,
        nav: Decimal
    ) -> List[PortfolioAction]:
        """
        Generate portfolio rebalancing and risk management actions.
        
        Args:
            current_trades: Current portfolio trades
            portfolio_metrics: Current portfolio risk metrics
            nav: Portfolio NAV
            
        Returns:
            List of recommended portfolio actions
        """
        actions = []
        limits = self.portfolio_limits[self.risk_level]
        
        # Check for excessive delta exposure
        nav_factor = float(nav) / 100000
        max_delta = limits['max_portfolio_delta'] * nav_factor
        
        if abs(portfolio_metrics.net_delta) > max_delta * 0.8:  # 80% threshold
            action = PortfolioAction(
                action_type="hedge",
                priority="high" if abs(portfolio_metrics.net_delta) > max_delta else "medium",
                description=f"Delta exposure at {portfolio_metrics.net_delta:.2f}, approaching limit of ±{max_delta:.2f}",
                recommended_adjustments={
                    "hedge_type": "delta_hedge",
                    "target_delta": 0.0,
                    "hedge_amount": abs(portfolio_metrics.net_delta)
                }
            )
            actions.append(action)
        
        # Check for excessive vega exposure
        max_vega = limits['max_portfolio_vega'] * nav_factor
        min_vega = limits['min_portfolio_vega'] * nav_factor
        
        if portfolio_metrics.net_vega > max_vega * 0.8:
            actions.append(PortfolioAction(
                action_type="reduce",
                priority="medium",
                description=f"Vega exposure at {portfolio_metrics.net_vega:.2f}, approaching limit of {max_vega:.2f}",
                recommended_adjustments={
                    "action": "reduce_long_vega",
                    "target_reduction": portfolio_metrics.net_vega - max_vega * 0.6
                }
            ))
        elif portfolio_metrics.net_vega < min_vega * 0.8:
            actions.append(PortfolioAction(
                action_type="rebalance",
                priority="medium",
                description=f"Vega exposure at {portfolio_metrics.net_vega:.2f}, below comfortable range",
                recommended_adjustments={
                    "action": "add_long_vega",
                    "target_increase": min_vega * 0.6 - portfolio_metrics.net_vega
                }
            ))
        
        # Check for sector over-concentration
        max_sector_allocation = limits['max_sector_allocation']
        for sector, risk_score in portfolio_metrics.sector_risk_scores.items():
            if risk_score > max_sector_allocation * 0.8:
                actions.append(PortfolioAction(
                    action_type="rebalance",
                    priority="medium",
                    description=f"Over-concentration in {sector}: {risk_score:.1%}",
                    affected_positions=[
                        trade.strategy.underlying for trade in current_trades
                        if GICS_SECTORS.get(trade.strategy.underlying) == sector
                    ],
                    recommended_adjustments={
                        "action": "reduce_sector_exposure",
                        "sector": sector,
                        "target_allocation": max_sector_allocation * 0.6
                    }
                ))
        
        # Check overall concentration risk
        if portfolio_metrics.concentration_risk > 0.8:
            actions.append(PortfolioAction(
                action_type="rebalance",
                priority="high",
                description=f"High concentration risk: {portfolio_metrics.concentration_risk:.1%}",
                recommended_adjustments={
                    "action": "diversify_holdings",
                    "target_concentration": 0.6
                }
            ))
        
        # Check correlation risk
        if portfolio_metrics.correlation_risk > limits['max_correlation']:
            actions.append(PortfolioAction(
                action_type="rebalance",
                priority="medium",
                description=f"High correlation risk: {portfolio_metrics.correlation_risk:.2f}",
                recommended_adjustments={
                    "action": "reduce_correlated_positions",
                    "max_correlation": limits['max_correlation']
                }
            ))
        
        return actions
    
    def _initialize_risk_budgets(self, risk_level: RiskLevel) -> RiskBudget:
        """Initialize risk budgets based on risk level."""
        
        if risk_level == RiskLevel.CONSERVATIVE:
            return RiskBudget(
                total_risk_budget=Decimal('5000'),     # 5% of 100k NAV
                per_trade_limit=Decimal('300'),        # $300 max per trade
                sector_limit=Decimal('1500'),          # $1500 max per sector
                strategy_type_limit=Decimal('2000'),   # $2000 max per strategy type
                correlation_limit=0.6,
                concentration_limit=0.15
            )
        elif risk_level == RiskLevel.MODERATE:
            return RiskBudget(
                total_risk_budget=Decimal('10000'),    # 10% of 100k NAV
                per_trade_limit=Decimal('500'),        # $500 max per trade
                sector_limit=Decimal('3000'),          # $3000 max per sector
                strategy_type_limit=Decimal('4000'),   # $4000 max per strategy type
                correlation_limit=0.7,
                concentration_limit=0.20
            )
        else:  # AGGRESSIVE
            return RiskBudget(
                total_risk_budget=Decimal('20000'),    # 20% of 100k NAV
                per_trade_limit=Decimal('1000'),       # $1000 max per trade
                sector_limit=Decimal('6000'),          # $6000 max per sector
                strategy_type_limit=Decimal('8000'),   # $8000 max per strategy type
                correlation_limit=0.8,
                concentration_limit=0.25
            )
    
    def _calculate_portfolio_greeks(
        self,
        trades: List[TradeCandidate],
        nav: Decimal
    ) -> Dict[str, float]:
        """Calculate portfolio-level Greeks."""
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        
        for trade in trades:
            for leg in trade.strategy.legs:
                multiplier = 100  # Options multiplier
                position_sign = 1 if leg.direction == TradeDirection.BUY else -1
                
                if leg.option.greeks:
                    total_delta += float(leg.option.greeks.delta or 0) * leg.quantity * multiplier * position_sign
                    total_gamma += float(leg.option.greeks.gamma or 0) * leg.quantity * multiplier * position_sign
                    total_theta += float(leg.option.greeks.theta or 0) * leg.quantity * multiplier * position_sign
                    total_vega += float(leg.option.greeks.vega or 0) * leg.quantity * multiplier * position_sign
                    total_rho += float(leg.option.greeks.rho or 0) * leg.quantity * multiplier * position_sign
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }
    
    def _calculate_portfolio_var(
        self,
        trades: List[TradeCandidate],
        nav: Decimal,
        confidence_level: float = 0.95
    ) -> Decimal:
        """Calculate portfolio Value at Risk."""
        
        # Simplified VaR calculation based on max losses
        individual_vars = []
        
        for trade in trades:
            if trade.strategy.max_loss:
                individual_vars.append(float(trade.strategy.max_loss))
            else:
                individual_vars.append(0.0)
        
        if not individual_vars:
            return Decimal('0')
        
        # Conservative aggregation (sum of individual VaRs)
        # In practice, would use correlation matrix for more accurate calculation
        total_var = sum(individual_vars)
        
        return Decimal(str(total_var))
    
    def _calculate_diversification_score(self, trades: List[TradeCandidate]) -> float:
        """Calculate portfolio diversification score (0-1)."""
        
        if len(trades) <= 1:
            return 0.0
        
        # Count unique sectors
        sectors = set()
        strategy_types = set()
        underlyings = set()
        
        for trade in trades:
            sectors.add(GICS_SECTORS.get(trade.strategy.underlying, "Unknown"))
            strategy_types.add(trade.strategy.strategy_type.value)
            underlyings.add(trade.strategy.underlying)
        
        # Calculate diversification components
        sector_diversity = min(1.0, len(sectors) / 5)      # Ideal: 5+ sectors
        strategy_diversity = min(1.0, len(strategy_types) / 3)  # Ideal: 3+ strategy types
        underlying_diversity = min(1.0, len(underlyings) / 8)   # Ideal: 8+ underlyings
        
        # Weighted average
        diversification_score = (
            sector_diversity * 0.4 +
            strategy_diversity * 0.3 +
            underlying_diversity * 0.3
        )
        
        return diversification_score
    
    def _calculate_concentration_risk(
        self,
        trades: List[TradeCandidate],
        nav: Decimal
    ) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index."""
        
        if not trades:
            return 0.0
        
        # Calculate position weights
        position_weights = []
        total_exposure = Decimal('0')
        
        for trade in trades:
            exposure = trade.strategy.max_loss or Decimal('0')
            total_exposure += exposure
        
        if total_exposure == 0:
            return 0.0
        
        for trade in trades:
            exposure = trade.strategy.max_loss or Decimal('0')
            weight = float(exposure / total_exposure)
            position_weights.append(weight)
        
        # Calculate HHI
        hhi = sum(weight ** 2 for weight in position_weights)
        
        return hhi
    
    def _calculate_correlation_risk(self, trades: List[TradeCandidate]) -> float:
        """Calculate average correlation risk between positions."""
        
        if len(trades) < 2:
            return 0.0
        
        # Simplified correlation based on sector and strategy type
        correlations = []
        
        for i, trade1 in enumerate(trades):
            for j, trade2 in enumerate(trades[i+1:], i+1):
                sector1 = GICS_SECTORS.get(trade1.strategy.underlying, "Unknown")
                sector2 = GICS_SECTORS.get(trade2.strategy.underlying, "Unknown")
                
                strategy1 = trade1.strategy.strategy_type.value
                strategy2 = trade2.strategy.strategy_type.value
                
                # High correlation if same sector
                if sector1 == sector2:
                    correlation = 0.8
                # Medium correlation if same strategy type
                elif strategy1 == strategy2:
                    correlation = 0.5
                # Low correlation otherwise
                else:
                    correlation = 0.2
                
                correlations.append(correlation)
        
        return statistics.mean(correlations) if correlations else 0.0
    
    def _calculate_sector_exposure(self, trades: List[TradeCandidate]) -> Dict[str, Decimal]:
        """Calculate exposure by GICS sector."""
        
        sector_exposure = {}
        
        for trade in trades:
            sector = GICS_SECTORS.get(trade.strategy.underlying, "Unknown")
            exposure = trade.strategy.max_loss or Decimal('0')
            
            if sector in sector_exposure:
                sector_exposure[sector] += exposure
            else:
                sector_exposure[sector] = exposure
        
        return sector_exposure
    
    def _calculate_sector_risk_scores(
        self,
        sector_exposure: Dict[str, Decimal],
        nav: Decimal
    ) -> Dict[str, float]:
        """Calculate risk scores by sector as percentage of NAV."""
        
        sector_risk_scores = {}
        
        for sector, exposure in sector_exposure.items():
            risk_score = float(exposure / nav) if nav > 0 else 0.0
            sector_risk_scores[sector] = risk_score
        
        return sector_risk_scores
    
    def _calculate_strategy_allocation(self, trades: List[TradeCandidate]) -> Dict[str, Decimal]:
        """Calculate allocation by strategy type."""
        
        strategy_allocation = {}
        
        for trade in trades:
            strategy_type = trade.strategy.strategy_type.value
            exposure = trade.strategy.max_loss or Decimal('0')
            
            if strategy_type in strategy_allocation:
                strategy_allocation[strategy_type] += exposure
            else:
                strategy_allocation[strategy_type] = exposure
        
        return strategy_allocation
    
    def _kelly_criterion_sizing(
        self,
        trade_candidate: TradeCandidate,
        nav: Decimal
    ) -> PositionSizeRecommendation:
        """Calculate position size using Kelly Criterion."""
        
        strategy = trade_candidate.strategy
        
        if not all([strategy.probability_of_profit, strategy.max_profit, strategy.max_loss]):
            return PositionSizeRecommendation(
                recommended_contracts=1,
                max_contracts=1,
                sizing_method=PositionSizeMethod.KELLY_CRITERION,
                risk_amount=strategy.max_loss or Decimal('500'),
                justification="Insufficient data for Kelly calculation, using minimum size",
                warnings=["Missing probability or profit/loss data"]
            )
        
        # Kelly fraction = (bp - q) / b
        p = strategy.probability_of_profit  # Probability of profit
        b = float(strategy.max_profit) / float(strategy.max_loss)  # Odds ratio
        q = 1 - p  # Probability of loss
        
        kelly_fraction = (b * p - q) / b
        
        # Conservative sizing (use 1/4 Kelly for risk management)
        conservative_fraction = min(0.25 * kelly_fraction, 0.02)  # Max 2% of NAV
        conservative_fraction = max(0, conservative_fraction)  # No negative sizing
        
        # Calculate number of contracts
        max_risk_amount = float(nav) * conservative_fraction
        max_loss_per_contract = float(strategy.max_loss)
        
        if max_loss_per_contract <= 0:
            recommended_contracts = 1
        else:
            recommended_contracts = int(max_risk_amount / max_loss_per_contract)
        
        # Apply practical limits
        recommended_contracts = max(1, min(10, recommended_contracts))
        
        # Calculate actual risk amount
        actual_risk = Decimal(str(recommended_contracts * max_loss_per_contract))
        
        warnings = []
        if kelly_fraction < 0:
            warnings.append("Negative Kelly fraction suggests unfavorable trade")
        if conservative_fraction >= 0.02:
            warnings.append("Position size capped at 2% of NAV")
        
        return PositionSizeRecommendation(
            recommended_contracts=recommended_contracts,
            max_contracts=min(10, int(float(nav) * 0.05 / max_loss_per_contract)),  # 5% NAV max
            sizing_method=PositionSizeMethod.KELLY_CRITERION,
            risk_amount=actual_risk,
            justification=f"Kelly fraction: {kelly_fraction:.3f}, Conservative fraction: {conservative_fraction:.3f}",
            warnings=warnings
        )
    
    def _fixed_dollar_sizing(
        self,
        trade_candidate: TradeCandidate,
        nav: Decimal
    ) -> PositionSizeRecommendation:
        """Calculate position size using fixed dollar risk."""
        
        strategy = trade_candidate.strategy
        max_loss_per_contract = float(strategy.max_loss or 500)
        
        # Use risk budget per trade limit
        target_risk = float(self.risk_budgets.per_trade_limit)
        contracts = int(target_risk / max_loss_per_contract)
        contracts = max(1, min(10, contracts))
        
        return PositionSizeRecommendation(
            recommended_contracts=contracts,
            max_contracts=10,
            sizing_method=PositionSizeMethod.FIXED_DOLLAR,
            risk_amount=Decimal(str(contracts * max_loss_per_contract)),
            justification=f"Fixed risk target: ${target_risk:.0f} per trade"
        )
    
    def _percent_nav_sizing(
        self,
        trade_candidate: TradeCandidate,
        nav: Decimal
    ) -> PositionSizeRecommendation:
        """Calculate position size as percentage of NAV."""
        
        strategy = trade_candidate.strategy
        max_loss_per_contract = float(strategy.max_loss or 500)
        
        # Risk 1% of NAV per trade
        target_risk = float(nav) * 0.01
        contracts = int(target_risk / max_loss_per_contract)
        contracts = max(1, min(10, contracts))
        
        return PositionSizeRecommendation(
            recommended_contracts=contracts,
            max_contracts=10,
            sizing_method=PositionSizeMethod.PERCENT_OF_NAV,
            risk_amount=Decimal(str(contracts * max_loss_per_contract)),
            justification="1% of NAV risk target"
        )
    
    def _risk_parity_sizing(
        self,
        trade_candidate: TradeCandidate,
        current_trades: List[TradeCandidate],
        nav: Decimal
    ) -> PositionSizeRecommendation:
        """Calculate position size using risk parity approach."""
        
        # Target equal risk contribution from each position
        num_positions = len(current_trades) + 1  # Including new trade
        target_risk_per_position = float(self.risk_budgets.total_risk_budget) / num_positions
        
        strategy = trade_candidate.strategy
        max_loss_per_contract = float(strategy.max_loss or 500)
        
        contracts = int(target_risk_per_position / max_loss_per_contract)
        contracts = max(1, min(10, contracts))
        
        return PositionSizeRecommendation(
            recommended_contracts=contracts,
            max_contracts=10,
            sizing_method=PositionSizeMethod.RISK_PARITY,
            risk_amount=Decimal(str(contracts * max_loss_per_contract)),
            justification=f"Risk parity: ${target_risk_per_position:.0f} per position"
        )
    
    def _volatility_adjusted_sizing(
        self,
        trade_candidate: TradeCandidate,
        nav: Decimal
    ) -> PositionSizeRecommendation:
        """Calculate position size adjusted for volatility."""
        
        # Placeholder for volatility-adjusted sizing
        # Would typically use historical volatility or VIX level
        base_size = self._fixed_dollar_sizing(trade_candidate, nav)
        
        # Adjust based on market volatility (simplified)
        # In high volatility, reduce size; in low volatility, increase size
        volatility_multiplier = 1.0  # Would calculate from market data
        
        adjusted_contracts = int(base_size.recommended_contracts * volatility_multiplier)
        adjusted_contracts = max(1, min(10, adjusted_contracts))
        
        return PositionSizeRecommendation(
            recommended_contracts=adjusted_contracts,
            max_contracts=10,
            sizing_method=PositionSizeMethod.VOLATILITY_ADJUSTED,
            risk_amount=Decimal(str(adjusted_contracts * float(trade_candidate.strategy.max_loss or 500))),
            justification=f"Volatility adjustment: {volatility_multiplier:.2f}x base size"
        )