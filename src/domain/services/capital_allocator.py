"""
Capital allocation framework for intelligent position sizing and portfolio optimization.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
import logging

from ...data.models.options import OptionQuote, OptionType
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, TradeLeg, StrategyType
)
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, CalculationError
)

from .portfolio_risk_controller import RiskLevel, PositionSizeMethod
from .constraint_engine import GICS_SECTORS


class AllocationMethod(Enum):
    """Capital allocation methodologies."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY_OPTIMAL = "kelly_optimal"
    RISK_PARITY = "risk_parity"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    CONVICTION_WEIGHTED = "conviction_weighted"
    MODERN_PORTFOLIO_THEORY = "modern_portfolio_theory"


class ConvictionLevel(Enum):
    """Conviction levels for trades."""
    LOW = "low"           # 0.5x base allocation
    MEDIUM = "medium"     # 1.0x base allocation
    HIGH = "high"         # 1.5x base allocation
    VERY_HIGH = "very_high"  # 2.0x base allocation


@dataclass
class CapitalConstraints:
    """Capital allocation constraints."""
    total_capital: Decimal
    available_capital: Decimal
    reserved_capital: Decimal
    emergency_reserve_pct: float = 0.10  # 10% emergency reserve
    
    # Allocation limits
    max_per_trade_pct: float = 0.05      # 5% max per trade
    max_per_sector_pct: float = 0.25     # 25% max per sector
    max_per_strategy_pct: float = 0.40   # 40% max per strategy type
    
    # Risk limits
    max_total_risk_pct: float = 0.15     # 15% max total portfolio risk
    max_leverage_ratio: float = 1.5      # 1.5x max leverage


@dataclass
class AllocationTarget:
    """Target allocation for a strategy type or asset."""
    strategy_type: str
    target_allocation_pct: float
    current_allocation_pct: float
    min_allocation_pct: float
    max_allocation_pct: float
    priority_score: float


@dataclass
class PositionAllocation:
    """Recommended position allocation."""
    trade_candidate: TradeCandidate
    allocated_capital: Decimal
    position_size_contracts: int
    allocation_pct: float
    conviction_level: ConvictionLevel
    allocation_method: AllocationMethod
    justification: str
    risk_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation recommendation."""
    total_allocated_capital: Decimal
    available_remaining: Decimal
    position_allocations: List[PositionAllocation]
    allocation_summary: Dict[str, float]
    risk_summary: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


class CapitalAllocator:
    """
    Intelligent capital allocation framework implementing multiple methodologies.
    
    Features:
    - Kelly criterion application for optimal sizing
    - Risk parity across different strategy types and sectors
    - Available margin calculation and optimization
    - Liquidity requirements for entry/exit planning
    - Stress testing for capital adequacy
    - Position scaling based on conviction levels
    - Modern Portfolio Theory optimization
    - Dynamic rebalancing recommendations
    """
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.risk_level = risk_level
        
        # Initialize target allocations by strategy type
        self.strategy_targets = self._initialize_strategy_targets(risk_level)
        
        # Initialize sector targets
        self.sector_targets = self._initialize_sector_targets()
        
        # Risk-free rate for calculations
        self.risk_free_rate = 0.05
    
    @handle_errors(operation_name="allocate_capital")
    def allocate_capital_to_trades(
        self,
        trade_candidates: List[TradeCandidate],
        current_trades: List[TradeCandidate],
        capital_constraints: CapitalConstraints,
        allocation_method: AllocationMethod = AllocationMethod.KELLY_OPTIMAL,
        conviction_scores: Optional[Dict[str, ConvictionLevel]] = None
    ) -> PortfolioAllocation:
        """
        Allocate capital across trade candidates using specified methodology.
        
        Args:
            trade_candidates: List of viable trade candidates
            current_trades: Current portfolio positions
            capital_constraints: Capital constraints and limits
            allocation_method: Allocation methodology to use
            conviction_scores: Optional conviction levels for trades
            
        Returns:
            Complete portfolio allocation recommendation
        """
        self.logger.info(f"Allocating capital to {len(trade_candidates)} candidates using {allocation_method.value}")
        
        if not trade_candidates:
            return PortfolioAllocation(
                total_allocated_capital=Decimal('0'),
                available_remaining=capital_constraints.available_capital,
                position_allocations=[],
                allocation_summary={},
                risk_summary={},
                warnings=["No trade candidates provided"]
            )
        
        # Calculate current portfolio state
        current_allocations = self._calculate_current_allocations(current_trades, capital_constraints)
        
        # Apply allocation method
        if allocation_method == AllocationMethod.KELLY_OPTIMAL:
            allocations = self._kelly_optimal_allocation(
                trade_candidates, capital_constraints, current_allocations, conviction_scores
            )
        elif allocation_method == AllocationMethod.RISK_PARITY:
            allocations = self._risk_parity_allocation(
                trade_candidates, capital_constraints, current_allocations
            )
        elif allocation_method == AllocationMethod.EQUAL_WEIGHT:
            allocations = self._equal_weight_allocation(
                trade_candidates, capital_constraints, current_allocations
            )
        elif allocation_method == AllocationMethod.VOLATILITY_WEIGHTED:
            allocations = self._volatility_weighted_allocation(
                trade_candidates, capital_constraints, current_allocations
            )
        elif allocation_method == AllocationMethod.CONVICTION_WEIGHTED:
            allocations = self._conviction_weighted_allocation(
                trade_candidates, capital_constraints, current_allocations, conviction_scores
            )
        else:  # MODERN_PORTFOLIO_THEORY
            allocations = self._mpt_allocation(
                trade_candidates, capital_constraints, current_allocations
            )
        
        # Validate allocations against constraints
        validated_allocations = self._validate_allocations(allocations, capital_constraints, current_allocations)
        
        # Calculate summary metrics
        allocation_summary = self._calculate_allocation_summary(validated_allocations)
        risk_summary = self._calculate_risk_summary(validated_allocations)
        
        # Calculate totals
        total_allocated = sum(alloc.allocated_capital for alloc in validated_allocations)
        remaining_capital = capital_constraints.available_capital - total_allocated
        
        return PortfolioAllocation(
            total_allocated_capital=total_allocated,
            available_remaining=remaining_capital,
            position_allocations=validated_allocations,
            allocation_summary=allocation_summary,
            risk_summary=risk_summary
        )
    
    @handle_errors(operation_name="calculate_kelly_position_size")
    def calculate_kelly_position_size(
        self,
        trade_candidate: TradeCandidate,
        available_capital: Decimal,
        conviction_multiplier: float = 1.0,
        max_kelly_fraction: float = 0.25
    ) -> Tuple[int, Decimal, str]:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            trade_candidate: Trade to size
            available_capital: Available capital
            conviction_multiplier: Conviction adjustment (0.5 - 2.0)
            max_kelly_fraction: Maximum Kelly fraction to use
            
        Returns:
            Tuple of (contracts, allocated_capital, justification)
        """
        strategy = trade_candidate.strategy
        
        # Validate required data
        if not all([strategy.probability_of_profit, strategy.max_profit, strategy.max_loss]):
            return (
                1,
                strategy.max_loss or Decimal('500'),
                "Insufficient data for Kelly calculation - using minimum size"
            )
        
        # Kelly Criterion: f = (bp - q) / b
        p = strategy.probability_of_profit  # Probability of profit
        q = 1 - p  # Probability of loss
        b = float(strategy.max_profit) / float(strategy.max_loss)  # Odds ratio
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conviction multiplier and safety limits
        adjusted_kelly = kelly_fraction * conviction_multiplier
        safe_kelly = min(adjusted_kelly, max_kelly_fraction)
        safe_kelly = max(0, safe_kelly)  # No negative sizing
        
        # Calculate capital allocation
        capital_to_risk = float(available_capital) * safe_kelly
        max_loss_per_contract = float(strategy.max_loss)
        
        if max_loss_per_contract <= 0:
            contracts = 1
        else:
            contracts = int(capital_to_risk / max_loss_per_contract)
        
        # Apply practical limits
        contracts = max(1, min(10, contracts))
        allocated_capital = Decimal(str(contracts * max_loss_per_contract))
        
        justification = (
            f"Kelly fraction: {kelly_fraction:.3f}, "
            f"Adjusted: {safe_kelly:.3f}, "
            f"Capital at risk: ${capital_to_risk:.0f}"
        )
        
        return contracts, allocated_capital, justification
    
    @handle_errors(operation_name="calculate_optimal_portfolio")
    def calculate_optimal_portfolio_weights(
        self,
        trade_candidates: List[TradeCandidate],
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using Modern Portfolio Theory.
        
        Args:
            trade_candidates: List of trade candidates
            correlation_matrix: Correlation matrix between trades
            
        Returns:
            Dictionary of optimal weights by trade ID
        """
        if len(trade_candidates) == 1:
            return {f"trade_0": 1.0}
        
        # Extract expected returns and risks
        expected_returns = []
        risks = []
        
        for i, candidate in enumerate(trade_candidates):
            if candidate.expected_return and candidate.strategy.max_loss:
                expected_return = float(candidate.expected_return) / float(candidate.strategy.max_loss)
                risk = 1.0  # Simplified - all trades have equal risk weighting
            else:
                expected_return = 0.05  # Default 5% return
                risk = 1.0
            
            expected_returns.append(expected_return)
            risks.append(risk)
        
        # Equal weight if no correlation data
        if not correlation_matrix:
            equal_weight = 1.0 / len(trade_candidates)
            return {f"trade_{i}": equal_weight for i in range(len(trade_candidates))}
        
        # Simplified MPT optimization (equal weight for now)
        # In practice, would solve quadratic optimization problem
        equal_weight = 1.0 / len(trade_candidates)
        return {f"trade_{i}": equal_weight for i in range(len(trade_candidates))}
    
    def _initialize_strategy_targets(self, risk_level: RiskLevel) -> Dict[str, AllocationTarget]:
        """Initialize target allocations by strategy type."""
        
        if risk_level == RiskLevel.CONSERVATIVE:
            targets = {
                "PUT_CREDIT_SPREAD": AllocationTarget("PUT_CREDIT_SPREAD", 0.30, 0.0, 0.20, 0.40, 0.8),
                "CALL_CREDIT_SPREAD": AllocationTarget("CALL_CREDIT_SPREAD", 0.25, 0.0, 0.15, 0.35, 0.7),
                "IRON_CONDOR": AllocationTarget("IRON_CONDOR", 0.25, 0.0, 0.15, 0.40, 0.9),
                "COVERED_CALL": AllocationTarget("COVERED_CALL", 0.15, 0.0, 0.05, 0.25, 0.6),
                "CASH_SECURED_PUT": AllocationTarget("CASH_SECURED_PUT", 0.05, 0.0, 0.0, 0.15, 0.5)
            }
        elif risk_level == RiskLevel.MODERATE:
            targets = {
                "PUT_CREDIT_SPREAD": AllocationTarget("PUT_CREDIT_SPREAD", 0.25, 0.0, 0.15, 0.35, 0.8),
                "CALL_CREDIT_SPREAD": AllocationTarget("CALL_CREDIT_SPREAD", 0.25, 0.0, 0.15, 0.35, 0.8),
                "IRON_CONDOR": AllocationTarget("IRON_CONDOR", 0.20, 0.0, 0.10, 0.35, 0.9),
                "COVERED_CALL": AllocationTarget("COVERED_CALL", 0.15, 0.0, 0.05, 0.25, 0.7),
                "CASH_SECURED_PUT": AllocationTarget("CASH_SECURED_PUT", 0.10, 0.0, 0.0, 0.20, 0.6),
                "IRON_BUTTERFLY": AllocationTarget("IRON_BUTTERFLY", 0.05, 0.0, 0.0, 0.15, 0.5)
            }
        else:  # AGGRESSIVE
            targets = {
                "PUT_CREDIT_SPREAD": AllocationTarget("PUT_CREDIT_SPREAD", 0.20, 0.0, 0.10, 0.30, 0.8),
                "CALL_CREDIT_SPREAD": AllocationTarget("CALL_CREDIT_SPREAD", 0.20, 0.0, 0.10, 0.30, 0.8),
                "IRON_CONDOR": AllocationTarget("IRON_CONDOR", 0.15, 0.0, 0.05, 0.25, 0.7),
                "COVERED_CALL": AllocationTarget("COVERED_CALL", 0.15, 0.0, 0.05, 0.25, 0.6),
                "CASH_SECURED_PUT": AllocationTarget("CASH_SECURED_PUT", 0.15, 0.0, 0.05, 0.25, 0.6),
                "IRON_BUTTERFLY": AllocationTarget("IRON_BUTTERFLY", 0.10, 0.0, 0.0, 0.20, 0.5),
                "CALENDAR_SPREAD": AllocationTarget("CALENDAR_SPREAD", 0.05, 0.0, 0.0, 0.15, 0.4)
            }
        
        return targets
    
    def _initialize_sector_targets(self) -> Dict[str, float]:
        """Initialize target sector allocations."""
        
        return {
            "Information Technology": 0.25,
            "Health Care": 0.15,
            "Financials": 0.15,
            "Consumer Discretionary": 0.12,
            "Communication Services": 0.10,
            "Industrials": 0.08,
            "Consumer Staples": 0.06,
            "Energy": 0.04,
            "Utilities": 0.03,
            "Materials": 0.02
        }
    
    def _calculate_current_allocations(
        self,
        current_trades: List[TradeCandidate],
        constraints: CapitalConstraints
    ) -> Dict[str, float]:
        """Calculate current portfolio allocations."""
        
        current_allocations = {
            'by_strategy': {},
            'by_sector': {},
            'total_allocated': 0.0
        }
        
        total_exposure = Decimal('0')
        for trade in current_trades:
            if trade.strategy.max_loss:
                total_exposure += trade.strategy.max_loss
        
        # Calculate strategy allocations
        strategy_exposure = {}
        for trade in current_trades:
            strategy_type = trade.strategy.strategy_type.value
            exposure = trade.strategy.max_loss or Decimal('0')
            
            if strategy_type in strategy_exposure:
                strategy_exposure[strategy_type] += exposure
            else:
                strategy_exposure[strategy_type] = exposure
        
        for strategy_type, exposure in strategy_exposure.items():
            allocation_pct = float(exposure / constraints.total_capital) if constraints.total_capital > 0 else 0.0
            current_allocations['by_strategy'][strategy_type] = allocation_pct
        
        # Calculate sector allocations
        sector_exposure = {}
        for trade in current_trades:
            sector = GICS_SECTORS.get(trade.strategy.underlying, "Unknown")
            exposure = trade.strategy.max_loss or Decimal('0')
            
            if sector in sector_exposure:
                sector_exposure[sector] += exposure
            else:
                sector_exposure[sector] = exposure
        
        for sector, exposure in sector_exposure.items():
            allocation_pct = float(exposure / constraints.total_capital) if constraints.total_capital > 0 else 0.0
            current_allocations['by_sector'][sector] = allocation_pct
        
        current_allocations['total_allocated'] = float(total_exposure / constraints.total_capital) if constraints.total_capital > 0 else 0.0
        
        return current_allocations
    
    def _kelly_optimal_allocation(
        self,
        candidates: List[TradeCandidate],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any],
        conviction_scores: Optional[Dict[str, ConvictionLevel]]
    ) -> List[PositionAllocation]:
        """Allocate capital using Kelly Criterion."""
        
        allocations = []
        remaining_capital = constraints.available_capital
        
        # Sort candidates by Kelly fraction (highest first)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: self._calculate_kelly_fraction(c),
            reverse=True
        )
        
        for i, candidate in enumerate(sorted_candidates):
            # Get conviction multiplier
            conviction_key = f"trade_{i}"
            conviction = conviction_scores.get(conviction_key, ConvictionLevel.MEDIUM) if conviction_scores else ConvictionLevel.MEDIUM
            conviction_multiplier = self._get_conviction_multiplier(conviction)
            
            # Calculate Kelly position size
            contracts, allocated_capital, justification = self.calculate_kelly_position_size(
                candidate, remaining_capital, conviction_multiplier
            )
            
            if allocated_capital <= remaining_capital:
                allocation_pct = float(allocated_capital / constraints.total_capital)
                
                allocations.append(PositionAllocation(
                    trade_candidate=candidate,
                    allocated_capital=allocated_capital,
                    position_size_contracts=contracts,
                    allocation_pct=allocation_pct,
                    conviction_level=conviction,
                    allocation_method=AllocationMethod.KELLY_OPTIMAL,
                    justification=justification
                ))
                
                remaining_capital -= allocated_capital
            
            # Stop if we've run out of capital
            if remaining_capital < Decimal('100'):  # Minimum remaining threshold
                break
        
        return allocations
    
    def _risk_parity_allocation(
        self,
        candidates: List[TradeCandidate],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any]
    ) -> List[PositionAllocation]:
        """Allocate capital using risk parity approach."""
        
        if not candidates:
            return []
        
        # Target equal risk contribution from each position
        target_risk_per_position = float(constraints.available_capital) * constraints.max_total_risk_pct / len(candidates)
        
        allocations = []
        
        for candidate in candidates:
            max_loss_per_contract = float(candidate.strategy.max_loss or 500)
            contracts = int(target_risk_per_position / max_loss_per_contract)
            contracts = max(1, min(10, contracts))
            
            allocated_capital = Decimal(str(contracts * max_loss_per_contract))
            allocation_pct = float(allocated_capital / constraints.total_capital)
            
            allocations.append(PositionAllocation(
                trade_candidate=candidate,
                allocated_capital=allocated_capital,
                position_size_contracts=contracts,
                allocation_pct=allocation_pct,
                conviction_level=ConvictionLevel.MEDIUM,
                allocation_method=AllocationMethod.RISK_PARITY,
                justification=f"Risk parity: ${target_risk_per_position:.0f} target risk per position"
            ))
        
        return allocations
    
    def _equal_weight_allocation(
        self,
        candidates: List[TradeCandidate],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any]
    ) -> List[PositionAllocation]:
        """Allocate capital equally across all positions."""
        
        if not candidates:
            return []
        
        capital_per_trade = constraints.available_capital / len(candidates)
        
        allocations = []
        
        for candidate in candidates:
            max_loss_per_contract = float(candidate.strategy.max_loss or 500)
            contracts = int(float(capital_per_trade) / max_loss_per_contract)
            contracts = max(1, min(10, contracts))
            
            allocated_capital = Decimal(str(contracts * max_loss_per_contract))
            allocation_pct = float(allocated_capital / constraints.total_capital)
            
            allocations.append(PositionAllocation(
                trade_candidate=candidate,
                allocated_capital=allocated_capital,
                position_size_contracts=contracts,
                allocation_pct=allocation_pct,
                conviction_level=ConvictionLevel.MEDIUM,
                allocation_method=AllocationMethod.EQUAL_WEIGHT,
                justification=f"Equal weight: ${capital_per_trade:.0f} per position"
            ))
        
        return allocations
    
    def _volatility_weighted_allocation(
        self,
        candidates: List[TradeCandidate],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any]
    ) -> List[PositionAllocation]:
        """Allocate capital inversely proportional to volatility."""
        
        # Simplified volatility weighting
        # In practice, would calculate actual volatility metrics
        return self._equal_weight_allocation(candidates, constraints, current_allocations)
    
    def _conviction_weighted_allocation(
        self,
        candidates: List[TradeCandidate],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any],
        conviction_scores: Optional[Dict[str, ConvictionLevel]]
    ) -> List[PositionAllocation]:
        """Allocate capital based on conviction levels."""
        
        if not conviction_scores:
            return self._equal_weight_allocation(candidates, constraints, current_allocations)
        
        # Calculate total conviction weight
        total_weight = 0.0
        for i in range(len(candidates)):
            conviction = conviction_scores.get(f"trade_{i}", ConvictionLevel.MEDIUM)
            weight = self._get_conviction_multiplier(conviction)
            total_weight += weight
        
        allocations = []
        
        for i, candidate in enumerate(candidates):
            conviction = conviction_scores.get(f"trade_{i}", ConvictionLevel.MEDIUM)
            conviction_weight = self._get_conviction_multiplier(conviction)
            
            # Allocate proportional to conviction
            capital_allocation = constraints.available_capital * (conviction_weight / total_weight)
            
            max_loss_per_contract = float(candidate.strategy.max_loss or 500)
            contracts = int(float(capital_allocation) / max_loss_per_contract)
            contracts = max(1, min(10, contracts))
            
            allocated_capital = Decimal(str(contracts * max_loss_per_contract))
            allocation_pct = float(allocated_capital / constraints.total_capital)
            
            allocations.append(PositionAllocation(
                trade_candidate=candidate,
                allocated_capital=allocated_capital,
                position_size_contracts=contracts,
                allocation_pct=allocation_pct,
                conviction_level=conviction,
                allocation_method=AllocationMethod.CONVICTION_WEIGHTED,
                justification=f"Conviction weight: {conviction_weight:.1f}x"
            ))
        
        return allocations
    
    def _mpt_allocation(
        self,
        candidates: List[TradeCandidate],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any]
    ) -> List[PositionAllocation]:
        """Allocate capital using Modern Portfolio Theory."""
        
        # Simplified MPT - use equal weight for now
        # In practice, would solve mean-variance optimization
        return self._equal_weight_allocation(candidates, constraints, current_allocations)
    
    def _validate_allocations(
        self,
        allocations: List[PositionAllocation],
        constraints: CapitalConstraints,
        current_allocations: Dict[str, Any]
    ) -> List[PositionAllocation]:
        """Validate and adjust allocations against constraints."""
        
        validated = []
        
        for allocation in allocations:
            # Check maximum per trade constraint
            if allocation.allocation_pct > constraints.max_per_trade_pct:
                # Scale down to maximum
                scale_factor = constraints.max_per_trade_pct / allocation.allocation_pct
                allocation.allocated_capital *= Decimal(str(scale_factor))
                allocation.position_size_contracts = int(allocation.position_size_contracts * scale_factor)
                allocation.position_size_contracts = max(1, allocation.position_size_contracts)
                allocation.allocation_pct = constraints.max_per_trade_pct
                allocation.justification += f" (scaled to {constraints.max_per_trade_pct:.1%} max)"
            
            validated.append(allocation)
        
        return validated
    
    def _calculate_allocation_summary(self, allocations: List[PositionAllocation]) -> Dict[str, float]:
        """Calculate allocation summary statistics."""
        
        summary = {
            'total_positions': len(allocations),
            'by_strategy': {},
            'by_sector': {},
            'by_conviction': {}
        }
        
        # By strategy type
        for allocation in allocations:
            strategy_type = allocation.trade_candidate.strategy.strategy_type.value
            if strategy_type in summary['by_strategy']:
                summary['by_strategy'][strategy_type] += allocation.allocation_pct
            else:
                summary['by_strategy'][strategy_type] = allocation.allocation_pct
        
        # By sector
        for allocation in allocations:
            sector = GICS_SECTORS.get(allocation.trade_candidate.strategy.underlying, "Unknown")
            if sector in summary['by_sector']:
                summary['by_sector'][sector] += allocation.allocation_pct
            else:
                summary['by_sector'][sector] = allocation.allocation_pct
        
        # By conviction
        for allocation in allocations:
            conviction = allocation.conviction_level.value
            if conviction in summary['by_conviction']:
                summary['by_conviction'][conviction] += allocation.allocation_pct
            else:
                summary['by_conviction'][conviction] = allocation.allocation_pct
        
        return summary
    
    def _calculate_risk_summary(self, allocations: List[PositionAllocation]) -> Dict[str, float]:
        """Calculate risk summary metrics."""
        
        total_risk = sum(float(alloc.allocated_capital) for alloc in allocations)
        
        risk_summary = {
            'total_capital_at_risk': total_risk,
            'average_position_size': total_risk / len(allocations) if allocations else 0,
            'largest_position_pct': max((alloc.allocation_pct for alloc in allocations), default=0),
            'portfolio_concentration': self._calculate_concentration_score(allocations)
        }
        
        return risk_summary
    
    def _calculate_kelly_fraction(self, candidate: TradeCandidate) -> float:
        """Calculate Kelly fraction for a trade candidate."""
        
        strategy = candidate.strategy
        
        if not all([strategy.probability_of_profit, strategy.max_profit, strategy.max_loss]):
            return 0.0
        
        p = strategy.probability_of_profit
        b = float(strategy.max_profit) / float(strategy.max_loss)
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        return max(0, kelly_fraction)
    
    def _get_conviction_multiplier(self, conviction: ConvictionLevel) -> float:
        """Get multiplier for conviction level."""
        
        multipliers = {
            ConvictionLevel.LOW: 0.5,
            ConvictionLevel.MEDIUM: 1.0,
            ConvictionLevel.HIGH: 1.5,
            ConvictionLevel.VERY_HIGH: 2.0
        }
        
        return multipliers.get(conviction, 1.0)
    
    def _calculate_concentration_score(self, allocations: List[PositionAllocation]) -> float:
        """Calculate portfolio concentration score using HHI."""
        
        if not allocations:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        hhi = sum(alloc.allocation_pct ** 2 for alloc in allocations)
        
        return hhi