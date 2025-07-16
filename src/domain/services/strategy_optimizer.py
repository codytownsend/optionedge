"""
Strategy parameter optimization system for enhanced strategy selection.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
import logging

from ...data.models.options import OptionsChain, OptionQuote, OptionType, Greeks
from ...data.models.market_data import TechnicalIndicators, FundamentalData, StockQuote
from ...data.models.trades import StrategyDefinition, StrategyType
from ...infrastructure.error_handling import (
    handle_errors, CalculationError, BusinessLogicError
)

from .strategy_generation_service import StrategyCandidate, StrategyGenerationConfig


class OptimizationObjective(Enum):
    """Optimization objectives for strategy selection."""
    MAXIMIZE_EXPECTED_RETURN = "maximize_expected_return"
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"  
    MAXIMIZE_PROBABILITY = "maximize_probability"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_KELLY = "maximize_kelly"
    BALANCED_APPROACH = "balanced_approach"


class VolatilityRegime(Enum):
    """Market volatility regimes for adaptive parameters."""
    LOW_VOL = "low_vol"
    NORMAL_VOL = "normal_vol"
    HIGH_VOL = "high_vol"
    EXTREME_VOL = "extreme_vol"


@dataclass
class MarketConditions:
    """Current market conditions for optimization context."""
    volatility_regime: VolatilityRegime
    trend_direction: str  # "bullish", "bearish", "sideways"
    trend_strength: float  # 0-1
    iv_rank: float  # 0-100
    earnings_proximity: Optional[int] = None  # Days to earnings
    economic_regime: Optional[str] = None


@dataclass 
class OptimizationConstraints:
    """Constraints for strategy optimization."""
    min_probability_of_profit: float = 0.65
    max_loss_per_trade: Decimal = Decimal('500')
    min_credit_to_max_loss: float = 0.33
    max_dte: int = 45
    min_dte: int = 7
    max_portfolio_delta: float = 0.30
    min_portfolio_vega: float = -0.05
    max_correlation: float = 0.7
    min_liquidity_score: float = 0.6


@dataclass
class OptimizationResult:
    """Result of strategy optimization process."""
    original_candidates: List[StrategyCandidate]
    optimized_candidates: List[StrategyCandidate]
    selected_strategy: Optional[StrategyCandidate]
    optimization_scores: Dict[str, float]
    market_conditions: MarketConditions
    optimization_rationale: str
    confidence_score: float
    alternative_strategies: List[StrategyCandidate] = field(default_factory=list)


class StrategyOptimizer:
    """
    Advanced strategy parameter optimization system.
    
    Features:
    - Strike selection using probability-weighted expected returns
    - Expiration date optimization balancing time decay and event risk
    - Position sizing calculation based on Kelly criterion and risk parity
    - Greeks-neutral strategy construction for market-neutral approaches
    - Volatility arbitrage opportunity identification
    - Market regime-aware parameter adjustments
    - Multi-objective optimization with configurable weights
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Default optimization weights for balanced approach
        self.default_weights = {
            "expected_return": 0.25,
            "probability_of_profit": 0.25,
            "sharpe_ratio": 0.20,
            "kelly_fraction": 0.15,
            "liquidity": 0.10,
            "greeks_balance": 0.05
        }
        
        # Regime-specific weight adjustments
        self.regime_weights = {
            VolatilityRegime.LOW_VOL: {
                "expected_return": 0.30,
                "probability_of_profit": 0.20,
                "kelly_fraction": 0.20
            },
            VolatilityRegime.HIGH_VOL: {
                "probability_of_profit": 0.35,
                "sharpe_ratio": 0.25,
                "expected_return": 0.15
            }
        }
    
    @handle_errors(operation_name="optimize_strategies")
    def optimize_strategy_selection(
        self,
        candidates: List[StrategyCandidate],
        market_data: Dict[str, Any],
        objective: OptimizationObjective = OptimizationObjective.BALANCED_APPROACH,
        constraints: Optional[OptimizationConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize strategy selection based on market conditions and objectives.
        
        Args:
            candidates: Strategy candidates to optimize
            market_data: Current market data context
            objective: Optimization objective
            constraints: Optimization constraints
            
        Returns:
            Optimization result with selected strategy and rationale
        """
        self.logger.info(f"Optimizing {len(candidates)} strategy candidates")
        
        if not candidates:
            raise BusinessLogicError(
                "No strategy candidates provided for optimization",
                rule_name="candidate_availability"
            )
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Analyze market conditions
        market_conditions = self._analyze_market_conditions(market_data)
        
        # Filter candidates by hard constraints
        filtered_candidates = self._apply_hard_constraints(candidates, constraints)
        
        if not filtered_candidates:
            raise BusinessLogicError(
                "No candidates passed hard constraints",
                rule_name="constraint_filtering"
            )
        
        # Optimize parameters for remaining candidates
        optimized_candidates = self._optimize_candidate_parameters(
            filtered_candidates, market_conditions, constraints
        )
        
        # Apply objective-specific optimization
        scored_candidates = self._apply_optimization_objective(
            optimized_candidates, objective, market_conditions
        )
        
        # Select best strategy
        selected_strategy = self._select_optimal_strategy(
            scored_candidates, market_conditions, objective
        )
        
        # Generate alternatives
        alternatives = scored_candidates[1:4] if len(scored_candidates) > 1 else []
        
        # Create optimization result
        result = OptimizationResult(
            original_candidates=candidates,
            optimized_candidates=optimized_candidates,
            selected_strategy=selected_strategy,
            optimization_scores=self._calculate_optimization_scores(scored_candidates),
            market_conditions=market_conditions,
            optimization_rationale=self._generate_rationale(selected_strategy, market_conditions, objective),
            confidence_score=self._calculate_confidence_score(selected_strategy, market_conditions),
            alternative_strategies=alternatives
        )
        
        self.logger.info(
            f"Optimization completed. Selected: {selected_strategy.strategy.strategy_type.value if selected_strategy else 'None'}"
        )
        
        return result
    
    def optimize_strike_selection(
        self,
        options_chain: OptionsChain,
        strategy_type: StrategyType,
        market_conditions: MarketConditions,
        constraints: OptimizationConstraints
    ) -> Dict[str, Any]:
        """
        Optimize strike selection using probability-weighted expected returns.
        
        Args:
            options_chain: Available options
            strategy_type: Type of strategy to optimize
            market_conditions: Current market conditions
            constraints: Optimization constraints
            
        Returns:
            Optimal strike configuration
        """
        optimal_strikes = {}
        
        for expiration in options_chain.get_expirations():
            dte = (expiration - date.today()).days
            if not (constraints.min_dte <= dte <= constraints.max_dte):
                continue
            
            strikes = options_chain.get_strikes(expiration)
            if not strikes:
                continue
            
            # Calculate probability-weighted returns for each strike combination
            strike_scores = self._calculate_strike_probabilities(
                options_chain, expiration, strikes, strategy_type, market_conditions
            )
            
            if strike_scores:
                optimal_strikes[expiration] = max(strike_scores.items(), key=lambda x: x[1])
        
        return optimal_strikes
    
    def optimize_expiration_selection(
        self,
        options_chain: OptionsChain,
        strategy_type: StrategyType,
        market_conditions: MarketConditions,
        constraints: OptimizationConstraints
    ) -> date:
        """
        Optimize expiration date balancing time decay and event risk.
        
        Args:
            options_chain: Available options
            strategy_type: Type of strategy
            market_conditions: Current market conditions
            constraints: Optimization constraints
            
        Returns:
            Optimal expiration date
        """
        expiration_scores = {}
        
        for expiration in options_chain.get_expirations():
            dte = (expiration - date.today()).days
            if not (constraints.min_dte <= dte <= constraints.max_dte):
                continue
            
            # Calculate expiration score based on multiple factors
            score = self._calculate_expiration_score(
                expiration, dte, strategy_type, market_conditions
            )
            
            expiration_scores[expiration] = score
        
        if not expiration_scores:
            # Fallback to middle of DTE range
            target_dte = (constraints.min_dte + constraints.max_dte) // 2
            target_date = date.today() + timedelta(days=target_dte)
            
            # Find closest available expiration
            available_expirations = options_chain.get_expirations()
            if available_expirations:
                return min(available_expirations, key=lambda exp: abs((exp - target_date).days))
            else:
                raise BusinessLogicError("No valid expirations available")
        
        return max(expiration_scores.items(), key=lambda x: x[1])[0]
    
    def calculate_kelly_position_size(
        self,
        strategy: StrategyCandidate,
        portfolio_nav: Decimal,
        risk_tolerance: float = 1.0
    ) -> int:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            strategy: Strategy candidate
            portfolio_nav: Portfolio net asset value
            risk_tolerance: Risk tolerance multiplier (0-2)
            
        Returns:
            Optimal number of contracts
        """
        if not strategy.kelly_fraction or not strategy.strategy.max_loss:
            return 1  # Default position size
        
        try:
            # Kelly fraction adjusted for risk tolerance
            adjusted_kelly = strategy.kelly_fraction * risk_tolerance
            
            # Maximum capital to risk based on Kelly
            max_risk_capital = float(portfolio_nav) * adjusted_kelly
            
            # Position size based on max loss per contract
            max_loss_per_contract = float(strategy.strategy.max_loss)
            
            if max_loss_per_contract <= 0:
                return 1
            
            # Calculate number of contracts
            position_size = int(max_risk_capital / max_loss_per_contract)
            
            # Apply practical limits
            position_size = max(1, min(10, position_size))  # 1-10 contracts
            
            return position_size
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate Kelly position size: {str(e)}")
            return 1
    
    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> MarketConditions:
        """Analyze current market conditions for optimization context."""
        
        # Default conditions
        conditions = MarketConditions(
            volatility_regime=VolatilityRegime.NORMAL_VOL,
            trend_direction="sideways",
            trend_strength=0.5,
            iv_rank=50.0
        )
        
        # Analyze technical indicators
        technicals = market_data.get('technical_summary')
        if technicals:
            conditions.trend_direction = getattr(technicals.trend_analysis, 'trend_direction', 'sideways')
            conditions.trend_strength = getattr(technicals.trend_analysis, 'trend_strength', 0.5)
            
            # Volatility regime from technical analysis
            volatility_analysis = getattr(technicals, 'volatility_analysis', None)
            if volatility_analysis:
                vol_regime = getattr(volatility_analysis, 'volatility_regime', 'normal')
                if vol_regime == 'low':
                    conditions.volatility_regime = VolatilityRegime.LOW_VOL
                elif vol_regime == 'high':
                    conditions.volatility_regime = VolatilityRegime.HIGH_VOL
                elif vol_regime == 'extreme':
                    conditions.volatility_regime = VolatilityRegime.EXTREME_VOL
        
        # Analyze IV rank if available
        market_data_obj = market_data.get('market_data')
        if market_data_obj and hasattr(market_data_obj, 'calculate_iv_rank'):
            conditions.iv_rank = market_data_obj.calculate_iv_rank()
        
        return conditions
    
    def _apply_hard_constraints(
        self,
        candidates: List[StrategyCandidate],
        constraints: OptimizationConstraints
    ) -> List[StrategyCandidate]:
        """Apply hard constraints to filter candidates."""
        
        filtered = []
        
        for candidate in candidates:
            strategy = candidate.strategy
            
            # Probability constraint
            if (strategy.probability_of_profit and 
                strategy.probability_of_profit < constraints.min_probability_of_profit):
                continue
            
            # Max loss constraint
            if (strategy.max_loss and 
                strategy.max_loss > constraints.max_loss_per_trade):
                continue
            
            # Credit-to-max-loss ratio constraint
            if (strategy.credit_to_max_loss_ratio and
                strategy.credit_to_max_loss_ratio < constraints.min_credit_to_max_loss):
                continue
            
            # DTE constraints
            if strategy.days_to_expiration:
                if not (constraints.min_dte <= strategy.days_to_expiration <= constraints.max_dte):
                    continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _optimize_candidate_parameters(
        self,
        candidates: List[StrategyCandidate],
        market_conditions: MarketConditions,
        constraints: OptimizationConstraints
    ) -> List[StrategyCandidate]:
        """Optimize parameters for each candidate based on market conditions."""
        
        optimized = []
        
        for candidate in candidates:
            try:
                # Create optimized copy
                optimized_candidate = self._create_optimized_candidate(
                    candidate, market_conditions, constraints
                )
                
                if optimized_candidate:
                    optimized.append(optimized_candidate)
                    
            except Exception as e:
                self.logger.warning(f"Failed to optimize candidate: {str(e)}")
                # Keep original if optimization fails
                optimized.append(candidate)
        
        return optimized
    
    def _create_optimized_candidate(
        self,
        candidate: StrategyCandidate,
        market_conditions: MarketConditions,
        constraints: OptimizationConstraints
    ) -> Optional[StrategyCandidate]:
        """Create optimized version of strategy candidate."""
        
        # For now, return the original candidate
        # In a full implementation, this would adjust strikes, quantities, etc.
        # based on market conditions and optimization criteria
        
        optimized = StrategyCandidate(
            strategy=candidate.strategy,
            expected_return=candidate.expected_return,
            sharpe_ratio=candidate.sharpe_ratio,
            kelly_fraction=candidate.kelly_fraction,
            max_drawdown=candidate.max_drawdown,
            win_rate=candidate.win_rate,
            profit_factor=candidate.profit_factor,
            optimization_score=candidate.optimization_score
        )
        
        # Apply market condition adjustments
        self._apply_market_condition_adjustments(optimized, market_conditions)
        
        return optimized
    
    def _apply_market_condition_adjustments(
        self,
        candidate: StrategyCandidate,
        market_conditions: MarketConditions
    ):
        """Apply market condition-specific adjustments to candidate."""
        
        # Adjust based on volatility regime
        if market_conditions.volatility_regime == VolatilityRegime.HIGH_VOL:
            # In high volatility, favor higher probability strategies
            if candidate.strategy.probability_of_profit:
                candidate.optimization_score = (candidate.optimization_score or 0) * 1.1
        
        elif market_conditions.volatility_regime == VolatilityRegime.LOW_VOL:
            # In low volatility, favor higher return strategies
            if candidate.expected_return and candidate.expected_return > Decimal('50'):
                candidate.optimization_score = (candidate.optimization_score or 0) * 1.1
        
        # Adjust based on trend
        if market_conditions.trend_direction == "bullish" and market_conditions.trend_strength > 0.7:
            # Strong bullish trend - favor put credit spreads
            if candidate.strategy.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
                candidate.optimization_score = (candidate.optimization_score or 0) * 1.15
        
        elif market_conditions.trend_direction == "bearish" and market_conditions.trend_strength > 0.7:
            # Strong bearish trend - favor call credit spreads
            if candidate.strategy.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
                candidate.optimization_score = (candidate.optimization_score or 0) * 1.15
        
        else:
            # Sideways market - favor iron condors
            if candidate.strategy.strategy_type == StrategyType.IRON_CONDOR:
                candidate.optimization_score = (candidate.optimization_score or 0) * 1.1
    
    def _apply_optimization_objective(
        self,
        candidates: List[StrategyCandidate],
        objective: OptimizationObjective,
        market_conditions: MarketConditions
    ) -> List[StrategyCandidate]:
        """Apply objective-specific optimization scoring."""
        
        for candidate in candidates:
            objective_score = self._calculate_objective_score(candidate, objective, market_conditions)
            
            # Combine with existing optimization score
            if candidate.optimization_score:
                candidate.optimization_score = (candidate.optimization_score + objective_score) / 2
            else:
                candidate.optimization_score = objective_score
        
        return sorted(candidates, key=lambda c: c.optimization_score or 0, reverse=True)
    
    def _calculate_objective_score(
        self,
        candidate: StrategyCandidate,
        objective: OptimizationObjective,
        market_conditions: MarketConditions
    ) -> float:
        """Calculate score based on specific optimization objective."""
        
        if objective == OptimizationObjective.MAXIMIZE_EXPECTED_RETURN:
            if candidate.expected_return:
                return min(100, max(0, float(candidate.expected_return) / 100 * 100))
            return 0
        
        elif objective == OptimizationObjective.MAXIMIZE_PROBABILITY:
            if candidate.strategy.probability_of_profit:
                return candidate.strategy.probability_of_profit * 100
            return 0
        
        elif objective == OptimizationObjective.MAXIMIZE_SHARPE_RATIO:
            if candidate.sharpe_ratio:
                return min(100, max(0, candidate.sharpe_ratio * 50 + 50))
            return 0
        
        elif objective == OptimizationObjective.MINIMIZE_RISK:
            if candidate.strategy.max_loss:
                # Invert max loss for scoring (lower risk = higher score)
                risk_score = 100 - min(100, float(candidate.strategy.max_loss) / 500 * 100)
                return max(0, risk_score)
            return 0
        
        elif objective == OptimizationObjective.MAXIMIZE_KELLY:
            if candidate.kelly_fraction:
                return candidate.kelly_fraction * 400  # Scale to 0-100
            return 0
        
        else:  # BALANCED_APPROACH
            return candidate.optimization_score or 0
    
    def _select_optimal_strategy(
        self,
        candidates: List[StrategyCandidate],
        market_conditions: MarketConditions,
        objective: OptimizationObjective
    ) -> Optional[StrategyCandidate]:
        """Select the optimal strategy from scored candidates."""
        
        if not candidates:
            return None
        
        # Apply final selection criteria
        best_candidates = candidates[:3]  # Top 3 candidates
        
        # Apply tie-breaking logic if scores are close
        if len(best_candidates) > 1:
            top_score = best_candidates[0].optimization_score or 0
            close_candidates = [
                c for c in best_candidates 
                if abs((c.optimization_score or 0) - top_score) < 5
            ]
            
            if len(close_candidates) > 1:
                # Use secondary criteria for tie-breaking
                return self._apply_tie_breaking(close_candidates, market_conditions)
        
        return best_candidates[0]
    
    def _apply_tie_breaking(
        self,
        candidates: List[StrategyCandidate],
        market_conditions: MarketConditions
    ) -> StrategyCandidate:
        """Apply tie-breaking criteria when scores are close."""
        
        # Tie-breaking hierarchy:
        # 1. Higher probability of profit
        # 2. Better credit-to-max-loss ratio
        # 3. Higher liquidity
        # 4. Market condition preference
        
        # Sort by probability of profit
        prob_sorted = sorted(
            candidates,
            key=lambda c: c.strategy.probability_of_profit or 0,
            reverse=True
        )
        
        # If probabilities are close, use credit ratio
        top_prob = prob_sorted[0].strategy.probability_of_profit or 0
        close_prob = [
            c for c in prob_sorted
            if abs((c.strategy.probability_of_profit or 0) - top_prob) < 0.05
        ]
        
        if len(close_prob) > 1:
            ratio_sorted = sorted(
                close_prob,
                key=lambda c: c.strategy.credit_to_max_loss_ratio or 0,
                reverse=True
            )
            return ratio_sorted[0]
        
        return prob_sorted[0]
    
    def _calculate_strike_probabilities(
        self,
        options_chain: OptionsChain,
        expiration: date,
        strikes: List[Decimal],
        strategy_type: StrategyType,
        market_conditions: MarketConditions
    ) -> Dict[Tuple[Decimal, ...], float]:
        """Calculate probability-weighted scores for strike combinations."""
        
        strike_scores = {}
        
        # This is a simplified implementation
        # In practice, would use sophisticated probability models
        
        if strategy_type in [StrategyType.PUT_CREDIT_SPREAD, StrategyType.CALL_CREDIT_SPREAD]:
            # Score two-strike combinations
            for i, short_strike in enumerate(strikes[:-1]):
                for long_strike in strikes[i+1:]:
                    if strategy_type == StrategyType.PUT_CREDIT_SPREAD:
                        if short_strike <= long_strike:
                            continue
                    else:  # CALL
                        if short_strike >= long_strike:
                            continue
                    
                    # Simple probability scoring based on distance from current price
                    if options_chain.underlying_price:
                        current_price = float(options_chain.underlying_price)
                        short_distance = abs(float(short_strike) - current_price) / current_price
                        
                        # Prefer moderate OTM strikes
                        if 0.02 <= short_distance <= 0.10:  # 2-10% OTM
                            score = 1.0 - short_distance
                        else:
                            score = 0.5
                        
                        strike_scores[(short_strike, long_strike)] = score
        
        return strike_scores
    
    def _calculate_expiration_score(
        self,
        expiration: date,
        dte: int,
        strategy_type: StrategyType,
        market_conditions: MarketConditions
    ) -> float:
        """Calculate score for expiration date based on multiple factors."""
        
        score = 0.0
        
        # Base score from DTE (prefer 20-35 days)
        if 20 <= dte <= 35:
            score += 1.0
        elif 15 <= dte <= 45:
            score += 0.8
        else:
            score += 0.5
        
        # Adjust for volatility regime
        if market_conditions.volatility_regime == VolatilityRegime.HIGH_VOL:
            # Prefer shorter DTE in high volatility
            score += max(0, (45 - dte) / 45 * 0.3)
        elif market_conditions.volatility_regime == VolatilityRegime.LOW_VOL:
            # Prefer longer DTE in low volatility
            score += max(0, dte / 45 * 0.3)
        
        # Adjust for earnings proximity (if available)
        if market_conditions.earnings_proximity:
            days_to_earnings = market_conditions.earnings_proximity
            if abs(dte - days_to_earnings) < 5:
                score -= 0.2  # Penalty for earnings risk
        
        return score
    
    def _calculate_optimization_scores(self, candidates: List[StrategyCandidate]) -> Dict[str, float]:
        """Calculate summary optimization scores."""
        
        if not candidates:
            return {}
        
        scores = [c.optimization_score or 0 for c in candidates]
        
        return {
            "best_score": max(scores),
            "average_score": statistics.mean(scores),
            "score_range": max(scores) - min(scores),
            "candidates_count": len(candidates)
        }
    
    def _generate_rationale(
        self,
        selected_strategy: Optional[StrategyCandidate],
        market_conditions: MarketConditions,
        objective: OptimizationObjective
    ) -> str:
        """Generate human-readable rationale for strategy selection."""
        
        if not selected_strategy:
            return "No suitable strategy found meeting optimization criteria."
        
        strategy = selected_strategy.strategy
        rationale_parts = []
        
        # Strategy type rationale
        if strategy.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            rationale_parts.append("Put credit spread selected for bullish/neutral outlook")
        elif strategy.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
            rationale_parts.append("Call credit spread selected for bearish/neutral outlook")
        elif strategy.strategy_type == StrategyType.IRON_CONDOR:
            rationale_parts.append("Iron condor selected for range-bound market expectation")
        
        # Market condition rationale
        if market_conditions.volatility_regime == VolatilityRegime.HIGH_VOL:
            rationale_parts.append("High volatility environment favors premium selling")
        elif market_conditions.volatility_regime == VolatilityRegime.LOW_VOL:
            rationale_parts.append("Low volatility supports income-generating strategies")
        
        # Performance rationale
        if strategy.probability_of_profit and strategy.probability_of_profit > 0.70:
            rationale_parts.append(f"High probability of profit ({strategy.probability_of_profit:.1%})")
        
        if strategy.credit_to_max_loss_ratio and strategy.credit_to_max_loss_ratio > 0.40:
            rationale_parts.append(f"Favorable risk/reward ratio ({strategy.credit_to_max_loss_ratio:.1%})")
        
        return ". ".join(rationale_parts) + "."
    
    def _calculate_confidence_score(
        self,
        selected_strategy: Optional[StrategyCandidate],
        market_conditions: MarketConditions
    ) -> float:
        """Calculate confidence score for the optimization result."""
        
        if not selected_strategy:
            return 0.0
        
        confidence_factors = []
        
        # Strategy quality factors
        if selected_strategy.optimization_score and selected_strategy.optimization_score > 75:
            confidence_factors.append(0.9)
        elif selected_strategy.optimization_score and selected_strategy.optimization_score > 60:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Market condition clarity
        if market_conditions.trend_strength > 0.8:
            confidence_factors.append(0.9)  # Clear trend
        elif market_conditions.trend_strength > 0.5:
            confidence_factors.append(0.7)  # Moderate trend
        else:
            confidence_factors.append(0.5)  # Unclear trend
        
        # Data quality (simplified)
        confidence_factors.append(0.8)  # Assume good data quality
        
        return statistics.mean(confidence_factors)