"""
Dynamic weight allocation system with market regime-based adjustments and performance backtesting.
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

from ...data.models.market_data import OHLCVData, TechnicalIndicators
from ...data.models.trades import TradeCandidate, StrategyType
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, CalculationError
)

from .scoring_engine import ScoringWeights, MarketRegimeType, ScoredTradeCandidate


class PerformanceMetric(Enum):
    """Performance metrics for weight optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    TOTAL_RETURN = "total_return"


@dataclass
class PerformanceWindow:
    """Performance measurement window."""
    start_date: date
    end_date: date
    returns: List[float]
    trades_count: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float


@dataclass
class WeightOptimizationResult:
    """Result of weight optimization process."""
    optimized_weights: ScoringWeights
    performance_improvement: float
    confidence_score: float
    backtest_periods: int
    validation_score: float
    optimization_rationale: str


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""
    regime: MarketRegimeType
    sample_size: int
    avg_return: float
    volatility: float
    sharpe_ratio: float
    win_rate: float
    optimal_weights: ScoringWeights
    performance_attribution: Dict[str, float]


class DynamicWeightManager:
    """
    Dynamic weight allocation system implementing market regime-based weighting with performance optimization.
    
    Features:
    - Market regime detection and classification
    - Historical performance backtesting with rolling windows
    - Cross-validation to prevent overfitting
    - Performance attribution analysis by component
    - Adaptive weight adjustments based on regime changes
    - Real-time performance monitoring and optimization
    - Confidence scoring for weight adjustments
    """
    
    def __init__(self, lookback_periods: int = 252):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.lookback_periods = lookback_periods  # Trading days for analysis
        
        # Performance tracking
        self.performance_history: List[PerformanceWindow] = []
        self.regime_performance: Dict[MarketRegimeType, RegimePerformance] = {}
        
        # Weight optimization parameters
        self.optimization_window = 63  # Quarter for optimization
        self.validation_window = 21    # Month for validation
        self.min_sample_size = 30      # Minimum trades for reliable statistics
        
        # Base weights for different market conditions
        self.regime_base_weights = self._initialize_regime_base_weights()
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_sharpe_ratio': 0.5,
            'min_win_rate': 0.55,
            'max_drawdown': 0.15,
            'min_trades_for_optimization': 50
        }
    
    @handle_errors(operation_name="optimize_weights")
    def optimize_weights_for_regime(
        self,
        market_regime: MarketRegimeType,
        historical_performance: List[Dict[str, Any]],
        current_weights: ScoringWeights,
        optimization_metric: PerformanceMetric = PerformanceMetric.SHARPE_RATIO
    ) -> WeightOptimizationResult:
        """
        Optimize scoring weights for a specific market regime based on historical performance.
        
        Args:
            market_regime: Target market regime
            historical_performance: Historical trade performance data
            current_weights: Current scoring weights
            optimization_metric: Metric to optimize for
            
        Returns:
            Weight optimization result with performance analysis
        """
        self.logger.info(f"Optimizing weights for {market_regime.value} using {optimization_metric.value}")
        
        if len(historical_performance) < self.performance_thresholds['min_trades_for_optimization']:
            return WeightOptimizationResult(
                optimized_weights=current_weights,
                performance_improvement=0.0,
                confidence_score=0.0,
                backtest_periods=0,
                validation_score=0.0,
                optimization_rationale="Insufficient historical data for optimization"
            )
        
        # Filter performance data for the specific regime
        regime_performance_data = self._filter_performance_by_regime(
            historical_performance, market_regime
        )
        
        if len(regime_performance_data) < self.min_sample_size:
            return WeightOptimizationResult(
                optimized_weights=current_weights,
                performance_improvement=0.0,
                confidence_score=0.0,
                backtest_periods=len(regime_performance_data),
                validation_score=0.0,
                optimization_rationale=f"Insufficient regime-specific data: {len(regime_performance_data)} trades"
            )
        
        # Perform weight optimization using grid search
        optimized_weights = self._grid_search_optimization(
            regime_performance_data, optimization_metric
        )
        
        # Validate optimization using cross-validation
        validation_score = self._cross_validate_weights(
            regime_performance_data, optimized_weights, optimization_metric
        )
        
        # Calculate performance improvement
        baseline_performance = self._calculate_performance_metric(
            regime_performance_data, current_weights, optimization_metric
        )
        optimized_performance = self._calculate_performance_metric(
            regime_performance_data, optimized_weights, optimization_metric
        )
        
        performance_improvement = optimized_performance - baseline_performance
        
        # Calculate confidence score
        confidence_score = self._calculate_optimization_confidence(
            regime_performance_data, optimized_weights, validation_score
        )
        
        # Generate rationale
        rationale = self._generate_optimization_rationale(
            market_regime, performance_improvement, confidence_score, len(regime_performance_data)
        )
        
        return WeightOptimizationResult(
            optimized_weights=optimized_weights,
            performance_improvement=performance_improvement,
            confidence_score=confidence_score,
            backtest_periods=len(regime_performance_data),
            validation_score=validation_score,
            optimization_rationale=rationale
        )
    
    @handle_errors(operation_name="analyze_regime_performance")
    def analyze_regime_performance(
        self,
        historical_performance: List[Dict[str, Any]],
        market_regimes: List[MarketRegimeType]
    ) -> Dict[MarketRegimeType, RegimePerformance]:
        """
        Analyze performance characteristics across different market regimes.
        
        Args:
            historical_performance: Historical trade performance data
            market_regimes: List of market regimes to analyze
            
        Returns:
            Dictionary of regime performance analyses
        """
        self.logger.info(f"Analyzing performance across {len(market_regimes)} market regimes")
        
        regime_analyses = {}
        
        for regime in market_regimes:
            regime_data = self._filter_performance_by_regime(historical_performance, regime)
            
            if len(regime_data) < 10:  # Minimum sample size
                continue
            
            # Calculate performance metrics
            returns = [trade['pnl'] for trade in regime_data]
            winning_trades = [r for r in returns if r > 0]
            
            avg_return = statistics.mean(returns) if returns else 0.0
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
            win_rate = len(winning_trades) / len(returns) if returns else 0.0
            
            # Find optimal weights for this regime
            optimal_weights = self._find_optimal_weights_for_regime(regime_data)
            
            # Calculate performance attribution
            attribution = self._calculate_performance_attribution(regime_data, optimal_weights)
            
            regime_analyses[regime] = RegimePerformance(
                regime=regime,
                sample_size=len(regime_data),
                avg_return=avg_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                optimal_weights=optimal_weights,
                performance_attribution=attribution
            )
        
        self.regime_performance = regime_analyses
        return regime_analyses
    
    @handle_errors(operation_name="adaptive_weight_adjustment")
    def get_adaptive_weights(
        self,
        current_regime: MarketRegimeType,
        regime_strength: float,
        base_weights: ScoringWeights,
        performance_feedback: Optional[List[Dict[str, Any]]] = None
    ) -> ScoringWeights:
        """
        Get adaptively adjusted weights based on current market regime and performance feedback.
        
        Args:
            current_regime: Current market regime
            regime_strength: Strength of regime signal (0-1)
            base_weights: Base scoring weights
            performance_feedback: Recent performance data for adjustment
            
        Returns:
            Adaptively adjusted scoring weights
        """
        # Start with regime-based base weights
        regime_weights = self.regime_base_weights.get(current_regime, base_weights)
        
        # Apply regime strength weighting
        adjusted_weights = self._blend_weights(base_weights, regime_weights, regime_strength)
        
        # Apply performance-based adjustments if available
        if performance_feedback and len(performance_feedback) >= 20:
            performance_weights = self._calculate_performance_based_adjustments(
                performance_feedback, adjusted_weights
            )
            # Blend with performance adjustments (lower weight to prevent overfitting)
            adjusted_weights = self._blend_weights(adjusted_weights, performance_weights, 0.3)
        
        # Apply regime-specific optimizations if available
        if current_regime in self.regime_performance:
            optimal_regime_weights = self.regime_performance[current_regime].optimal_weights
            confidence = min(1.0, self.regime_performance[current_regime].sample_size / 100)
            adjusted_weights = self._blend_weights(adjusted_weights, optimal_regime_weights, confidence * 0.4)
        
        return adjusted_weights
    
    def _initialize_regime_base_weights(self) -> Dict[MarketRegimeType, ScoringWeights]:
        """Initialize base weights for different market regimes."""
        
        return {
            MarketRegimeType.HIGH_VOLATILITY: ScoringWeights(
                pop=0.30, iv_rank=0.25, momentum=0.15, flow=0.15, risk_reward=0.10, liquidity=0.05
            ),
            MarketRegimeType.LOW_VOLATILITY: ScoringWeights(
                pop=0.20, iv_rank=0.15, momentum=0.25, flow=0.20, risk_reward=0.15, liquidity=0.05
            ),
            MarketRegimeType.TRENDING_UP: ScoringWeights(
                pop=0.20, iv_rank=0.15, momentum=0.30, flow=0.20, risk_reward=0.10, liquidity=0.05
            ),
            MarketRegimeType.TRENDING_DOWN: ScoringWeights(
                pop=0.35, iv_rank=0.20, momentum=0.10, flow=0.15, risk_reward=0.15, liquidity=0.05
            ),
            MarketRegimeType.SIDEWAYS: ScoringWeights(
                pop=0.25, iv_rank=0.20, momentum=0.15, flow=0.15, risk_reward=0.20, liquidity=0.05
            ),
            MarketRegimeType.CRISIS: ScoringWeights(
                pop=0.40, iv_rank=0.15, momentum=0.05, flow=0.10, risk_reward=0.20, liquidity=0.10
            )
        }
    
    def _filter_performance_by_regime(
        self,
        performance_data: List[Dict[str, Any]],
        regime: MarketRegimeType
    ) -> List[Dict[str, Any]]:
        """Filter performance data for a specific market regime."""
        
        filtered_data = []
        
        for trade in performance_data:
            trade_regime = trade.get('market_regime')
            if trade_regime == regime or trade_regime == regime.value:
                filtered_data.append(trade)
        
        return filtered_data
    
    def _grid_search_optimization(
        self,
        performance_data: List[Dict[str, Any]],
        optimization_metric: PerformanceMetric
    ) -> ScoringWeights:
        """Perform grid search optimization for weights."""
        
        # Define search grid (coarse grid for computational efficiency)
        weight_ranges = {
            'pop': np.arange(0.15, 0.45, 0.05),
            'iv_rank': np.arange(0.10, 0.30, 0.05),
            'momentum': np.arange(0.10, 0.35, 0.05),
            'flow': np.arange(0.05, 0.25, 0.05),
            'risk_reward': np.arange(0.05, 0.25, 0.05),
            'liquidity': np.arange(0.02, 0.08, 0.02)
        }
        
        best_performance = float('-inf')
        best_weights = ScoringWeights()
        
        # Sample a subset of combinations to avoid computational explosion
        max_combinations = 1000
        combinations_tested = 0
        
        for pop in weight_ranges['pop']:
            if combinations_tested >= max_combinations:
                break
            for iv_rank in weight_ranges['iv_rank']:
                if combinations_tested >= max_combinations:
                    break
                for momentum in weight_ranges['momentum']:
                    if combinations_tested >= max_combinations:
                        break
                    
                    # Calculate remaining weights proportionally
                    remaining_weight = 1.0 - pop - iv_rank - momentum
                    if remaining_weight < 0.20:  # Ensure minimum for other components
                        continue
                    
                    # Distribute remaining weight proportionally
                    flow = min(0.25, remaining_weight * 0.4)
                    risk_reward = min(0.25, remaining_weight * 0.4)
                    liquidity = remaining_weight - flow - risk_reward
                    
                    if liquidity < 0.02 or liquidity > 0.10:
                        continue
                    
                    test_weights = ScoringWeights(
                        pop=pop, iv_rank=iv_rank, momentum=momentum,
                        flow=flow, risk_reward=risk_reward, liquidity=liquidity
                    )
                    
                    performance = self._calculate_performance_metric(
                        performance_data, test_weights, optimization_metric
                    )
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_weights = test_weights
                    
                    combinations_tested += 1
        
        self.logger.debug(f"Grid search tested {combinations_tested} combinations, best {optimization_metric.value}: {best_performance:.3f}")
        
        return best_weights
    
    def _calculate_performance_metric(
        self,
        performance_data: List[Dict[str, Any]],
        weights: ScoringWeights,
        metric: PerformanceMetric
    ) -> float:
        """Calculate performance metric for given weights."""
        
        if not performance_data:
            return 0.0
        
        # Simulate scoring with these weights and calculate resulting performance
        scores = []
        returns = []
        
        for trade in performance_data:
            # Calculate weighted score (simplified simulation)
            score = (
                trade.get('pop_score', 50) * weights.pop +
                trade.get('iv_rank_score', 50) * weights.iv_rank +
                trade.get('momentum_score', 50) * weights.momentum +
                trade.get('flow_score', 50) * weights.flow +
                trade.get('risk_reward_score', 50) * weights.risk_reward +
                trade.get('liquidity_score', 50) * weights.liquidity
            )
            scores.append(score)
            returns.append(trade.get('pnl', 0))
        
        # Calculate performance metric
        if metric == PerformanceMetric.SHARPE_RATIO:
            if len(returns) > 1:
                avg_return = statistics.mean(returns)
                volatility = statistics.stdev(returns)
                return avg_return / volatility if volatility > 0 else 0.0
            return 0.0
        
        elif metric == PerformanceMetric.WIN_RATE:
            winning_trades = sum(1 for r in returns if r > 0)
            return winning_trades / len(returns) if returns else 0.0
        
        elif metric == PerformanceMetric.TOTAL_RETURN:
            return sum(returns)
        
        elif metric == PerformanceMetric.PROFIT_FACTOR:
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            return gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        else:
            return statistics.mean(returns) if returns else 0.0
    
    def _cross_validate_weights(
        self,
        performance_data: List[Dict[str, Any]],
        weights: ScoringWeights,
        metric: PerformanceMetric,
        k_folds: int = 5
    ) -> float:
        """Perform k-fold cross-validation on weights."""
        
        if len(performance_data) < k_folds:
            return 0.0
        
        # Shuffle data
        shuffled_data = performance_data.copy()
        np.random.shuffle(shuffled_data)
        
        fold_size = len(shuffled_data) // k_folds
        validation_scores = []
        
        for i in range(k_folds):
            # Create train/validation split
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else len(shuffled_data)
            
            validation_data = shuffled_data[start_idx:end_idx]
            
            if len(validation_data) > 0:
                score = self._calculate_performance_metric(validation_data, weights, metric)
                validation_scores.append(score)
        
        return statistics.mean(validation_scores) if validation_scores else 0.0
    
    def _calculate_optimization_confidence(
        self,
        performance_data: List[Dict[str, Any]],
        weights: ScoringWeights,
        validation_score: float
    ) -> float:
        """Calculate confidence score for weight optimization."""
        
        confidence_factors = []
        
        # Sample size factor
        sample_size_factor = min(1.0, len(performance_data) / 100)
        confidence_factors.append(sample_size_factor)
        
        # Validation performance factor
        validation_factor = max(0.0, min(1.0, validation_score / 2.0))  # Assume good validation score is ~2.0
        confidence_factors.append(validation_factor)
        
        # Weight distribution factor (penalize extreme weights)
        weight_values = [weights.pop, weights.iv_rank, weights.momentum, 
                        weights.flow, weights.risk_reward, weights.liquidity]
        weight_entropy = -sum(w * math.log(w + 1e-10) for w in weight_values)
        max_entropy = math.log(len(weight_values))
        entropy_factor = weight_entropy / max_entropy
        confidence_factors.append(entropy_factor)
        
        return statistics.mean(confidence_factors)
    
    def _find_optimal_weights_for_regime(self, regime_data: List[Dict[str, Any]]) -> ScoringWeights:
        """Find optimal weights for a specific regime using simplified optimization."""
        
        if len(regime_data) < 20:
            return ScoringWeights()  # Default weights
        
        # Use grid search with smaller grid for regime-specific optimization
        return self._grid_search_optimization(regime_data, PerformanceMetric.SHARPE_RATIO)
    
    def _calculate_performance_attribution(
        self,
        regime_data: List[Dict[str, Any]],
        weights: ScoringWeights
    ) -> Dict[str, float]:
        """Calculate performance attribution by scoring component."""
        
        if not regime_data:
            return {}
        
        attribution = {}
        total_weighted_score = 0
        
        # Calculate weighted contribution of each component
        component_contributions = {
            'pop': [],
            'iv_rank': [],
            'momentum': [],
            'flow': [],
            'risk_reward': [],
            'liquidity': []
        }
        
        for trade in regime_data:
            pop_contrib = trade.get('pop_score', 50) * weights.pop
            iv_contrib = trade.get('iv_rank_score', 50) * weights.iv_rank
            momentum_contrib = trade.get('momentum_score', 50) * weights.momentum
            flow_contrib = trade.get('flow_score', 50) * weights.flow
            rr_contrib = trade.get('risk_reward_score', 50) * weights.risk_reward
            liq_contrib = trade.get('liquidity_score', 50) * weights.liquidity
            
            total_score = (pop_contrib + iv_contrib + momentum_contrib + 
                          flow_contrib + rr_contrib + liq_contrib)
            
            if total_score > 0:
                component_contributions['pop'].append(pop_contrib / total_score * trade.get('pnl', 0))
                component_contributions['iv_rank'].append(iv_contrib / total_score * trade.get('pnl', 0))
                component_contributions['momentum'].append(momentum_contrib / total_score * trade.get('pnl', 0))
                component_contributions['flow'].append(flow_contrib / total_score * trade.get('pnl', 0))
                component_contributions['risk_reward'].append(rr_contrib / total_score * trade.get('pnl', 0))
                component_contributions['liquidity'].append(liq_contrib / total_score * trade.get('pnl', 0))
        
        # Calculate average attribution
        for component, contributions in component_contributions.items():
            attribution[component] = statistics.mean(contributions) if contributions else 0.0
        
        return attribution
    
    def _blend_weights(
        self,
        weights1: ScoringWeights,
        weights2: ScoringWeights,
        blend_factor: float
    ) -> ScoringWeights:
        """Blend two sets of weights with specified factor."""
        
        blend_factor = max(0.0, min(1.0, blend_factor))
        
        return ScoringWeights(
            pop=weights1.pop * (1 - blend_factor) + weights2.pop * blend_factor,
            iv_rank=weights1.iv_rank * (1 - blend_factor) + weights2.iv_rank * blend_factor,
            momentum=weights1.momentum * (1 - blend_factor) + weights2.momentum * blend_factor,
            flow=weights1.flow * (1 - blend_factor) + weights2.flow * blend_factor,
            risk_reward=weights1.risk_reward * (1 - blend_factor) + weights2.risk_reward * blend_factor,
            liquidity=weights1.liquidity * (1 - blend_factor) + weights2.liquidity * blend_factor
        )
    
    def _calculate_performance_based_adjustments(
        self,
        recent_performance: List[Dict[str, Any]],
        current_weights: ScoringWeights
    ) -> ScoringWeights:
        """Calculate weight adjustments based on recent performance."""
        
        # Analyze which components are performing well
        component_performance = {
            'pop': [],
            'iv_rank': [],
            'momentum': [],
            'flow': [],
            'risk_reward': [],
            'liquidity': []
        }
        
        for trade in recent_performance:
            pnl = trade.get('pnl', 0)
            
            # Correlate component scores with performance
            for component in component_performance:
                score_key = f"{component}_score"
                if score_key in trade:
                    # Simple correlation: higher score should lead to better performance
                    component_performance[component].append((trade[score_key], pnl))
        
        # Calculate adjustments based on performance correlation
        adjusted_weights = ScoringWeights(
            pop=current_weights.pop,
            iv_rank=current_weights.iv_rank,
            momentum=current_weights.momentum,
            flow=current_weights.flow,
            risk_reward=current_weights.risk_reward,
            liquidity=current_weights.liquidity
        )
        
        # Simple adjustment: increase weights for components with positive correlation to returns
        adjustment_factor = 0.1  # Maximum 10% adjustment
        
        for component, performance_data in component_performance.items():
            if len(performance_data) > 10:
                scores, returns = zip(*performance_data)
                correlation = np.corrcoef(scores, returns)[0, 1] if len(scores) > 1 else 0
                
                # Adjust weight based on correlation
                current_weight = getattr(adjusted_weights, component)
                adjustment = correlation * adjustment_factor
                new_weight = max(0.02, min(0.50, current_weight + adjustment))
                setattr(adjusted_weights, component, new_weight)
        
        # Renormalize weights
        total_weight = (adjusted_weights.pop + adjusted_weights.iv_rank + 
                       adjusted_weights.momentum + adjusted_weights.flow + 
                       adjusted_weights.risk_reward + adjusted_weights.liquidity)
        
        if total_weight > 0:
            adjusted_weights.pop /= total_weight
            adjusted_weights.iv_rank /= total_weight
            adjusted_weights.momentum /= total_weight
            adjusted_weights.flow /= total_weight
            adjusted_weights.risk_reward /= total_weight
            adjusted_weights.liquidity /= total_weight
        
        return adjusted_weights
    
    def _generate_optimization_rationale(
        self,
        regime: MarketRegimeType,
        improvement: float,
        confidence: float,
        sample_size: int
    ) -> str:
        """Generate human-readable optimization rationale."""
        
        rationale_parts = []
        
        # Regime-specific rationale
        regime_name = regime.value.replace('_', ' ').title()
        rationale_parts.append(f"Optimized for {regime_name} conditions")
        
        # Performance improvement
        if improvement > 0.1:
            rationale_parts.append(f"Significant improvement: +{improvement:.1%}")
        elif improvement > 0.05:
            rationale_parts.append(f"Moderate improvement: +{improvement:.1%}")
        elif improvement > 0:
            rationale_parts.append(f"Marginal improvement: +{improvement:.1%}")
        else:
            rationale_parts.append("No improvement over baseline")
        
        # Confidence assessment
        if confidence > 0.8:
            rationale_parts.append("High confidence")
        elif confidence > 0.6:
            rationale_parts.append("Medium confidence")
        else:
            rationale_parts.append("Low confidence")
        
        # Sample size assessment
        rationale_parts.append(f"Based on {sample_size} historical trades")
        
        return "; ".join(rationale_parts)