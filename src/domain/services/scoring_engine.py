"""
Multi-factor scoring model for options strategy ranking and selection.
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
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData, OHLCVData
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, TradeLeg, StrategyType
)
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, CalculationError
)

from .constraint_engine import GICS_SECTORS
from .dynamic_constraint_manager import MarketRegime, VolatilityRegime


class MarketRegimeType(Enum):
    """Market regime types for scoring weight adjustments."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class ScoringWeights:
    """Scoring weights for different components."""
    pop: float = 0.25           # Probability of profit weight
    iv_rank: float = 0.20       # IV rank weight  
    momentum: float = 0.20      # Momentum weight
    flow: float = 0.15          # Flow weight
    risk_reward: float = 0.15   # Risk/reward weight
    liquidity: float = 0.05     # Liquidity weight


@dataclass
class ComponentScores:
    """Individual component scores for a trade."""
    pop_score: float
    iv_rank_score: float
    momentum_score: float
    flow_score: float
    risk_reward_score: float
    liquidity_score: float
    
    # Z-scores for ranking
    momentum_z: float = 0.0
    flow_z: float = 0.0
    
    # Calculated scores
    model_score: float = 0.0
    
    # Supporting data
    iv_current: Optional[float] = None
    iv_rank: Optional[float] = None


@dataclass
class ScoredTradeCandidate:
    """Trade candidate with comprehensive scoring."""
    trade_candidate: TradeCandidate
    component_scores: ComponentScores
    ranking_tier: int = 1  # 1 = highest tier
    ranking_rationale: str = ""


# Sector ETF mapping for flow calculations
SECTOR_ETF_MAP = {
    "Information Technology": "XLK",
    "Health Care": "XLV", 
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB"
}


class ScoringEngine:
    """
    Multi-factor scoring model for options strategies with dynamic weight allocation.
    
    Features:
    - Composite scoring system with 6 primary components
    - Dynamic weight adjustments based on market regime
    - Momentum and flow Z-score calculations
    - IV rank scoring with historical analysis
    - Risk/reward ratio optimization
    - Liquidity quality assessment
    - Market regime-based weight rebalancing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Base scoring weights
        self.base_weights = ScoringWeights()
        
        # Market regime weight adjustments
        self.regime_adjustments = {
            MarketRegimeType.HIGH_VOLATILITY: {
                'iv_rank': 1.3,
                'momentum': 0.7,
                'pop': 1.1
            },
            MarketRegimeType.LOW_VOLATILITY: {
                'momentum': 1.3,
                'flow': 1.2,
                'iv_rank': 0.8,
                'risk_reward': 1.1
            },
            MarketRegimeType.TRENDING_UP: {
                'momentum': 1.4,
                'pop': 0.9,
                'flow': 1.2
            },
            MarketRegimeType.TRENDING_DOWN: {
                'pop': 1.3,
                'momentum': 0.7,
                'risk_reward': 1.2
            },
            MarketRegimeType.SIDEWAYS: {
                'pop': 1.2,
                'risk_reward': 1.1,
                'liquidity': 1.1
            },
            MarketRegimeType.CRISIS: {
                'pop': 1.5,
                'liquidity': 1.3,
                'iv_rank': 0.8,
                'momentum': 0.6
            }
        }
    
    @handle_errors(operation_name="calculate_model_score")
    def calculate_model_score(
        self,
        trade_candidate: TradeCandidate,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, List[OHLCVData]]] = None,
        market_regime: Optional[MarketRegimeType] = None
    ) -> ScoredTradeCandidate:
        """
        Calculate composite model score for a trade candidate.
        
        Args:
            trade_candidate: Trade candidate to score
            market_data: Current market data context
            historical_data: Historical price data for calculations
            market_regime: Current market regime
            
        Returns:
            Scored trade candidate with component breakdown
        """
        strategy = trade_candidate.strategy
        ticker = strategy.underlying
        
        self.logger.debug(f"Calculating model score for {ticker} {strategy.strategy_type.value}")
        
        # Calculate component scores
        pop_score = self._normalize_pop_score(strategy.probability_of_profit)
        iv_rank_score = self._calculate_iv_rank_score(ticker, market_data, historical_data)
        momentum_score, momentum_z = self._calculate_momentum_score(ticker, historical_data)
        flow_score, flow_z = self._calculate_flow_score(ticker, market_data)
        risk_reward_score = self._calculate_risk_reward_score(strategy)
        liquidity_score = self._calculate_liquidity_score(strategy.legs)
        
        # Get weights adjusted for market regime
        weights = self._adjust_weights_for_regime(self.base_weights, market_regime)
        
        # Calculate composite model score
        model_score = (
            pop_score * weights.pop +
            iv_rank_score * weights.iv_rank +
            momentum_score * weights.momentum +
            flow_score * weights.flow +
            risk_reward_score * weights.risk_reward +
            liquidity_score * weights.liquidity
        )
        
        # Create component scores object
        component_scores = ComponentScores(
            pop_score=pop_score,
            iv_rank_score=iv_rank_score,
            momentum_score=momentum_score,
            flow_score=flow_score,
            risk_reward_score=risk_reward_score,
            liquidity_score=liquidity_score,
            momentum_z=momentum_z,
            flow_z=flow_z,
            model_score=round(model_score, 2)
        )
        
        # Determine ranking tier and rationale
        ranking_tier = self._determine_ranking_tier(model_score, component_scores)
        ranking_rationale = self._generate_ranking_rationale(component_scores, weights, market_regime)
        
        return ScoredTradeCandidate(
            trade_candidate=trade_candidate,
            component_scores=component_scores,
            ranking_tier=ranking_tier,
            ranking_rationale=ranking_rationale
        )
    
    @handle_errors(operation_name="batch_score_trades")
    def batch_score_trades(
        self,
        trade_candidates: List[TradeCandidate],
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, List[OHLCVData]]] = None,
        market_regime: Optional[MarketRegimeType] = None
    ) -> List[ScoredTradeCandidate]:
        """
        Score multiple trade candidates in batch.
        
        Args:
            trade_candidates: List of trade candidates to score
            market_data: Current market data context
            historical_data: Historical price data
            market_regime: Current market regime
            
        Returns:
            List of scored trade candidates sorted by model score
        """
        self.logger.info(f"Batch scoring {len(trade_candidates)} trade candidates")
        
        scored_candidates = []
        
        for candidate in trade_candidates:
            try:
                scored_candidate = self.calculate_model_score(
                    candidate, market_data, historical_data, market_regime
                )
                scored_candidates.append(scored_candidate)
            except Exception as e:
                self.logger.warning(f"Failed to score {candidate.strategy.underlying}: {str(e)}")
                # Add with default score
                default_scores = ComponentScores(
                    pop_score=0, iv_rank_score=0, momentum_score=0,
                    flow_score=0, risk_reward_score=0, liquidity_score=0,
                    model_score=0.0
                )
                scored_candidates.append(ScoredTradeCandidate(
                    trade_candidate=candidate,
                    component_scores=default_scores,
                    ranking_tier=5,
                    ranking_rationale="Scoring failed - assigned default values"
                ))
        
        # Sort by model score (highest first)
        scored_candidates.sort(key=lambda x: x.component_scores.model_score, reverse=True)
        
        self.logger.info(f"Completed batch scoring. Top score: {scored_candidates[0].component_scores.model_score if scored_candidates else 0}")
        
        return scored_candidates
    
    def _normalize_pop_score(self, pop: Optional[float]) -> float:
        """Convert POP to 0-100 score."""
        if not pop:
            return 0.0
        
        # Linear scaling from minimum viable POP (0.65) to perfect (1.0)
        if pop < 0.65:
            return 0.0
        
        return ((pop - 0.65) / 0.35) * 100
    
    def _calculate_iv_rank_score(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, List[OHLCVData]]]
    ) -> float:
        """Calculate IV Rank score based on current IV vs historical range."""
        
        # Get current IV from market data
        ticker_data = market_data.get(ticker, {})
        iv_current = ticker_data.get('implied_volatility')
        
        if not iv_current:
            return 50.0  # Neutral score if no IV data
        
        # Get historical IV data or estimate from price volatility
        iv_history = self._get_iv_history(ticker, market_data, historical_data)
        
        if len(iv_history) < 50:
            return 50.0  # Neutral score if insufficient history
        
        # Calculate IV rank (percentile)
        iv_rank = self._calculate_percentile(iv_history, iv_current)
        
        # For credit strategies, high IV rank is favorable (selling premium)
        # For debit strategies, low IV rank is favorable (buying cheap options)
        
        # Most strategies in our system are credit strategies, so favor high IV rank
        return iv_rank
    
    def _calculate_momentum_score(
        self,
        ticker: str,
        historical_data: Optional[Dict[str, List[OHLCVData]]]
    ) -> Tuple[float, float]:
        """Calculate momentum score and Z-score."""
        
        if not historical_data or ticker not in historical_data:
            return 50.0, 0.0  # Neutral score and Z-score
        
        price_data = historical_data[ticker]
        if len(price_data) < 21:  # Need at least 21 days for 20-day momentum
            return 50.0, 0.0
        
        # Calculate momentum Z-score
        momentum_z = self._calculate_momentum_z_score(price_data)
        
        # Convert Z-score to 0-100 score
        # Z-scores typically range -3 to +3, normalize to 0-100
        # Positive momentum generally favored
        normalized_score = 50 + (momentum_z * 16.67)  # 16.67 = 50/3
        momentum_score = max(0, min(100, normalized_score))
        
        return momentum_score, momentum_z
    
    def _calculate_flow_score(
        self,
        ticker: str,
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate flow score and Z-score."""
        
        # Calculate flow Z-score
        flow_z = self._calculate_flow_z_score(ticker, market_data)
        
        # Convert Z-score to 0-100 score
        # Similar to momentum, positive flow favored
        normalized_score = 50 + (flow_z * 16.67)
        flow_score = max(0, min(100, normalized_score))
        
        return flow_score, flow_z
    
    def _calculate_risk_reward_score(self, strategy: StrategyDefinition) -> float:
        """Calculate risk/reward ratio score."""
        
        if not strategy.max_loss or strategy.max_loss <= 0:
            return 0.0
        
        # For credit strategies, higher credit relative to max loss is better
        if strategy.net_credit and strategy.net_credit > 0:
            ratio = float(strategy.net_credit) / float(strategy.max_loss)
            # Scale ratio (0.33 minimum to 1.0 maximum) to 0-100
            score = ((ratio - 0.33) / 0.67) * 100
        elif strategy.max_profit:
            # For other strategies, use profit potential
            ratio = float(strategy.max_profit) / float(strategy.max_loss)
            score = min(ratio * 25, 100)  # Cap at 100
        else:
            return 0.0
        
        return max(0, min(100, score))
    
    def _calculate_liquidity_score(self, legs: List[TradeLeg]) -> float:
        """Calculate liquidity quality score."""
        
        if not legs:
            return 0.0
        
        total_score = 0.0
        
        for leg in legs:
            option = leg.option
            leg_score = 0.0
            
            # Open interest component (0-40 points)
            if option.open_interest:
                oi_score = min(option.open_interest / 1000 * 40, 40)
                leg_score += oi_score
            
            # Bid-ask spread component (0-40 points)
            if option.bid and option.ask and option.bid > 0 and option.ask > 0:
                mid_price = (option.bid + option.ask) / 2
                spread_pct = (option.ask - option.bid) / mid_price if mid_price > 0 else 1
                spread_score = max(0, 40 * (1 - spread_pct * 20))  # Penalize wide spreads
                leg_score += spread_score
            
            # Volume component (0-20 points)
            if option.volume:
                volume_score = min(option.volume / 100 * 20, 20)
                leg_score += volume_score
            
            total_score += leg_score
        
        # Average across legs
        return total_score / len(legs)
    
    def _calculate_momentum_z_score(self, price_data: List[OHLCVData]) -> float:
        """Calculate standardized momentum score."""
        
        if len(price_data) < 272:  # Need 252 + 21 for full calculation
            return 0.0
        
        # Get prices
        prices = [float(candle.close) for candle in price_data]
        
        # Calculate current 20-day momentum
        current_price = prices[-1]
        price_20d_ago = prices[-21]
        momentum_20d = (current_price / price_20d_ago) - 1
        
        # Calculate historical momentum data (252 trading days = 1 year)
        historical_momentum = []
        for i in range(252, len(prices)):
            hist_momentum = (prices[i] / prices[i-20]) - 1
            historical_momentum.append(hist_momentum)
        
        if len(historical_momentum) < 50:  # Need sufficient history
            return 0.0
        
        # Calculate Z-score
        mean_momentum = statistics.mean(historical_momentum)
        std_momentum = statistics.stdev(historical_momentum) if len(historical_momentum) > 1 else 0
        
        if std_momentum == 0:
            return 0.0
        
        momentum_z = (momentum_20d - mean_momentum) / std_momentum
        
        # Cap extreme values
        return max(-3.0, min(3.0, momentum_z))
    
    def _calculate_flow_z_score(self, ticker: str, market_data: Dict[str, Any]) -> float:
        """Calculate flow Z-score from ETF and options flow."""
        
        # Get sector ETF for the ticker
        sector = GICS_SECTORS.get(ticker, "Unknown")
        sector_etf = SECTOR_ETF_MAP.get(sector, "SPY")
        
        # ETF flow component
        etf_flow_data = market_data.get('etf_flows', {})
        etf_flows = etf_flow_data.get(sector_etf, [])
        
        if len(etf_flows) >= 20:
            current_flow = etf_flows[-1] if etf_flows else 0
            historical_flows = etf_flows[-252:] if len(etf_flows) >= 252 else etf_flows
            
            if len(historical_flows) > 1:
                flow_mean = statistics.mean(historical_flows)
                flow_std = statistics.stdev(historical_flows)
                etf_flow_z = (current_flow - flow_mean) / flow_std if flow_std > 0 else 0
            else:
                etf_flow_z = 0
        else:
            etf_flow_z = 0
        
        # Options flow component (put/call ratio, unusual volume)
        options_flow_data = market_data.get('options_flows', {})
        ticker_options_flows = options_flow_data.get(ticker, {})
        
        put_call_ratio = ticker_options_flows.get('put_call_ratio', 1.0)
        volume_ratio = ticker_options_flows.get('volume_vs_oi_ratio', 1.0)
        
        # Convert put/call ratio to flow signal (low P/C = bullish flow)
        pc_signal = -math.log(max(0.1, put_call_ratio))  # Negative log for inverse relationship
        
        # Volume vs OI ratio signal (high ratio = new interest)
        volume_signal = math.log(max(0.1, volume_ratio))
        
        # Combine flow signals (weight ETF flow more heavily)
        combined_flow_z = (0.6 * etf_flow_z + 0.25 * pc_signal + 0.15 * volume_signal)
        
        # Cap extreme values
        return max(-3.0, min(3.0, combined_flow_z))
    
    def _get_iv_history(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, List[OHLCVData]]]
    ) -> List[float]:
        """Get historical IV data or estimate from price volatility."""
        
        # Try to get actual IV history from market data
        iv_history_data = market_data.get('iv_history', {})
        if ticker in iv_history_data:
            return iv_history_data[ticker]
        
        # Fall back to estimated IV from historical price volatility
        if historical_data and ticker in historical_data:
            price_data = historical_data[ticker]
            if len(price_data) >= 252:  # 1 year of data
                return self._estimate_iv_from_prices(price_data)
        
        return []
    
    def _estimate_iv_from_prices(self, price_data: List[OHLCVData]) -> List[float]:
        """Estimate implied volatility from historical price movements."""
        
        prices = [float(candle.close) for candle in price_data]
        iv_estimates = []
        
        # Use 20-day rolling volatility as IV proxy
        for i in range(20, len(prices)):
            recent_prices = prices[i-20:i]
            returns = []
            for j in range(1, len(recent_prices)):
                daily_return = math.log(recent_prices[j] / recent_prices[j-1])
                returns.append(daily_return)
            
            if len(returns) > 1:
                daily_vol = statistics.stdev(returns)
                annualized_vol = daily_vol * math.sqrt(252)
                iv_estimates.append(annualized_vol)
        
        return iv_estimates
    
    def _calculate_percentile(self, data: List[float], value: float) -> float:
        """Calculate percentile rank of value in data."""
        
        if not data:
            return 50.0
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Find position of value
        count_below = sum(1 for x in sorted_data if x < value)
        count_equal = sum(1 for x in sorted_data if x == value)
        
        # Calculate percentile rank
        percentile = (count_below + 0.5 * count_equal) / n * 100
        
        return percentile
    
    def _adjust_weights_for_regime(
        self,
        base_weights: ScoringWeights,
        market_regime: Optional[MarketRegimeType]
    ) -> ScoringWeights:
        """Adjust scoring weights based on market conditions."""
        
        if not market_regime:
            return base_weights
        
        # Get regime adjustments
        adjustments = self.regime_adjustments.get(market_regime, {})
        
        # Apply adjustments
        adjusted_weights = ScoringWeights(
            pop=base_weights.pop * adjustments.get('pop', 1.0),
            iv_rank=base_weights.iv_rank * adjustments.get('iv_rank', 1.0),
            momentum=base_weights.momentum * adjustments.get('momentum', 1.0),
            flow=base_weights.flow * adjustments.get('flow', 1.0),
            risk_reward=base_weights.risk_reward * adjustments.get('risk_reward', 1.0),
            liquidity=base_weights.liquidity * adjustments.get('liquidity', 1.0)
        )
        
        # Renormalize weights to sum to 1.0
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
    
    def _determine_ranking_tier(self, model_score: float, component_scores: ComponentScores) -> int:
        """Determine ranking tier based on model score and components."""
        
        if model_score >= 80:
            return 1  # Tier 1: Excellent
        elif model_score >= 65:
            return 2  # Tier 2: Good
        elif model_score >= 50:
            return 3  # Tier 3: Average
        elif model_score >= 35:
            return 4  # Tier 4: Below Average
        else:
            return 5  # Tier 5: Poor
    
    def _generate_ranking_rationale(
        self,
        scores: ComponentScores,
        weights: ScoringWeights,
        market_regime: Optional[MarketRegimeType]
    ) -> str:
        """Generate human-readable ranking rationale."""
        
        rationale_parts = []
        
        # Identify strongest components
        component_values = [
            ("POP", scores.pop_score),
            ("IV Rank", scores.iv_rank_score),
            ("Momentum", scores.momentum_score),
            ("Flow", scores.flow_score),
            ("Risk/Reward", scores.risk_reward_score),
            ("Liquidity", scores.liquidity_score)
        ]
        
        # Sort by score
        component_values.sort(key=lambda x: x[1], reverse=True)
        
        # Mention top 2 components
        if component_values[0][1] > 70:
            rationale_parts.append(f"Strong {component_values[0][0]} ({component_values[0][1]:.0f})")
        
        if component_values[1][1] > 60:
            rationale_parts.append(f"Good {component_values[1][0]} ({component_values[1][1]:.0f})")
        
        # Mention market regime if applicable
        if market_regime:
            regime_name = market_regime.value.replace('_', ' ').title()
            rationale_parts.append(f"{regime_name} conditions")
        
        # Add momentum/flow Z-scores if significant
        if abs(scores.momentum_z) > 1.5:
            direction = "positive" if scores.momentum_z > 0 else "negative"
            rationale_parts.append(f"{direction} momentum ({scores.momentum_z:.1f}σ)")
        
        if abs(scores.flow_z) > 1.5:
            direction = "positive" if scores.flow_z > 0 else "negative"
            rationale_parts.append(f"{direction} flow ({scores.flow_z:.1f}σ)")
        
        return "; ".join(rationale_parts) if rationale_parts else "Standard scoring criteria applied"