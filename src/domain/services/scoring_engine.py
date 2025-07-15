"""
Scoring engine service for ranking and evaluating trade candidates.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import math
import logging

from ..entities.strategy import Strategy
from ...data.models.trades import TradeCandidate, StrategyType
from ...data.models.market_data import TechnicalIndicators, FundamentalData, EconomicIndicator

logger = logging.getLogger(__name__)


class ScoringFactorType(str, Enum):
    """Types of scoring factors."""
    PROBABILITY_OF_PROFIT = "probability_of_profit"
    RETURN_ON_CAPITAL = "return_on_capital"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    IMPLIED_VOLATILITY_RANK = "iv_rank"
    MOMENTUM_SCORE = "momentum_score"
    FLOW_SCORE = "flow_score"
    LIQUIDITY_SCORE = "liquidity_score"
    TIME_DECAY_SCORE = "time_decay_score"
    TECHNICAL_SCORE = "technical_score"
    FUNDAMENTAL_SCORE = "fundamental_score"
    MARKET_REGIME_SCORE = "market_regime_score"


@dataclass
class ScoringFactor:
    """Individual scoring factor with weight and calculation method."""
    
    factor_type: ScoringFactorType
    weight: float
    enabled: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    normalization_method: str = "linear"  # linear, logarithmic, sigmoid
    
    def __post_init__(self):
        if not (0 <= self.weight <= 1):
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


@dataclass
class ScoringResult:
    """Result of scoring a trade candidate."""
    
    trade_candidate: TradeCandidate
    total_score: float
    factor_scores: Dict[ScoringFactorType, float]
    normalized_scores: Dict[ScoringFactorType, float]
    warnings: List[str]
    
    def __post_init__(self):
        if not self.warnings:
            self.warnings = []


class ScoringModel:
    """
    Configurable scoring model for trade evaluation.
    
    Allows customization of scoring factors, weights, and calculation methods
    to adapt to different market conditions and trading strategies.
    """
    
    def __init__(self, name: str, factors: List[ScoringFactor]):
        self.name = name
        self.factors = {factor.factor_type: factor for factor in factors}
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0."""
        total_weight = sum(factor.weight for factor in self.factors.values() if factor.enabled)
        if not (0.95 <= total_weight <= 1.05):  # Allow small rounding errors
            logger.warning(f"Scoring model '{self.name}' weights sum to {total_weight:.3f}, not 1.0")
    
    def get_enabled_factors(self) -> List[ScoringFactor]:
        """Get list of enabled scoring factors."""
        return [factor for factor in self.factors.values() if factor.enabled]
    
    def update_factor_weight(self, factor_type: ScoringFactorType, new_weight: float):
        """Update weight for a specific factor."""
        if factor_type in self.factors:
            self.factors[factor_type].weight = new_weight
            self._validate_weights()
    
    def enable_factor(self, factor_type: ScoringFactorType, enabled: bool = True):
        """Enable or disable a scoring factor."""
        if factor_type in self.factors:
            self.factors[factor_type].enabled = enabled


class TradeScorer:
    """
    Service for scoring individual trade candidates.
    
    Calculates scores for various factors and combines them according
    to the specified scoring model.
    """
    
    def __init__(self, scoring_model: ScoringModel):
        self.scoring_model = scoring_model
        self._factor_calculators = {
            ScoringFactorType.PROBABILITY_OF_PROFIT: self._calculate_pop_score,
            ScoringFactorType.RETURN_ON_CAPITAL: self._calculate_roc_score,
            ScoringFactorType.RISK_REWARD_RATIO: self._calculate_risk_reward_score,
            ScoringFactorType.IMPLIED_VOLATILITY_RANK: self._calculate_iv_rank_score,
            ScoringFactorType.MOMENTUM_SCORE: self._calculate_momentum_score,
            ScoringFactorType.FLOW_SCORE: self._calculate_flow_score,
            ScoringFactorType.LIQUIDITY_SCORE: self._calculate_liquidity_score,
            ScoringFactorType.TIME_DECAY_SCORE: self._calculate_time_decay_score,
            ScoringFactorType.TECHNICAL_SCORE: self._calculate_technical_score,
            ScoringFactorType.FUNDAMENTAL_SCORE: self._calculate_fundamental_score,
            ScoringFactorType.MARKET_REGIME_SCORE: self._calculate_market_regime_score,
        }
    
    def score_trade(self, 
                   trade_candidate: TradeCandidate,
                   technical_indicators: Optional[TechnicalIndicators] = None,
                   fundamental_data: Optional[FundamentalData] = None,
                   market_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a trade candidate using the configured scoring model.
        
        Args:
            trade_candidate: Trade to score
            technical_indicators: Technical analysis data
            fundamental_data: Fundamental analysis data
            market_context: Additional market context data
            
        Returns:
            Comprehensive scoring result
        """
        factor_scores = {}
        normalized_scores = {}
        warnings = []
        
        # Calculate raw scores for each enabled factor
        for factor in self.scoring_model.get_enabled_factors():
            try:
                calculator = self._factor_calculators.get(factor.factor_type)
                if calculator:
                    raw_score = calculator(
                        trade_candidate, technical_indicators, fundamental_data, market_context
                    )
                    
                    if raw_score is not None:
                        factor_scores[factor.factor_type] = raw_score
                        normalized_score = self._normalize_score(raw_score, factor)
                        normalized_scores[factor.factor_type] = normalized_score
                    else:
                        warnings.append(f"Could not calculate {factor.factor_type.value}")
                else:
                    warnings.append(f"No calculator for {factor.factor_type.value}")
                    
            except Exception as e:
                logger.warning(f"Error calculating {factor.factor_type.value}: {e}")
                warnings.append(f"Error in {factor.factor_type.value}: {str(e)}")
        
        # Calculate weighted total score
        total_score = 0.0
        total_weight = 0.0
        
        for factor in self.scoring_model.get_enabled_factors():
            if factor.factor_type in normalized_scores:
                weighted_score = normalized_scores[factor.factor_type] * factor.weight
                total_score += weighted_score
                total_weight += factor.weight
        
        # Normalize by total weight if not exactly 1.0
        if total_weight > 0 and total_weight != 1.0:
            total_score = total_score / total_weight
        
        return ScoringResult(
            trade_candidate=trade_candidate,
            total_score=total_score,
            factor_scores=factor_scores,
            normalized_scores=normalized_scores,
            warnings=warnings
        )
    
    def _normalize_score(self, raw_score: float, factor: ScoringFactor) -> float:
        """Normalize raw score to 0-1 range."""
        if factor.normalization_method == "linear":
            if factor.min_value is not None and factor.max_value is not None:
                if factor.max_value == factor.min_value:
                    return 1.0
                normalized = (raw_score - factor.min_value) / (factor.max_value - factor.min_value)
                return max(0.0, min(1.0, normalized))
            else:
                # Default linear normalization for 0-1 range
                return max(0.0, min(1.0, raw_score))
        
        elif factor.normalization_method == "sigmoid":
            # Sigmoid normalization
            return 1 / (1 + math.exp(-raw_score))
        
        elif factor.normalization_method == "logarithmic":
            # Logarithmic normalization
            if raw_score <= 0:
                return 0.0
            log_score = math.log(1 + raw_score)
            return min(1.0, log_score / 5.0)  # Normalize to roughly 0-1 range
        
        else:
            return raw_score
    
    def _calculate_pop_score(self, 
                           trade_candidate: TradeCandidate,
                           technical_indicators: Optional[TechnicalIndicators],
                           fundamental_data: Optional[FundamentalData],
                           market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate probability of profit score."""
        return trade_candidate.probability_of_profit
    
    def _calculate_roc_score(self,
                           trade_candidate: TradeCandidate,
                           technical_indicators: Optional[TechnicalIndicators],
                           fundamental_data: Optional[FundamentalData],
                           market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate return on capital score."""
        return trade_candidate.strategy.calculate_return_on_capital()
    
    def _calculate_risk_reward_score(self,
                                   trade_candidate: TradeCandidate,
                                   technical_indicators: Optional[TechnicalIndicators],
                                   fundamental_data: Optional[FundamentalData],
                                   market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate risk-reward ratio score (inverted - lower is better)."""
        risk_reward = trade_candidate.strategy.get_risk_reward_ratio()
        if risk_reward is None or risk_reward <= 0:
            return None
        
        # Invert the ratio (lower risk/reward is better)
        # Score ranges from 0 to 1, where 1 is best
        optimal_ratio = 2.0  # Optimal risk/reward ratio
        if risk_reward <= optimal_ratio:
            return 1.0
        else:
            # Exponential decay for ratios above optimal
            return math.exp(-(risk_reward - optimal_ratio) / optimal_ratio)
    
    def _calculate_iv_rank_score(self,
                               trade_candidate: TradeCandidate,
                               technical_indicators: Optional[TechnicalIndicators],
                               fundamental_data: Optional[FundamentalData],
                               market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate implied volatility rank score."""
        return trade_candidate.iv_rank
    
    def _calculate_momentum_score(self,
                                trade_candidate: TradeCandidate,
                                technical_indicators: Optional[TechnicalIndicators],
                                fundamental_data: Optional[FundamentalData],
                                market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate momentum score from technical indicators."""
        if not technical_indicators:
            return trade_candidate.momentum_z_score
        
        # Combine multiple momentum indicators
        momentum_scores = []
        
        # Short-term momentum
        if technical_indicators.momentum_1d is not None:
            momentum_scores.append(technical_indicators.momentum_1d / 10.0)  # Normalize
        
        if technical_indicators.momentum_5d is not None:
            momentum_scores.append(technical_indicators.momentum_5d / 20.0)  # Normalize
        
        # RSI momentum
        if technical_indicators.rsi_14 is not None:
            rsi = technical_indicators.rsi_14
            # Convert RSI to momentum score (50 is neutral)
            rsi_momentum = (rsi - 50) / 50  # Range: -1 to 1
            momentum_scores.append(rsi_momentum)
        
        if momentum_scores:
            return sum(momentum_scores) / len(momentum_scores)
        
        return trade_candidate.momentum_z_score
    
    def _calculate_flow_score(self,
                            trade_candidate: TradeCandidate,
                            technical_indicators: Optional[TechnicalIndicators],
                            fundamental_data: Optional[FundamentalData],
                            market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate flow score."""
        return trade_candidate.flow_z_score
    
    def _calculate_liquidity_score(self,
                                 trade_candidate: TradeCandidate,
                                 technical_indicators: Optional[TechnicalIndicators],
                                 fundamental_data: Optional[FundamentalData],
                                 market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate liquidity score based on option liquidity."""
        if not trade_candidate.liquidity_score:
            # Calculate from strategy legs
            total_score = 0.0
            leg_count = 0
            
            for leg in trade_candidate.strategy.legs:
                option = leg.option
                leg_score = 0.0
                
                # Volume component (0-0.4)
                volume = option.volume or 0
                if volume >= 100:
                    leg_score += 0.4
                elif volume >= 50:
                    leg_score += 0.3
                elif volume >= 10:
                    leg_score += 0.2
                
                # Open interest component (0-0.4)
                oi = option.open_interest or 0
                if oi >= 1000:
                    leg_score += 0.4
                elif oi >= 500:
                    leg_score += 0.3
                elif oi >= 100:
                    leg_score += 0.2
                
                # Spread component (0-0.2)
                spread_pct = option.bid_ask_spread_percent or 1.0
                if spread_pct <= 0.05:
                    leg_score += 0.2
                elif spread_pct <= 0.15:
                    leg_score += 0.15
                elif spread_pct <= 0.35:
                    leg_score += 0.1
                
                total_score += leg_score
                leg_count += 1
            
            return total_score / leg_count if leg_count > 0 else 0.0
        
        return trade_candidate.liquidity_score
    
    def _calculate_time_decay_score(self,
                                  trade_candidate: TradeCandidate,
                                  technical_indicators: Optional[TechnicalIndicators],
                                  fundamental_data: Optional[FundamentalData],
                                  market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate time decay score based on theta and days to expiration."""
        strategy = trade_candidate.strategy
        
        # Get net theta
        net_theta = strategy.calculate_net_greeks().theta
        days_to_exp = strategy.get_days_to_expiration()
        
        if net_theta is None or days_to_exp <= 0:
            return None
        
        # For credit strategies, positive theta is good
        # For debit strategies, negative theta is bad
        if strategy.is_credit_strategy():
            # Positive theta is good for credit strategies
            theta_score = max(0, net_theta / 100.0)  # Normalize
        else:
            # Negative theta is bad for debit strategies
            theta_score = max(0, -net_theta / 100.0)  # Invert and normalize
        
        # Days to expiration factor (sweet spot around 20-35 days)
        if 20 <= days_to_exp <= 35:
            dte_factor = 1.0
        elif 15 <= days_to_exp < 20 or 35 < days_to_exp <= 45:
            dte_factor = 0.8
        elif 7 <= days_to_exp < 15:
            dte_factor = 0.6
        else:
            dte_factor = 0.4
        
        return theta_score * dte_factor
    
    def _calculate_technical_score(self,
                                 trade_candidate: TradeCandidate,
                                 technical_indicators: Optional[TechnicalIndicators],
                                 fundamental_data: Optional[FundamentalData],
                                 market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate technical analysis score."""
        if not technical_indicators:
            return None
        
        scores = []
        
        # Trend strength (moving averages)
        if all(ma is not None for ma in [technical_indicators.sma_20, technical_indicators.sma_50]):
            sma_20 = float(technical_indicators.sma_20)
            sma_50 = float(technical_indicators.sma_50)
            
            # Trend score based on MA relationship
            if sma_20 > sma_50:
                trend_score = 0.6  # Uptrend
            else:
                trend_score = 0.4  # Downtrend
            
            scores.append(trend_score)
        
        # Volatility score
        if technical_indicators.hv_30 is not None:
            hv = technical_indicators.hv_30
            # Moderate volatility is preferred (15-30%)
            if 15 <= hv <= 30:
                vol_score = 1.0
            elif 10 <= hv < 15 or 30 < hv <= 40:
                vol_score = 0.8
            elif 5 <= hv < 10 or 40 < hv <= 60:
                vol_score = 0.6
            else:
                vol_score = 0.4
            
            scores.append(vol_score)
        
        # RSI score (avoid extreme readings)
        if technical_indicators.rsi_14 is not None:
            rsi = technical_indicators.rsi_14
            if 40 <= rsi <= 60:
                rsi_score = 1.0  # Neutral territory
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                rsi_score = 0.8
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                rsi_score = 0.6
            else:
                rsi_score = 0.4  # Extreme readings
            
            scores.append(rsi_score)
        
        return sum(scores) / len(scores) if scores else None
    
    def _calculate_fundamental_score(self,
                                   trade_candidate: TradeCandidate,
                                   technical_indicators: Optional[TechnicalIndicators],
                                   fundamental_data: Optional[FundamentalData],
                                   market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate fundamental analysis score."""
        if not fundamental_data:
            return None
        
        scores = []
        
        # Profitability score
        if fundamental_data.net_margin is not None:
            margin = fundamental_data.net_margin
            if margin > 0.15:  # > 15%
                margin_score = 1.0
            elif margin > 0.10:  # > 10%
                margin_score = 0.8
            elif margin > 0.05:  # > 5%
                margin_score = 0.6
            elif margin > 0:
                margin_score = 0.4
            else:
                margin_score = 0.2  # Unprofitable
            
            scores.append(margin_score)
        
        # Valuation score (P/E ratio)
        if fundamental_data.pe_ratio is not None and fundamental_data.pe_ratio > 0:
            pe = fundamental_data.pe_ratio
            if 10 <= pe <= 20:
                pe_score = 1.0  # Reasonable valuation
            elif 5 <= pe < 10 or 20 < pe <= 30:
                pe_score = 0.8
            elif pe < 5 or 30 < pe <= 50:
                pe_score = 0.6
            else:
                pe_score = 0.4  # Extreme valuation
            
            scores.append(pe_score)
        
        # Growth score (PEG ratio)
        if fundamental_data.peg_ratio is not None and fundamental_data.peg_ratio > 0:
            peg = fundamental_data.peg_ratio
            if 0.5 <= peg <= 1.5:
                peg_score = 1.0  # Good growth at reasonable price
            elif 0.2 <= peg < 0.5 or 1.5 < peg <= 2.5:
                peg_score = 0.8
            else:
                peg_score = 0.6
            
            scores.append(peg_score)
        
        return sum(scores) / len(scores) if scores else None
    
    def _calculate_market_regime_score(self,
                                     trade_candidate: TradeCandidate,
                                     technical_indicators: Optional[TechnicalIndicators],
                                     fundamental_data: Optional[FundamentalData],
                                     market_context: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate market regime score based on current market conditions."""
        if not market_context:
            return None
        
        # This would analyze current market regime (bull, bear, sideways)
        # and score strategies based on their suitability for the regime
        
        market_regime = market_context.get('regime', 'neutral')
        strategy_type = trade_candidate.strategy_type
        
        # Strategy suitability by market regime
        regime_scores = {
            'bull': {
                StrategyType.PUT_CREDIT_SPREAD: 0.9,
                StrategyType.CALL_CREDIT_SPREAD: 0.3,
                StrategyType.IRON_CONDOR: 0.5,
                StrategyType.COVERED_CALL: 0.7,
            },
            'bear': {
                StrategyType.PUT_CREDIT_SPREAD: 0.3,
                StrategyType.CALL_CREDIT_SPREAD: 0.9,
                StrategyType.IRON_CONDOR: 0.5,
                StrategyType.COVERED_CALL: 0.4,
            },
            'neutral': {
                StrategyType.PUT_CREDIT_SPREAD: 0.6,
                StrategyType.CALL_CREDIT_SPREAD: 0.6,
                StrategyType.IRON_CONDOR: 0.9,
                StrategyType.COVERED_CALL: 0.7,
            }
        }
        
        return regime_scores.get(market_regime, {}).get(strategy_type, 0.5)


class ScoringEngine:
    """
    Main scoring engine that orchestrates trade evaluation and ranking.
    
    Provides high-level interface for scoring multiple trade candidates
    and ranking them according to specified criteria.
    """
    
    def __init__(self, default_model: Optional[ScoringModel] = None):
        self.default_model = default_model or self._create_default_model()
        self.trade_scorer = TradeScorer(self.default_model)
    
    def score_trades(self,
                    trade_candidates: List[TradeCandidate],
                    technical_data: Optional[Dict[str, TechnicalIndicators]] = None,
                    fundamental_data: Optional[Dict[str, FundamentalData]] = None,
                    market_context: Optional[Dict[str, Any]] = None) -> List[ScoringResult]:
        """
        Score multiple trade candidates.
        
        Args:
            trade_candidates: List of trades to score
            technical_data: Technical indicators by symbol
            fundamental_data: Fundamental data by symbol
            market_context: Market context information
            
        Returns:
            List of scoring results, sorted by score (highest first)
        """
        results = []
        
        for candidate in trade_candidates:
            symbol = candidate.underlying
            
            # Get data for this symbol
            tech_indicators = technical_data.get(symbol) if technical_data else None
            fund_data = fundamental_data.get(symbol) if fundamental_data else None
            
            # Score the trade
            result = self.trade_scorer.score_trade(
                candidate, tech_indicators, fund_data, market_context
            )
            
            # Update the trade candidate with the score
            candidate.model_score = result.total_score
            
            results.append(result)
        
        # Sort by total score (highest first)
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        return results
    
    def rank_trades(self, scoring_results: List[ScoringResult]) -> List[TradeCandidate]:
        """
        Rank trades based on scoring results with tie-breaking.
        
        Args:
            scoring_results: List of scoring results
            
        Returns:
            Ranked list of trade candidates
        """
        # Sort with tie-breaking rules
        def sort_key(result: ScoringResult):
            candidate = result.trade_candidate
            
            # Primary: total score
            primary = result.total_score
            
            # Secondary: momentum Z-score
            secondary = candidate.momentum_z_score or 0
            
            # Tertiary: flow Z-score
            tertiary = candidate.flow_z_score or 0
            
            return (primary, secondary, tertiary)
        
        sorted_results = sorted(scoring_results, key=sort_key, reverse=True)
        
        # Extract trade candidates and assign ranks
        ranked_trades = []
        for i, result in enumerate(sorted_results):
            candidate = result.trade_candidate
            candidate.rank = i + 1
            ranked_trades.append(candidate)
        
        return ranked_trades
    
    def update_scoring_model(self, new_model: ScoringModel):
        """Update the scoring model."""
        self.default_model = new_model
        self.trade_scorer = TradeScorer(new_model)
    
    def _create_default_model(self) -> ScoringModel:
        """Create default scoring model."""
        factors = [
            ScoringFactor(
                factor_type=ScoringFactorType.PROBABILITY_OF_PROFIT,
                weight=0.25,
                min_value=0.0,
                max_value=1.0
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.RETURN_ON_CAPITAL,
                weight=0.20,
                min_value=0.0,
                max_value=2.0  # 200% annualized return cap
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.RISK_REWARD_RATIO,
                weight=0.15,
                normalization_method="sigmoid"
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.IMPLIED_VOLATILITY_RANK,
                weight=0.10,
                min_value=0.0,
                max_value=1.0
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.LIQUIDITY_SCORE,
                weight=0.10,
                min_value=0.0,
                max_value=1.0
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.TIME_DECAY_SCORE,
                weight=0.10,
                min_value=0.0,
                max_value=1.0
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.MOMENTUM_SCORE,
                weight=0.05,
                normalization_method="sigmoid"
            ),
            ScoringFactor(
                factor_type=ScoringFactorType.FLOW_SCORE,
                weight=0.05,
                normalization_method="sigmoid"
            )
        ]
        
        return ScoringModel("default", factors)


# Predefined scoring models for different scenarios
class ScoringModelFactory:
    """Factory for creating predefined scoring models."""
    
    @staticmethod
    def create_conservative_model() -> ScoringModel:
        """Create scoring model optimized for conservative trading."""
        factors = [
            ScoringFactor(ScoringFactorType.PROBABILITY_OF_PROFIT, weight=0.35),
            ScoringFactor(ScoringFactorType.RISK_REWARD_RATIO, weight=0.25),
            ScoringFactor(ScoringFactorType.LIQUIDITY_SCORE, weight=0.20),
            ScoringFactor(ScoringFactorType.RETURN_ON_CAPITAL, weight=0.20),
        ]
        return ScoringModel("conservative", factors)
    
    @staticmethod
    def create_aggressive_model() -> ScoringModel:
        """Create scoring model optimized for aggressive trading."""
        factors = [
            ScoringFactor(ScoringFactorType.RETURN_ON_CAPITAL, weight=0.30),
            ScoringFactor(ScoringFactorType.MOMENTUM_SCORE, weight=0.25),
            ScoringFactor(ScoringFactorType.PROBABILITY_OF_PROFIT, weight=0.20),
            ScoringFactor(ScoringFactorType.IMPLIED_VOLATILITY_RANK, weight=0.15),
            ScoringFactor(ScoringFactorType.FLOW_SCORE, weight=0.10),
        ]
        return ScoringModel("aggressive", factors)
    
    @staticmethod
    def create_income_model() -> ScoringModel:
        """Create scoring model optimized for income generation."""
        factors = [
            ScoringFactor(ScoringFactorType.TIME_DECAY_SCORE, weight=0.30),
            ScoringFactor(ScoringFactorType.PROBABILITY_OF_PROFIT, weight=0.25),
            ScoringFactor(ScoringFactorType.RETURN_ON_CAPITAL, weight=0.20),
            ScoringFactor(ScoringFactorType.LIQUIDITY_SCORE, weight=0.15),
            ScoringFactor(ScoringFactorType.IMPLIED_VOLATILITY_RANK, weight=0.10),
        ]
        return ScoringModel("income", factors)