"""
Dynamic constraint adjustment system based on market conditions and regimes.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

from ...data.models.options import OptionQuote, OptionType
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData, OHLCVData
from ...data.models.trades import TradeCandidate, StrategyDefinition
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, CalculationError
)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS_MODE = "crisis_mode"


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    EXTREMELY_LOW = "extremely_low"      # VIX < 12
    LOW = "low"                          # VIX 12-16
    NORMAL = "normal"                    # VIX 16-25
    ELEVATED = "elevated"                # VIX 25-35
    HIGH = "high"                        # VIX 35-50
    EXTREME = "extreme"                  # VIX > 50


class LiquidityRegime(Enum):
    """Market liquidity classifications."""
    ABUNDANT = "abundant"
    NORMAL = "normal"
    CONSTRAINED = "constrained"
    STRESSED = "stressed"


@dataclass
class MarketConditions:
    """Current market condition assessment."""
    # Regime classifications
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    
    # Market metrics
    vix_level: float
    term_structure_slope: float  # VIX9D vs VIX
    skew_level: float           # Put/call skew
    correlation_level: float    # Average stock correlation
    
    # Trend metrics
    trend_strength: float       # 0-1 scale
    trend_direction: str        # "up", "down", "sideways"
    momentum_score: float       # -3 to +3 Z-score
    
    # Economic indicators
    earnings_season: bool
    fed_meeting_proximity: Optional[int] = None  # Days to next Fed meeting
    economic_data_proximity: Optional[int] = None  # Days to major economic release
    
    # Risk metrics
    credit_spreads: Optional[float] = None
    yield_curve_slope: Optional[float] = None
    dollar_strength: Optional[float] = None


@dataclass
class ConstraintAdjustment:
    """Adjustment to apply to constraints based on market conditions."""
    constraint_name: str
    original_value: Any
    adjusted_value: Any
    adjustment_factor: float
    justification: str


@dataclass
class DynamicConstraintSet:
    """Dynamic constraint parameters adjusted for market conditions."""
    # Quote and data constraints
    max_quote_age_minutes: int = 10
    min_data_quality_score: float = 0.8
    
    # Probability and risk constraints
    min_probability_of_profit: float = 0.65
    min_credit_to_max_loss_ratio: float = 0.33
    max_loss_per_trade: Decimal = Decimal('500')
    
    # Liquidity constraints
    min_open_interest: int = 50
    max_bid_ask_spread_pct: float = 0.05
    min_daily_volume: int = 10
    
    # Portfolio constraints
    max_delta_per_100k: float = 0.30
    min_vega_per_100k: float = -0.05
    max_trades_per_sector: int = 2
    max_correlation: float = 0.70
    
    # Position sizing constraints
    max_position_size_pct: float = 0.02  # 2% of NAV
    min_position_size: Decimal = Decimal('100')
    
    # Strategy specific constraints
    max_dte: int = 45
    min_dte: int = 7
    
    # Adjustments applied
    adjustments: List[ConstraintAdjustment] = field(default_factory=list)


class DynamicConstraintManager:
    """
    Dynamic constraint adjustment system that modifies filtering parameters
    based on real-time market conditions, volatility regimes, and economic events.
    
    Features:
    - Market regime detection with automatic parameter adjustment
    - Volatility-based constraint scaling
    - Liquidity condition monitoring with adaptive filters
    - Economic event proximity analysis
    - Crisis mode with emergency risk reduction protocols
    - Earnings announcement proximity filters
    - Fed meeting and economic data event handling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Base constraint set (conservative defaults)
        self.base_constraints = DynamicConstraintSet()
        
        # Regime-specific adjustment factors
        self.regime_adjustments = self._initialize_regime_adjustments()
        
        # Economic event impact factors
        self.event_adjustments = {
            'earnings_season': {
                'min_probability_of_profit': 1.05,  # Require 5% higher POP
                'max_quote_age_minutes': 0.5,       # Require fresher quotes
                'min_open_interest': 1.5,           # Require 50% higher OI
            },
            'fed_meeting_week': {
                'min_probability_of_profit': 1.03,
                'max_delta_per_100k': 0.8,          # Reduce delta exposure
                'max_loss_per_trade': 0.8,          # Reduce position sizes
            },
            'economic_data_day': {
                'min_probability_of_profit': 1.02,
                'max_quote_age_minutes': 0.3,       # Very fresh quotes
            }
        }
    
    @handle_errors(operation_name="assess_market_conditions")
    def assess_current_market_conditions(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, List[OHLCVData]]] = None
    ) -> MarketConditions:
        """
        Assess current market conditions across multiple dimensions.
        
        Args:
            market_data: Current market data dictionary
            historical_data: Historical price data for trend analysis
            
        Returns:
            Comprehensive market condition assessment
        """
        self.logger.info("Assessing current market conditions for constraint adjustment")
        
        # Extract VIX level
        vix_level = market_data.get('vix', 20.0)  # Default if not available
        
        # Determine volatility regime
        volatility_regime = self._classify_volatility_regime(vix_level)
        
        # Determine market regime
        market_regime = self._classify_market_regime(market_data, historical_data)
        
        # Determine liquidity regime
        liquidity_regime = self._classify_liquidity_regime(market_data)
        
        # Calculate term structure and skew
        term_structure_slope = self._calculate_term_structure_slope(market_data)
        skew_level = self._calculate_skew_level(market_data)
        correlation_level = self._calculate_correlation_level(market_data)
        
        # Analyze trend characteristics
        trend_analysis = self._analyze_trend_characteristics(historical_data)
        
        # Check for economic events
        earnings_season = self._is_earnings_season(market_data)
        fed_meeting_proximity = self._get_fed_meeting_proximity()
        economic_data_proximity = self._get_economic_data_proximity()
        
        return MarketConditions(
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            liquidity_regime=liquidity_regime,
            vix_level=vix_level,
            term_structure_slope=term_structure_slope,
            skew_level=skew_level,
            correlation_level=correlation_level,
            trend_strength=trend_analysis['strength'],
            trend_direction=trend_analysis['direction'],
            momentum_score=trend_analysis['momentum'],
            earnings_season=earnings_season,
            fed_meeting_proximity=fed_meeting_proximity,
            economic_data_proximity=economic_data_proximity
        )
    
    @handle_errors(operation_name="adjust_constraints")
    def adjust_constraints_for_conditions(
        self,
        market_conditions: MarketConditions,
        base_constraints: Optional[DynamicConstraintSet] = None
    ) -> DynamicConstraintSet:
        """
        Adjust constraint parameters based on current market conditions.
        
        Args:
            market_conditions: Current market condition assessment
            base_constraints: Base constraint set to adjust
            
        Returns:
            Adjusted constraint set with justifications
        """
        if base_constraints is None:
            base_constraints = self.base_constraints
        
        # Create copy for adjustment
        adjusted = DynamicConstraintSet(**base_constraints.__dict__)
        adjusted.adjustments = []
        
        # Apply regime-based adjustments
        self._apply_volatility_adjustments(adjusted, market_conditions)
        self._apply_market_regime_adjustments(adjusted, market_conditions)
        self._apply_liquidity_adjustments(adjusted, market_conditions)
        self._apply_event_adjustments(adjusted, market_conditions)
        self._apply_crisis_mode_adjustments(adjusted, market_conditions)
        
        self.logger.info(f"Applied {len(adjusted.adjustments)} constraint adjustments")
        return adjusted
    
    def _initialize_regime_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Initialize regime-specific adjustment factors."""
        
        return {
            # Volatility regime adjustments
            'extremely_low_vol': {
                'min_probability_of_profit': 0.95,  # Lower POP acceptable
                'max_loss_per_trade': 1.2,          # Larger positions OK
                'min_credit_to_max_loss_ratio': 0.9, # Lower ratio acceptable
            },
            'low_vol': {
                'min_probability_of_profit': 0.98,
                'max_loss_per_trade': 1.1,
            },
            'high_vol': {
                'min_probability_of_profit': 1.05,  # Require higher POP
                'max_loss_per_trade': 0.8,          # Smaller positions
                'max_delta_per_100k': 0.7,          # Reduce delta exposure
                'min_open_interest': 1.5,           # Higher liquidity requirements
            },
            'extreme_vol': {
                'min_probability_of_profit': 1.1,
                'max_loss_per_trade': 0.6,
                'max_delta_per_100k': 0.5,
                'min_open_interest': 2.0,
                'max_quote_age_minutes': 0.5,       # Much fresher quotes
            },
            
            # Market regime adjustments
            'bull_market': {
                'min_probability_of_profit': 0.95,
                'max_delta_per_100k': 1.2,          # Allow more delta in bull market
            },
            'bear_market': {
                'min_probability_of_profit': 1.05,
                'max_delta_per_100k': 0.8,          # Reduce delta in bear market
                'max_loss_per_trade': 0.9,
            },
            'crisis_mode': {
                'min_probability_of_profit': 1.15,
                'max_loss_per_trade': 0.5,
                'max_delta_per_100k': 0.3,
                'min_open_interest': 3.0,
                'max_quote_age_minutes': 0.2,
            },
            
            # Liquidity regime adjustments
            'constrained_liquidity': {
                'min_open_interest': 1.5,
                'max_bid_ask_spread_pct': 0.8,
                'min_daily_volume': 1.5,
            },
            'stressed_liquidity': {
                'min_open_interest': 2.0,
                'max_bid_ask_spread_pct': 0.6,
                'min_daily_volume': 2.0,
                'max_quote_age_minutes': 0.5,
            }
        }
    
    def _classify_volatility_regime(self, vix_level: float) -> VolatilityRegime:
        """Classify volatility regime based on VIX level."""
        
        if vix_level < 12:
            return VolatilityRegime.EXTREMELY_LOW
        elif vix_level < 16:
            return VolatilityRegime.LOW
        elif vix_level < 25:
            return VolatilityRegime.NORMAL
        elif vix_level < 35:
            return VolatilityRegime.ELEVATED
        elif vix_level < 50:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _classify_market_regime(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, List[OHLCVData]]]
    ) -> MarketRegime:
        """Classify overall market regime."""
        
        vix_level = market_data.get('vix', 20.0)
        
        # Crisis mode detection
        if vix_level > 40:
            return MarketRegime.CRISIS_MODE
        
        # High/low volatility regimes
        if vix_level > 30:
            return MarketRegime.HIGH_VOLATILITY
        elif vix_level < 15:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based classification (simplified)
        if historical_data:
            spy_data = historical_data.get('SPY', [])
            if len(spy_data) >= 20:
                recent_prices = [float(candle.close) for candle in spy_data[-20:]]
                if len(recent_prices) >= 2:
                    trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    if trend > 0.05:  # 5% up trend
                        return MarketRegime.BULL_MARKET
                    elif trend < -0.05:  # 5% down trend
                        return MarketRegime.BEAR_MARKET
        
        return MarketRegime.SIDEWAYS_MARKET
    
    def _classify_liquidity_regime(self, market_data: Dict[str, Any]) -> LiquidityRegime:
        """Classify market liquidity regime."""
        
        # Simplified classification based on available metrics
        vix_level = market_data.get('vix', 20.0)
        
        # Use VIX as proxy for liquidity stress
        if vix_level > 35:
            return LiquidityRegime.STRESSED
        elif vix_level > 25:
            return LiquidityRegime.CONSTRAINED
        elif vix_level < 15:
            return LiquidityRegime.ABUNDANT
        else:
            return LiquidityRegime.NORMAL
    
    def _calculate_term_structure_slope(self, market_data: Dict[str, Any]) -> float:
        """Calculate VIX term structure slope."""
        
        # Simplified calculation
        vix = market_data.get('vix', 20.0)
        vix9d = market_data.get('vix9d', vix)  # Default to VIX if not available
        
        return vix9d - vix
    
    def _calculate_skew_level(self, market_data: Dict[str, Any]) -> float:
        """Calculate put/call skew level."""
        
        # Simplified skew calculation
        # In practice, would calculate from option chain data
        return market_data.get('skew', 0.0)
    
    def _calculate_correlation_level(self, market_data: Dict[str, Any]) -> float:
        """Calculate average stock correlation level."""
        
        # Simplified correlation measure
        # In practice, would calculate from individual stock movements
        return market_data.get('correlation', 0.5)
    
    def _analyze_trend_characteristics(
        self,
        historical_data: Optional[Dict[str, List[OHLCVData]]]
    ) -> Dict[str, Any]:
        """Analyze trend strength, direction, and momentum."""
        
        if not historical_data:
            return {
                'strength': 0.5,
                'direction': 'sideways',
                'momentum': 0.0
            }
        
        # Use SPY as market proxy
        spy_data = historical_data.get('SPY', [])
        if len(spy_data) < 20:
            return {
                'strength': 0.5,
                'direction': 'sideways',
                'momentum': 0.0
            }
        
        prices = [float(candle.close) for candle in spy_data[-20:]]
        
        # Calculate trend strength (R-squared of linear regression)
        x = list(range(len(prices)))
        mean_x = sum(x) / len(x)
        mean_y = sum(prices) / len(prices)
        
        numerator = sum((x[i] - mean_x) * (prices[i] - mean_y) for i in range(len(prices)))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(len(prices)))
        denominator_y = sum((prices[i] - mean_y) ** 2 for i in range(len(prices)))
        
        if denominator_x == 0 or denominator_y == 0:
            correlation = 0
        else:
            correlation = numerator / math.sqrt(denominator_x * denominator_y)
        
        trend_strength = abs(correlation)
        
        # Calculate direction
        if len(prices) >= 2:
            total_change = (prices[-1] - prices[0]) / prices[0]
            if total_change > 0.02:  # 2% threshold
                direction = 'up'
            elif total_change < -0.02:
                direction = 'down'
            else:
                direction = 'sideways'
        else:
            direction = 'sideways'
        
        # Calculate momentum (simplified)
        if len(prices) >= 10:
            recent_change = (prices[-1] - prices[-5]) / prices[-5]
            momentum = recent_change * 10  # Scale to approximate Z-score
        else:
            momentum = 0.0
        
        return {
            'strength': trend_strength,
            'direction': direction,
            'momentum': momentum
        }
    
    def _is_earnings_season(self, market_data: Dict[str, Any]) -> bool:
        """Check if currently in earnings season."""
        
        # Simplified earnings season detection
        # In practice, would check actual earnings calendar
        current_month = datetime.now().month
        
        # Earnings seasons: Jan, Apr, Jul, Oct
        earnings_months = [1, 4, 7, 10]
        
        return current_month in earnings_months
    
    def _get_fed_meeting_proximity(self) -> Optional[int]:
        """Get days until next Fed meeting."""
        
        # Simplified - would use actual Fed calendar
        # Return None if no meeting soon, or days if within 2 weeks
        return None
    
    def _get_economic_data_proximity(self) -> Optional[int]:
        """Get days until major economic data release."""
        
        # Simplified - would use actual economic calendar
        # Return None if no major release soon, or days if within 1 week
        return None
    
    def _apply_volatility_adjustments(
        self,
        constraints: DynamicConstraintSet,
        conditions: MarketConditions
    ):
        """Apply volatility regime adjustments."""
        
        regime_key = f"{conditions.volatility_regime.value}_vol"
        adjustments = self.regime_adjustments.get(regime_key, {})
        
        for constraint_name, factor in adjustments.items():
            if hasattr(constraints, constraint_name):
                original_value = getattr(constraints, constraint_name)
                
                if isinstance(original_value, (int, float)):
                    adjusted_value = original_value * factor
                    if isinstance(original_value, int):
                        adjusted_value = int(adjusted_value)
                elif isinstance(original_value, Decimal):
                    adjusted_value = original_value * Decimal(str(factor))
                else:
                    continue
                
                setattr(constraints, constraint_name, adjusted_value)
                
                constraints.adjustments.append(ConstraintAdjustment(
                    constraint_name=constraint_name,
                    original_value=original_value,
                    adjusted_value=adjusted_value,
                    adjustment_factor=factor,
                    justification=f"Volatility regime: {conditions.volatility_regime.value}"
                ))
    
    def _apply_market_regime_adjustments(
        self,
        constraints: DynamicConstraintSet,
        conditions: MarketConditions
    ):
        """Apply market regime adjustments."""
        
        regime_key = conditions.market_regime.value
        adjustments = self.regime_adjustments.get(regime_key, {})
        
        for constraint_name, factor in adjustments.items():
            if hasattr(constraints, constraint_name):
                original_value = getattr(constraints, constraint_name)
                
                if isinstance(original_value, (int, float)):
                    adjusted_value = original_value * factor
                    if isinstance(original_value, int):
                        adjusted_value = int(adjusted_value)
                elif isinstance(original_value, Decimal):
                    adjusted_value = original_value * Decimal(str(factor))
                else:
                    continue
                
                setattr(constraints, constraint_name, adjusted_value)
                
                constraints.adjustments.append(ConstraintAdjustment(
                    constraint_name=constraint_name,
                    original_value=original_value,
                    adjusted_value=adjusted_value,
                    adjustment_factor=factor,
                    justification=f"Market regime: {conditions.market_regime.value}"
                ))
    
    def _apply_liquidity_adjustments(
        self,
        constraints: DynamicConstraintSet,
        conditions: MarketConditions
    ):
        """Apply liquidity regime adjustments."""
        
        if conditions.liquidity_regime in [LiquidityRegime.CONSTRAINED, LiquidityRegime.STRESSED]:
            regime_key = f"{conditions.liquidity_regime.value}_liquidity"
            adjustments = self.regime_adjustments.get(regime_key, {})
            
            for constraint_name, factor in adjustments.items():
                if hasattr(constraints, constraint_name):
                    original_value = getattr(constraints, constraint_name)
                    
                    if isinstance(original_value, (int, float)):
                        adjusted_value = original_value * factor
                        if isinstance(original_value, int):
                            adjusted_value = int(adjusted_value)
                    elif isinstance(original_value, Decimal):
                        adjusted_value = original_value * Decimal(str(factor))
                    else:
                        continue
                    
                    setattr(constraints, constraint_name, adjusted_value)
                    
                    constraints.adjustments.append(ConstraintAdjustment(
                        constraint_name=constraint_name,
                        original_value=original_value,
                        adjusted_value=adjusted_value,
                        adjustment_factor=factor,
                        justification=f"Liquidity regime: {conditions.liquidity_regime.value}"
                    ))
    
    def _apply_event_adjustments(
        self,
        constraints: DynamicConstraintSet,
        conditions: MarketConditions
    ):
        """Apply economic event proximity adjustments."""
        
        events_to_check = []
        
        if conditions.earnings_season:
            events_to_check.append('earnings_season')
        
        if conditions.fed_meeting_proximity and conditions.fed_meeting_proximity <= 7:
            events_to_check.append('fed_meeting_week')
        
        if conditions.economic_data_proximity and conditions.economic_data_proximity <= 1:
            events_to_check.append('economic_data_day')
        
        for event in events_to_check:
            adjustments = self.event_adjustments.get(event, {})
            
            for constraint_name, factor in adjustments.items():
                if hasattr(constraints, constraint_name):
                    original_value = getattr(constraints, constraint_name)
                    
                    if isinstance(original_value, (int, float)):
                        adjusted_value = original_value * factor
                        if isinstance(original_value, int):
                            adjusted_value = int(adjusted_value)
                    elif isinstance(original_value, Decimal):
                        adjusted_value = original_value * Decimal(str(factor))
                    else:
                        continue
                    
                    setattr(constraints, constraint_name, adjusted_value)
                    
                    constraints.adjustments.append(ConstraintAdjustment(
                        constraint_name=constraint_name,
                        original_value=original_value,
                        adjusted_value=adjusted_value,
                        adjustment_factor=factor,
                        justification=f"Economic event: {event}"
                    ))
    
    def _apply_crisis_mode_adjustments(
        self,
        constraints: DynamicConstraintSet,
        conditions: MarketConditions
    ):
        """Apply emergency crisis mode adjustments."""
        
        if (conditions.market_regime == MarketRegime.CRISIS_MODE or 
            conditions.volatility_regime == VolatilityRegime.EXTREME):
            
            # Emergency risk reduction protocols
            crisis_adjustments = {
                'min_probability_of_profit': 1.15,
                'max_loss_per_trade': 0.4,
                'max_delta_per_100k': 0.2,
                'min_open_interest': 3.0,
                'max_quote_age_minutes': 0.1,
                'max_position_size_pct': 0.5,
            }
            
            for constraint_name, factor in crisis_adjustments.items():
                if hasattr(constraints, constraint_name):
                    original_value = getattr(constraints, constraint_name)
                    
                    if isinstance(original_value, (int, float)):
                        adjusted_value = original_value * factor
                        if isinstance(original_value, int):
                            adjusted_value = int(adjusted_value)
                    elif isinstance(original_value, Decimal):
                        adjusted_value = original_value * Decimal(str(factor))
                    else:
                        continue
                    
                    setattr(constraints, constraint_name, adjusted_value)
                    
                    constraints.adjustments.append(ConstraintAdjustment(
                        constraint_name=constraint_name,
                        original_value=original_value,
                        adjusted_value=adjusted_value,
                        adjustment_factor=factor,
                        justification="Emergency crisis mode protocols"
                    ))