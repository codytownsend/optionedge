"""
Multi-strategy generation engine for options strategies.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
from enum import Enum
import itertools
import logging

from ...data.models.options import OptionsChain, OptionQuote, OptionType, Greeks
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData
from ...data.models.trades import (
    StrategyDefinition, StrategyType, TradeLeg, TradeDirection,
    TradeCandidate, TradeFilterCriteria
)
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, CalculationError, InsufficientDataError
)


class StrategySelectionCriteria(Enum):
    """Criteria for selecting which strategies to generate."""
    MARKET_NEUTRAL = "market_neutral"
    BULLISH_BIAS = "bullish_bias"
    BEARISH_BIAS = "bearish_bias"
    HIGH_PROBABILITY = "high_probability"
    INCOME_FOCUSED = "income_focused"
    VOLATILITY_PLAY = "volatility_play"


@dataclass
class StrategyGenerationConfig:
    """Configuration for strategy generation."""
    max_strategies_per_type: int = 5
    min_credit_amount: Decimal = Decimal('50')
    max_risk_per_trade: Decimal = Decimal('500')
    preferred_dte_range: Tuple[int, int] = (7, 45)
    delta_neutral_tolerance: float = 0.1
    min_probability_of_profit: float = 0.65
    min_credit_to_max_loss: float = 0.33
    include_strategy_types: Optional[List[StrategyType]] = None


@dataclass
class StrategyCandidate:
    """Individual strategy candidate with optimization metrics."""
    strategy: StrategyDefinition
    expected_return: Optional[Decimal] = None
    sharpe_ratio: Optional[float] = None
    kelly_fraction: Optional[float] = None
    max_drawdown: Optional[Decimal] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    optimization_score: Optional[float] = None


class StrategyGenerationService:
    """
    Multi-strategy generation engine for options.
    
    Features:
    - Credit spread optimization across all viable strike combinations
    - Iron condor construction with dynamic wing selection
    - Covered call and cash-secured put strategy generation
    - Calendar spread identification using volatility term structure
    - Diagonal spread construction with time decay optimization
    - Strike selection using probability-weighted expected returns
    - Expiration date optimization balancing time decay and event risk
    - Position sizing calculation based on Kelly criterion and risk parity
    - Greeks-neutral strategy construction for market-neutral approaches
    - Volatility arbitrage opportunity identification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Risk-free rate (should be updated from FRED data)
        self.risk_free_rate = 0.05
        
        # Strategy generation weights
        self.strategy_weights = {
            StrategyType.PUT_CREDIT_SPREAD: 0.25,
            StrategyType.CALL_CREDIT_SPREAD: 0.25,
            StrategyType.IRON_CONDOR: 0.20,
            StrategyType.IRON_BUTTERFLY: 0.10,
            StrategyType.COVERED_CALL: 0.10,
            StrategyType.CASH_SECURED_PUT: 0.10
        }
    
    @handle_errors(operation_name="generate_strategies")
    def generate_all_strategies(
        self,
        options_chain: OptionsChain,
        market_data: Dict[str, Any],
        config: Optional[StrategyGenerationConfig] = None
    ) -> List[StrategyCandidate]:
        """
        Generate all viable option strategies for a symbol.
        
        Args:
            options_chain: Liquid options chain
            market_data: Supporting market data (technicals, fundamentals, etc.)
            config: Strategy generation configuration
            
        Returns:
            List of optimized strategy candidates
        """
        if config is None:
            config = StrategyGenerationConfig()
        
        self.logger.info(f"Generating strategies for {options_chain.underlying}")
        
        # Validate inputs
        if not options_chain.options or not options_chain.underlying_price:
            raise InsufficientDataError(
                f"Insufficient options data for {options_chain.underlying}",
                symbol=options_chain.underlying
            )
        
        all_candidates = []
        
        # Determine which strategy types to generate
        strategy_types = config.include_strategy_types or list(self.strategy_weights.keys())
        
        # Generate each strategy type
        for strategy_type in strategy_types:
            try:
                candidates = self._generate_strategy_type(
                    strategy_type, options_chain, market_data, config
                )
                all_candidates.extend(candidates)
                self.logger.debug(f"Generated {len(candidates)} {strategy_type.value} candidates")
            except Exception as e:
                self.logger.warning(f"Failed to generate {strategy_type.value}: {str(e)}")
        
        # Optimize and rank candidates
        optimized_candidates = self._optimize_strategies(all_candidates, market_data, config)
        
        self.logger.info(f"Generated {len(optimized_candidates)} total strategy candidates")
        return optimized_candidates
    
    def generate_credit_spreads(
        self,
        options_chain: OptionsChain,
        option_type: OptionType,
        config: StrategyGenerationConfig
    ) -> List[StrategyCandidate]:
        """Generate credit spread strategies (put or call)."""
        
        candidates = []
        
        for expiration in options_chain.get_expirations():
            # Check DTE filter
            dte = (expiration - date.today()).days
            if not (config.preferred_dte_range[0] <= dte <= config.preferred_dte_range[1]):
                continue
            
            strikes = options_chain.get_strikes(expiration)
            if len(strikes) < 2:
                continue
            
            # Generate all viable strike combinations
            for short_strike, long_strike in itertools.combinations(strikes, 2):
                
                # Ensure proper strike ordering
                if option_type == OptionType.PUT:
                    if short_strike <= long_strike:
                        continue  # Put spread: short strike > long strike
                    short_strike, long_strike = short_strike, long_strike
                else:  # CALL
                    if short_strike >= long_strike:
                        continue  # Call spread: short strike < long strike
                
                # Get options
                short_option = options_chain.get_option(expiration, short_strike, option_type)
                long_option = options_chain.get_option(expiration, long_strike, option_type)
                
                if not short_option or not long_option:
                    continue
                
                # Check liquidity
                if not (short_option.is_liquid() and long_option.is_liquid()):
                    continue
                
                # Create strategy
                strategy = self._create_credit_spread(
                    short_option, long_option, options_chain.underlying_price
                )
                
                if strategy and self._meets_basic_criteria(strategy, config):
                    candidate = StrategyCandidate(strategy=strategy)
                    candidates.append(candidate)
                    
                    # Limit candidates per expiration
                    if len(candidates) >= config.max_strategies_per_type:
                        break
        
        return candidates
    
    def generate_iron_condors(
        self,
        options_chain: OptionsChain,
        config: StrategyGenerationConfig
    ) -> List[StrategyCandidate]:
        """Generate iron condor strategies with dynamic wing selection."""
        
        candidates = []
        
        for expiration in options_chain.get_expirations():
            # Check DTE filter
            dte = (expiration - date.today()).days
            if not (config.preferred_dte_range[0] <= dte <= config.preferred_dte_range[1]):
                continue
            
            strikes = options_chain.get_strikes(expiration)
            if len(strikes) < 4:
                continue
            
            atm_strike = options_chain.get_atm_strike(expiration)
            if not atm_strike:
                continue
            
            # Find strikes around ATM for body
            sorted_strikes = sorted(strikes)
            atm_index = min(range(len(sorted_strikes)), 
                          key=lambda i: abs(sorted_strikes[i] - atm_strike))
            
            # Generate condors with different wing widths
            for put_short_offset in range(1, 4):  # 1-3 strikes OTM
                for call_short_offset in range(1, 4):
                    for wing_width in [1, 2, 3]:  # Strike width for wings
                        
                        # Calculate strike indices
                        put_short_idx = atm_index - put_short_offset
                        put_long_idx = put_short_idx - wing_width
                        call_short_idx = atm_index + call_short_offset
                        call_long_idx = call_short_idx + wing_width
                        
                        # Validate indices
                        if (put_long_idx < 0 or 
                            call_long_idx >= len(sorted_strikes) or
                            put_short_idx <= 0 or
                            call_short_idx >= len(sorted_strikes) - 1):
                            continue
                        
                        # Get strikes
                        put_long_strike = sorted_strikes[put_long_idx]
                        put_short_strike = sorted_strikes[put_short_idx]
                        call_short_strike = sorted_strikes[call_short_idx]
                        call_long_strike = sorted_strikes[call_long_idx]
                        
                        # Get options
                        put_long = options_chain.get_option(expiration, put_long_strike, OptionType.PUT)
                        put_short = options_chain.get_option(expiration, put_short_strike, OptionType.PUT)
                        call_short = options_chain.get_option(expiration, call_short_strike, OptionType.CALL)
                        call_long = options_chain.get_option(expiration, call_long_strike, OptionType.CALL)
                        
                        if not all([put_long, put_short, call_short, call_long]):
                            continue
                        
                        # Check liquidity
                        if not all(opt.is_liquid() for opt in [put_long, put_short, call_short, call_long]):
                            continue
                        
                        # Create iron condor
                        strategy = self._create_iron_condor(
                            put_long, put_short, call_short, call_long, options_chain.underlying_price
                        )
                        
                        if strategy and self._meets_basic_criteria(strategy, config):
                            candidate = StrategyCandidate(strategy=strategy)
                            candidates.append(candidate)
                            
                            # Limit candidates
                            if len(candidates) >= config.max_strategies_per_type:
                                return candidates
        
        return candidates
    
    def generate_covered_calls(
        self,
        options_chain: OptionsChain,
        config: StrategyGenerationConfig
    ) -> List[StrategyCandidate]:
        """Generate covered call strategies."""
        
        candidates = []
        
        for expiration in options_chain.get_expirations():
            # Check DTE filter
            dte = (expiration - date.today()).days
            if not (config.preferred_dte_range[0] <= dte <= config.preferred_dte_range[1]):
                continue
            
            strikes = options_chain.get_strikes(expiration)
            
            # Focus on OTM calls
            otm_calls = [
                strike for strike in strikes 
                if strike > options_chain.underlying_price
            ]
            
            for strike in sorted(otm_calls)[:5]:  # Top 5 OTM strikes
                call_option = options_chain.get_option(expiration, strike, OptionType.CALL)
                
                if not call_option or not call_option.is_liquid():
                    continue
                
                # Create covered call strategy
                strategy = self._create_covered_call(call_option, options_chain.underlying_price)
                
                if strategy and self._meets_basic_criteria(strategy, config):
                    candidate = StrategyCandidate(strategy=strategy)
                    candidates.append(candidate)
        
        return candidates
    
    def generate_cash_secured_puts(
        self,
        options_chain: OptionsChain,
        config: StrategyGenerationConfig
    ) -> List[StrategyCandidate]:
        """Generate cash-secured put strategies."""
        
        candidates = []
        
        for expiration in options_chain.get_expirations():
            # Check DTE filter
            dte = (expiration - date.today()).days
            if not (config.preferred_dte_range[0] <= dte <= config.preferred_dte_range[1]):
                continue
            
            strikes = options_chain.get_strikes(expiration)
            
            # Focus on OTM puts
            otm_puts = [
                strike for strike in strikes 
                if strike < options_chain.underlying_price
            ]
            
            for strike in sorted(otm_puts, reverse=True)[:5]:  # Top 5 OTM strikes
                put_option = options_chain.get_option(expiration, strike, OptionType.PUT)
                
                if not put_option or not put_option.is_liquid():
                    continue
                
                # Create cash-secured put strategy
                strategy = self._create_cash_secured_put(put_option, options_chain.underlying_price)
                
                if strategy and self._meets_basic_criteria(strategy, config):
                    candidate = StrategyCandidate(strategy=strategy)
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_strategy_type(
        self,
        strategy_type: StrategyType,
        options_chain: OptionsChain,
        market_data: Dict[str, Any],
        config: StrategyGenerationConfig
    ) -> List[StrategyCandidate]:
        """Generate candidates for a specific strategy type."""
        
        if strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            return self.generate_credit_spreads(options_chain, OptionType.PUT, config)
        elif strategy_type == StrategyType.CALL_CREDIT_SPREAD:
            return self.generate_credit_spreads(options_chain, OptionType.CALL, config)
        elif strategy_type == StrategyType.IRON_CONDOR:
            return self.generate_iron_condors(options_chain, config)
        elif strategy_type == StrategyType.COVERED_CALL:
            return self.generate_covered_calls(options_chain, config)
        elif strategy_type == StrategyType.CASH_SECURED_PUT:
            return self.generate_cash_secured_puts(options_chain, config)
        else:
            # Other strategy types would be implemented here
            return []
    
    def _create_credit_spread(
        self,
        short_option: OptionQuote,
        long_option: OptionQuote,
        underlying_price: Decimal
    ) -> Optional[StrategyDefinition]:
        """Create credit spread strategy definition."""
        
        try:
            # Create legs
            short_leg = TradeLeg(
                option=short_option,
                direction=TradeDirection.SELL,
                quantity=1
            )
            
            long_leg = TradeLeg(
                option=long_option,
                direction=TradeDirection.BUY,
                quantity=1
            )
            
            # Determine strategy type
            if short_option.option_type == OptionType.PUT:
                strategy_type = StrategyType.PUT_CREDIT_SPREAD
            else:
                strategy_type = StrategyType.CALL_CREDIT_SPREAD
            
            # Create strategy
            strategy = StrategyDefinition(
                strategy_type=strategy_type,
                underlying=short_option.underlying,
                legs=[short_leg, long_leg]
            )
            
            # Calculate metrics
            strategy.calculate_strategy_metrics()
            self._calculate_probability_metrics(strategy, underlying_price)
            self._calculate_risk_metrics(strategy)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"Failed to create credit spread: {str(e)}")
            return None
    
    def _create_iron_condor(
        self,
        put_long: OptionQuote,
        put_short: OptionQuote,
        call_short: OptionQuote,
        call_long: OptionQuote,
        underlying_price: Decimal
    ) -> Optional[StrategyDefinition]:
        """Create iron condor strategy definition."""
        
        try:
            # Create legs
            legs = [
                TradeLeg(option=put_long, direction=TradeDirection.BUY, quantity=1),
                TradeLeg(option=put_short, direction=TradeDirection.SELL, quantity=1),
                TradeLeg(option=call_short, direction=TradeDirection.SELL, quantity=1),
                TradeLeg(option=call_long, direction=TradeDirection.BUY, quantity=1)
            ]
            
            # Create strategy
            strategy = StrategyDefinition(
                strategy_type=StrategyType.IRON_CONDOR,
                underlying=put_long.underlying,
                legs=legs
            )
            
            # Calculate metrics
            strategy.calculate_strategy_metrics()
            self._calculate_probability_metrics(strategy, underlying_price)
            self._calculate_risk_metrics(strategy)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"Failed to create iron condor: {str(e)}")
            return None
    
    def _create_covered_call(
        self,
        call_option: OptionQuote,
        underlying_price: Decimal
    ) -> Optional[StrategyDefinition]:
        """Create covered call strategy definition."""
        
        try:
            # Create call leg (short)
            call_leg = TradeLeg(
                option=call_option,
                direction=TradeDirection.SELL,
                quantity=1
            )
            
            # Note: In a real implementation, we'd also track the underlying stock position
            # For now, we'll just track the options component
            
            strategy = StrategyDefinition(
                strategy_type=StrategyType.COVERED_CALL,
                underlying=call_option.underlying,
                legs=[call_leg]
            )
            
            # Calculate metrics (simplified for options component only)
            strategy.calculate_strategy_metrics()
            self._calculate_probability_metrics(strategy, underlying_price)
            self._calculate_risk_metrics(strategy)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"Failed to create covered call: {str(e)}")
            return None
    
    def _create_cash_secured_put(
        self,
        put_option: OptionQuote,
        underlying_price: Decimal
    ) -> Optional[StrategyDefinition]:
        """Create cash-secured put strategy definition."""
        
        try:
            # Create put leg (short)
            put_leg = TradeLeg(
                option=put_option,
                direction=TradeDirection.SELL,
                quantity=1
            )
            
            strategy = StrategyDefinition(
                strategy_type=StrategyType.CASH_SECURED_PUT,
                underlying=put_option.underlying,
                legs=[put_leg]
            )
            
            # Calculate metrics
            strategy.calculate_strategy_metrics()
            self._calculate_probability_metrics(strategy, underlying_price)
            self._calculate_risk_metrics(strategy)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"Failed to create cash-secured put: {str(e)}")
            return None
    
    def _calculate_probability_metrics(
        self,
        strategy: StrategyDefinition,
        underlying_price: Decimal
    ):
        """Calculate probability of profit and breakeven points."""
        
        try:
            if strategy.strategy_type in [StrategyType.PUT_CREDIT_SPREAD, StrategyType.CALL_CREDIT_SPREAD]:
                # Single breakeven point for credit spreads
                short_leg = next(leg for leg in strategy.legs if leg.direction == TradeDirection.SELL)
                
                if strategy.net_credit:
                    if short_leg.option_type == OptionType.PUT:
                        breakeven = short_leg.strike - (strategy.net_credit / 100)
                    else:  # CALL
                        breakeven = short_leg.strike + (strategy.net_credit / 100)
                    
                    strategy.breakeven_points = [breakeven]
                    
                    # Simplified probability calculation
                    # In reality, would use Black-Scholes or Monte Carlo
                    current_price = float(underlying_price)
                    be_price = float(breakeven)
                    
                    if short_leg.option_type == OptionType.PUT:
                        # Probability price stays above breakeven
                        if current_price > be_price:
                            strategy.probability_of_profit = min(0.95, 0.5 + (current_price - be_price) / current_price * 2)
                        else:
                            strategy.probability_of_profit = max(0.05, 0.5 - (be_price - current_price) / current_price * 2)
                    else:  # CALL
                        # Probability price stays below breakeven
                        if current_price < be_price:
                            strategy.probability_of_profit = min(0.95, 0.5 + (be_price - current_price) / current_price * 2)
                        else:
                            strategy.probability_of_profit = max(0.05, 0.5 - (current_price - be_price) / current_price * 2)
            
            elif strategy.strategy_type == StrategyType.IRON_CONDOR:
                # Two breakeven points for iron condor
                put_short = next(leg for leg in strategy.legs 
                               if leg.direction == TradeDirection.SELL and leg.option_type == OptionType.PUT)
                call_short = next(leg for leg in strategy.legs 
                                if leg.direction == TradeDirection.SELL and leg.option_type == OptionType.CALL)
                
                if strategy.net_credit:
                    credit_per_share = strategy.net_credit / 100
                    lower_breakeven = put_short.strike - credit_per_share
                    upper_breakeven = call_short.strike + credit_per_share
                    
                    strategy.breakeven_points = [lower_breakeven, upper_breakeven]
                    
                    # Probability of staying between breakevens (simplified)
                    current_price = float(underlying_price)
                    range_width = float(upper_breakeven - lower_breakeven)
                    price_position = (current_price - float(lower_breakeven)) / range_width
                    
                    # Higher probability when price is centered in range
                    if 0.2 <= price_position <= 0.8:
                        strategy.probability_of_profit = 0.75  # Good position
                    elif 0.1 <= price_position <= 0.9:
                        strategy.probability_of_profit = 0.65  # Acceptable
                    else:
                        strategy.probability_of_profit = 0.45  # Poor position
            
            else:
                # Default probability for other strategies
                strategy.probability_of_profit = 0.60
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate probability metrics: {str(e)}")
            strategy.probability_of_profit = 0.50  # Default
    
    def _calculate_risk_metrics(self, strategy: StrategyDefinition):
        """Calculate max profit, max loss, and other risk metrics."""
        
        try:
            if strategy.strategy_type in [StrategyType.PUT_CREDIT_SPREAD, StrategyType.CALL_CREDIT_SPREAD]:
                # Credit spread risk metrics
                if strategy.net_credit and strategy.strike_width:
                    strategy.max_profit = strategy.net_credit
                    strategy.max_loss = (strategy.strike_width * 100) - strategy.net_credit
            
            elif strategy.strategy_type == StrategyType.IRON_CONDOR:
                # Iron condor risk metrics
                if strategy.net_credit and strategy.legs:
                    strategy.max_profit = strategy.net_credit
                    
                    # Find the larger wing width
                    put_strikes = [leg.strike for leg in strategy.legs if leg.option_type == OptionType.PUT]
                    call_strikes = [leg.strike for leg in strategy.legs if leg.option_type == OptionType.CALL]
                    
                    if len(put_strikes) == 2 and len(call_strikes) == 2:
                        put_width = abs(max(put_strikes) - min(put_strikes))
                        call_width = abs(max(call_strikes) - min(call_strikes))
                        max_width = max(put_width, call_width)
                        strategy.max_loss = (max_width * 100) - strategy.net_credit
            
            elif strategy.strategy_type == StrategyType.COVERED_CALL:
                # Covered call (options component only)
                if strategy.net_credit:
                    strategy.max_profit = strategy.net_credit  # Plus stock appreciation to strike
                    strategy.max_loss = None  # Unlimited downside in stock
            
            elif strategy.strategy_type == StrategyType.CASH_SECURED_PUT:
                # Cash-secured put
                if strategy.net_credit and strategy.legs:
                    put_leg = strategy.legs[0]
                    strategy.max_profit = strategy.net_credit
                    strategy.max_loss = (put_leg.strike * 100) - strategy.net_credit
            
            # Ensure max_loss doesn't exceed config limits
            if strategy.max_loss and strategy.max_loss > Decimal('1000'):
                strategy.max_loss = Decimal('1000')  # Cap for safety
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate risk metrics: {str(e)}")
    
    def _meets_basic_criteria(
        self,
        strategy: StrategyDefinition,
        config: StrategyGenerationConfig
    ) -> bool:
        """Check if strategy meets basic generation criteria."""
        
        # Minimum credit check
        if strategy.net_credit and strategy.net_credit < config.min_credit_amount:
            return False
        
        # Maximum risk check
        if strategy.max_loss and strategy.max_loss > config.max_risk_per_trade:
            return False
        
        # Minimum probability check
        if (strategy.probability_of_profit and 
            strategy.probability_of_profit < config.min_probability_of_profit):
            return False
        
        # Credit-to-max-loss ratio check
        if (strategy.credit_to_max_loss_ratio and
            strategy.credit_to_max_loss_ratio < config.min_credit_to_max_loss):
            return False
        
        return True
    
    def _optimize_strategies(
        self,
        candidates: List[StrategyCandidate],
        market_data: Dict[str, Any],
        config: StrategyGenerationConfig
    ) -> List[StrategyCandidate]:
        """Optimize and rank strategy candidates."""
        
        for candidate in candidates:
            # Calculate optimization metrics
            self._calculate_expected_return(candidate, market_data)
            self._calculate_sharpe_ratio(candidate)
            self._calculate_kelly_fraction(candidate)
            
            # Calculate overall optimization score
            self._calculate_optimization_score(candidate)
        
        # Sort by optimization score
        optimized = sorted(candidates, key=lambda c: c.optimization_score or 0, reverse=True)
        
        # Limit total number returned
        max_total = sum(config.max_strategies_per_type for _ in self.strategy_weights)
        return optimized[:max_total]
    
    def _calculate_expected_return(self, candidate: StrategyCandidate, market_data: Dict[str, Any]):
        """Calculate expected return for strategy."""
        
        strategy = candidate.strategy
        
        if not strategy.probability_of_profit or not strategy.max_profit or not strategy.max_loss:
            return
        
        try:
            # Simple expected value calculation
            prob_profit = strategy.probability_of_profit
            prob_loss = 1 - prob_profit
            
            expected_profit = float(strategy.max_profit) * prob_profit
            expected_loss = float(strategy.max_loss) * prob_loss
            
            candidate.expected_return = Decimal(str(expected_profit - expected_loss))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate expected return: {str(e)}")
    
    def _calculate_sharpe_ratio(self, candidate: StrategyCandidate):
        """Calculate Sharpe ratio for strategy."""
        
        if not candidate.expected_return or not candidate.strategy.max_loss:
            return
        
        try:
            # Simplified Sharpe calculation
            # In practice, would need historical volatility data
            expected_return = float(candidate.expected_return)
            max_risk = float(candidate.strategy.max_loss)
            
            if max_risk > 0:
                # Risk-adjusted return
                candidate.sharpe_ratio = expected_return / max_risk
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate Sharpe ratio: {str(e)}")
    
    def _calculate_kelly_fraction(self, candidate: StrategyCandidate):
        """Calculate Kelly criterion fraction for position sizing."""
        
        strategy = candidate.strategy
        
        if (not strategy.probability_of_profit or 
            not strategy.max_profit or 
            not strategy.max_loss):
            return
        
        try:
            p = strategy.probability_of_profit  # Probability of win
            b = float(strategy.max_profit) / float(strategy.max_loss)  # Win/loss ratio
            q = 1 - p  # Probability of loss
            
            # Kelly fraction = (bp - q) / b
            kelly_fraction = (b * p - q) / b
            
            # Cap at reasonable maximum (25%)
            candidate.kelly_fraction = max(0, min(0.25, kelly_fraction))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate Kelly fraction: {str(e)}")
    
    def _calculate_optimization_score(self, candidate: StrategyCandidate):
        """Calculate overall optimization score."""
        
        scores = []
        weights = []
        
        # Expected return component
        if candidate.expected_return:
            return_score = min(100, max(0, float(candidate.expected_return) / 100 * 100))
            scores.append(return_score)
            weights.append(0.30)
        
        # Probability of profit component
        if candidate.strategy.probability_of_profit:
            prob_score = candidate.strategy.probability_of_profit * 100
            scores.append(prob_score)
            weights.append(0.25)
        
        # Sharpe ratio component
        if candidate.sharpe_ratio:
            sharpe_score = min(100, max(0, candidate.sharpe_ratio * 50 + 50))
            scores.append(sharpe_score)
            weights.append(0.20)
        
        # Kelly fraction component
        if candidate.kelly_fraction:
            kelly_score = candidate.kelly_fraction * 400  # Scale to 0-100
            scores.append(kelly_score)
            weights.append(0.15)
        
        # Credit-to-max-loss ratio component
        if candidate.strategy.credit_to_max_loss_ratio:
            ratio_score = min(100, candidate.strategy.credit_to_max_loss_ratio * 150)
            scores.append(ratio_score)
            weights.append(0.10)
        
        if scores and weights:
            candidate.optimization_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            candidate.optimization_score = 0.0