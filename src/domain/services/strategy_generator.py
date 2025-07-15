"""
Strategy generation service for creating options trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

from ..entities.strategy import Strategy, CreditSpreadStrategy, IronCondorStrategy, StrategyBuilder
from ..entities.option_contract import OptionContract
from ...data.models.options import OptionsChain, OptionQuote, OptionType
from ...data.models.trades import StrategyType, TradeDirection, TradeLeg
from ...data.models.market_data import StockQuote, TechnicalIndicators

logger = logging.getLogger(__name__)


class StrategyGenerationMode(str, Enum):
    """Strategy generation modes."""
    CONSERVATIVE = "conservative"    # Lower risk, higher probability strategies
    AGGRESSIVE = "aggressive"       # Higher risk, higher reward potential
    BALANCED = "balanced"           # Balanced risk/reward
    INCOME_FOCUSED = "income_focused"  # Maximize income generation
    HEDGING = "hedging"             # Risk mitigation strategies


@dataclass
class StrategyGenerationConfig:
    """Configuration for strategy generation."""
    
    # Basic parameters
    underlying_symbol: str
    underlying_price: Decimal
    mode: StrategyGenerationMode = StrategyGenerationMode.BALANCED
    
    # Strategy preferences
    allowed_strategies: Set[StrategyType] = None
    max_days_to_expiration: int = 45
    min_days_to_expiration: int = 7
    
    # Risk parameters
    max_loss_per_strategy: Decimal = Decimal('500')
    min_probability_of_profit: float = 0.65
    min_credit_to_max_loss: float = 0.33
    
    # Liquidity requirements
    min_option_volume: int = 10
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 0.50
    
    # Strike selection
    strike_selection_method: str = "probability_weighted"  # or "fixed_delta"
    target_delta_range: Tuple[float, float] = (0.15, 0.35)
    
    def __post_init__(self):
        if self.allowed_strategies is None:
            self.allowed_strategies = {
                StrategyType.PUT_CREDIT_SPREAD,
                StrategyType.CALL_CREDIT_SPREAD,
                StrategyType.IRON_CONDOR,
                StrategyType.COVERED_CALL
            }


@dataclass
class StrategyGenerationResult:
    """Result of strategy generation."""
    
    strategy: Strategy
    generation_config: StrategyGenerationConfig
    score: Optional[float] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class StrategyGenerator:
    """
    Service for generating options trading strategies.
    
    This service analyzes options chains and market conditions to generate
    viable trading strategies based on specified criteria and risk parameters.
    """
    
    def __init__(self):
        self._generators = {
            StrategyType.PUT_CREDIT_SPREAD: self._generate_put_credit_spreads,
            StrategyType.CALL_CREDIT_SPREAD: self._generate_call_credit_spreads,
            StrategyType.IRON_CONDOR: self._generate_iron_condors,
            StrategyType.COVERED_CALL: self._generate_covered_calls,
        }
    
    def generate_strategies(self,
                          options_chain: OptionsChain,
                          config: StrategyGenerationConfig,
                          stock_quote: Optional[StockQuote] = None,
                          technical_indicators: Optional[TechnicalIndicators] = None) -> List[StrategyGenerationResult]:
        """
        Generate strategies for a given options chain.
        
        Args:
            options_chain: Complete options chain
            config: Generation configuration
            stock_quote: Current stock quote
            technical_indicators: Technical analysis data
            
        Returns:
            List of generated strategy results
        """
        logger.info(f"Generating strategies for {config.underlying_symbol}")
        
        strategies = []
        
        # Filter options chain for liquidity and expiration
        filtered_options = self._filter_options_chain(options_chain, config)
        
        if not filtered_options:
            logger.warning(f"No liquid options found for {config.underlying_symbol}")
            return strategies
        
        # Generate strategies for each allowed type
        for strategy_type in config.allowed_strategies:
            if strategy_type in self._generators:
                try:
                    generated = self._generators[strategy_type](
                        filtered_options, config, stock_quote, technical_indicators
                    )
                    strategies.extend(generated)
                except Exception as e:
                    logger.warning(f"Failed to generate {strategy_type} for {config.underlying_symbol}: {e}")
        
        # Sort by quality/score
        strategies.sort(key=lambda x: x.score or 0, reverse=True)
        
        logger.info(f"Generated {len(strategies)} strategies for {config.underlying_symbol}")
        return strategies
    
    def _filter_options_chain(self, 
                             options_chain: OptionsChain, 
                             config: StrategyGenerationConfig) -> List[OptionQuote]:
        """Filter options chain for liquidity and expiration criteria."""
        filtered_options = []
        
        for option in options_chain.options:
            # Check expiration range
            days_to_exp = option.days_to_expiration
            if not (config.min_days_to_expiration <= days_to_exp <= config.max_days_to_expiration):
                continue
            
            # Check liquidity
            if not self._meets_liquidity_requirements(option, config):
                continue
            
            filtered_options.append(option)
        
        return filtered_options
    
    def _meets_liquidity_requirements(self, option: OptionQuote, config: StrategyGenerationConfig) -> bool:
        """Check if option meets liquidity requirements."""
        # Volume check
        if option.volume is not None and option.volume < config.min_option_volume:
            return False
        
        # Open interest check
        if option.open_interest is not None and option.open_interest < config.min_open_interest:
            return False
        
        # Spread check
        if option.bid_ask_spread_percent is not None:
            if option.bid_ask_spread_percent > config.max_bid_ask_spread_pct:
                return False
        
        # Must have valid bid and ask
        if option.bid is None or option.ask is None:
            return False
        
        return True
    
    def _generate_put_credit_spreads(self,
                                   options: List[OptionQuote],
                                   config: StrategyGenerationConfig,
                                   stock_quote: Optional[StockQuote],
                                   technical_indicators: Optional[TechnicalIndicators]) -> List[StrategyGenerationResult]:
        """Generate put credit spread strategies."""
        strategies = []
        
        # Get puts only
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        
        # Group by expiration
        puts_by_expiration = {}
        for put in puts:
            if put.expiration not in puts_by_expiration:
                puts_by_expiration[put.expiration] = []
            puts_by_expiration[put.expiration].append(put)
        
        # Generate spreads for each expiration
        for expiration, exp_puts in puts_by_expiration.items():
            # Sort by strike (descending for puts)
            exp_puts.sort(key=lambda x: x.strike, reverse=True)
            
            # Generate spread combinations
            for i in range(len(exp_puts)):
                for j in range(i + 1, len(exp_puts)):
                    short_put = exp_puts[i]  # Higher strike (short)
                    long_put = exp_puts[j]   # Lower strike (long protection)
                    
                    try:
                        strategy_result = self._create_put_credit_spread(
                            short_put, long_put, config
                        )
                        if strategy_result:
                            strategies.append(strategy_result)
                    except Exception as e:
                        logger.debug(f"Failed to create put credit spread: {e}")
        
        return strategies
    
    def _generate_call_credit_spreads(self,
                                    options: List[OptionQuote],
                                    config: StrategyGenerationConfig,
                                    stock_quote: Optional[StockQuote],
                                    technical_indicators: Optional[TechnicalIndicators]) -> List[StrategyGenerationResult]:
        """Generate call credit spread strategies."""
        strategies = []
        
        # Get calls only
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        
        # Group by expiration
        calls_by_expiration = {}
        for call in calls:
            if call.expiration not in calls_by_expiration:
                calls_by_expiration[call.expiration] = []
            calls_by_expiration[call.expiration].append(call)
        
        # Generate spreads for each expiration
        for expiration, exp_calls in calls_by_expiration.items():
            # Sort by strike (ascending for calls)
            exp_calls.sort(key=lambda x: x.strike)
            
            # Generate spread combinations
            for i in range(len(exp_calls)):
                for j in range(i + 1, len(exp_calls)):
                    short_call = exp_calls[i]  # Lower strike (short)
                    long_call = exp_calls[j]   # Higher strike (long protection)
                    
                    try:
                        strategy_result = self._create_call_credit_spread(
                            short_call, long_call, config
                        )
                        if strategy_result:
                            strategies.append(strategy_result)
                    except Exception as e:
                        logger.debug(f"Failed to create call credit spread: {e}")
        
        return strategies
    
    def _generate_iron_condors(self,
                             options: List[OptionQuote],
                             config: StrategyGenerationConfig,
                             stock_quote: Optional[StockQuote],
                             technical_indicators: Optional[TechnicalIndicators]) -> List[StrategyGenerationResult]:
        """Generate iron condor strategies."""
        strategies = []
        
        # Group by expiration
        options_by_expiration = {}
        for option in options:
            if option.expiration not in options_by_expiration:
                options_by_expiration[option.expiration] = {'calls': [], 'puts': []}
            
            if option.option_type == OptionType.CALL:
                options_by_expiration[option.expiration]['calls'].append(option)
            else:
                options_by_expiration[option.expiration]['puts'].append(option)
        
        # Generate iron condors for each expiration
        for expiration, exp_options in options_by_expiration.items():
            calls = sorted(exp_options['calls'], key=lambda x: x.strike)
            puts = sorted(exp_options['puts'], key=lambda x: x.strike, reverse=True)
            
            if len(calls) < 2 or len(puts) < 2:
                continue
            
            # Generate iron condor combinations
            for put_short in puts[:3]:  # Limit combinations for performance
                for put_long in puts:
                    if put_long.strike >= put_short.strike:
                        continue
                    
                    for call_short in calls[:3]:
                        if call_short.strike <= put_short.strike:
                            continue
                        
                        for call_long in calls:
                            if call_long.strike <= call_short.strike:
                                continue
                            
                            try:
                                strategy_result = self._create_iron_condor(
                                    put_short, put_long, call_short, call_long, config
                                )
                                if strategy_result:
                                    strategies.append(strategy_result)
                            except Exception as e:
                                logger.debug(f"Failed to create iron condor: {e}")
        
        return strategies
    
    def _generate_covered_calls(self,
                              options: List[OptionQuote],
                              config: StrategyGenerationConfig,
                              stock_quote: Optional[StockQuote],
                              technical_indicators: Optional[TechnicalIndicators]) -> List[StrategyGenerationResult]:
        """Generate covered call strategies."""
        # Covered calls require stock ownership, which is beyond basic options strategies
        # For now, return empty list
        return []
    
    def _create_put_credit_spread(self,
                                short_put: OptionQuote,
                                long_put: OptionQuote,
                                config: StrategyGenerationConfig) -> Optional[StrategyGenerationResult]:
        """Create a put credit spread strategy."""
        # Create trade legs
        short_leg = TradeLeg(
            option=short_put,
            direction=TradeDirection.SHORT,
            quantity=1
        )
        
        long_leg = TradeLeg(
            option=long_put,
            direction=TradeDirection.LONG,
            quantity=1
        )
        
        # Create strategy
        strategy = CreditSpreadStrategy(
            underlying_symbol=config.underlying_symbol,
            short_leg=short_leg,
            long_leg=long_leg,
            underlying_price=config.underlying_price
        )
        
        # Validate strategy meets criteria
        if not self._validate_strategy_criteria(strategy, config):
            return None
        
        # Calculate score
        score = self._calculate_strategy_score(strategy, config)
        
        return StrategyGenerationResult(
            strategy=strategy,
            generation_config=config,
            score=score
        )
    
    def _create_call_credit_spread(self,
                                 short_call: OptionQuote,
                                 long_call: OptionQuote,
                                 config: StrategyGenerationConfig) -> Optional[StrategyGenerationResult]:
        """Create a call credit spread strategy."""
        # Create trade legs
        short_leg = TradeLeg(
            option=short_call,
            direction=TradeDirection.SHORT,
            quantity=1
        )
        
        long_leg = TradeLeg(
            option=long_call,
            direction=TradeDirection.LONG,
            quantity=1
        )
        
        # Create strategy
        strategy = CreditSpreadStrategy(
            underlying_symbol=config.underlying_symbol,
            short_leg=short_leg,
            long_leg=long_leg,
            underlying_price=config.underlying_price
        )
        
        # Validate strategy meets criteria
        if not self._validate_strategy_criteria(strategy, config):
            return None
        
        # Calculate score
        score = self._calculate_strategy_score(strategy, config)
        
        return StrategyGenerationResult(
            strategy=strategy,
            generation_config=config,
            score=score
        )
    
    def _create_iron_condor(self,
                          put_short: OptionQuote,
                          put_long: OptionQuote,
                          call_short: OptionQuote,
                          call_long: OptionQuote,
                          config: StrategyGenerationConfig) -> Optional[StrategyGenerationResult]:
        """Create an iron condor strategy."""
        # Create trade legs
        put_short_leg = TradeLeg(option=put_short, direction=TradeDirection.SHORT, quantity=1)
        put_long_leg = TradeLeg(option=put_long, direction=TradeDirection.LONG, quantity=1)
        call_short_leg = TradeLeg(option=call_short, direction=TradeDirection.SHORT, quantity=1)
        call_long_leg = TradeLeg(option=call_long, direction=TradeDirection.LONG, quantity=1)
        
        # Create strategy
        strategy = IronCondorStrategy(
            underlying_symbol=config.underlying_symbol,
            put_spread_short=put_short_leg,
            put_spread_long=put_long_leg,
            call_spread_short=call_short_leg,
            call_spread_long=call_long_leg,
            underlying_price=config.underlying_price
        )
        
        # Validate strategy meets criteria
        if not self._validate_strategy_criteria(strategy, config):
            return None
        
        # Calculate score
        score = self._calculate_strategy_score(strategy, config)
        
        return StrategyGenerationResult(
            strategy=strategy,
            generation_config=config,
            score=score
        )
    
    def _validate_strategy_criteria(self, strategy: Strategy, config: StrategyGenerationConfig) -> bool:
        """Validate that strategy meets minimum criteria."""
        # Check max loss
        max_loss = strategy.calculate_max_loss()
        if max_loss and abs(max_loss) > config.max_loss_per_strategy:
            return False
        
        # Check probability of profit
        pop = strategy.calculate_probability_of_profit()
        if pop and pop < config.min_probability_of_profit:
            return False
        
        # Check credit-to-max-loss ratio for credit strategies
        if strategy.is_credit_strategy():
            net_credit = strategy.calculate_net_premium()
            if max_loss and net_credit > 0:
                ratio = float(net_credit / abs(max_loss))
                if ratio < config.min_credit_to_max_loss:
                    return False
        
        return True
    
    def _calculate_strategy_score(self, strategy: Strategy, config: StrategyGenerationConfig) -> float:
        """Calculate a score for the strategy based on multiple factors."""
        score = 0.0
        
        # Probability of profit (weight: 30%)
        pop = strategy.calculate_probability_of_profit()
        if pop:
            score += (pop - 0.5) * 0.3 * 100  # Scale to 0-15 points
        
        # Return on capital (weight: 25%)
        roc = strategy.calculate_return_on_capital()
        if roc:
            # Normalize annualized return (cap at 100%)
            normalized_roc = min(roc, 1.0)
            score += normalized_roc * 0.25 * 100  # Scale to 0-25 points
        
        # Credit-to-max-loss ratio (weight: 20%)
        if strategy.is_credit_strategy():
            net_credit = strategy.calculate_net_premium()
            max_loss = strategy.calculate_max_loss()
            if net_credit > 0 and max_loss:
                ratio = float(net_credit / abs(max_loss))
                # Normalize ratio (typical range 0-1)
                normalized_ratio = min(ratio, 1.0)
                score += normalized_ratio * 0.20 * 100  # Scale to 0-20 points
        
        # Days to expiration factor (weight: 15%)
        days_to_exp = strategy.get_days_to_expiration()
        if days_to_exp > 0:
            # Prefer strategies with moderate time to expiration (20-35 days optimal)
            if 20 <= days_to_exp <= 35:
                dte_score = 1.0
            elif 15 <= days_to_exp < 20 or 35 < days_to_exp <= 45:
                dte_score = 0.8
            elif 7 <= days_to_exp < 15:
                dte_score = 0.6
            else:
                dte_score = 0.4
            
            score += dte_score * 0.15 * 100  # Scale to 0-15 points
        
        # Risk-reward ratio (weight: 10%)
        risk_reward = strategy.get_risk_reward_ratio()
        if risk_reward:
            # Lower risk-reward ratio is better (invert and normalize)
            # Typical range 1-5, optimal around 2-3
            if risk_reward <= 3:
                rr_score = 1.0
            elif risk_reward <= 5:
                rr_score = 0.6
            else:
                rr_score = 0.2
            
            score += rr_score * 0.10 * 100  # Scale to 0-10 points
        
        return max(0, score)  # Ensure non-negative score


class StrategyGeneratorFactory:
    """Factory for creating strategy generators with different configurations."""
    
    @staticmethod
    def create_conservative_generator() -> StrategyGenerator:
        """Create generator optimized for conservative strategies."""
        return StrategyGenerator()
    
    @staticmethod
    def create_aggressive_generator() -> StrategyGenerator:
        """Create generator optimized for aggressive strategies."""
        return StrategyGenerator()
    
    @staticmethod
    def create_income_generator() -> StrategyGenerator:
        """Create generator optimized for income generation."""
        return StrategyGenerator()
    
    @staticmethod
    def create_default_config(underlying_symbol: str, 
                            underlying_price: Decimal,
                            mode: StrategyGenerationMode = StrategyGenerationMode.BALANCED) -> StrategyGenerationConfig:
        """Create default configuration for strategy generation."""
        return StrategyGenerationConfig(
            underlying_symbol=underlying_symbol,
            underlying_price=underlying_price,
            mode=mode
        )