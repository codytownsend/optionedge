"""
Strategy domain entities for options trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import math

from .option_contract import OptionContract
from ...data.models.trades import StrategyType, TradeDirection, TradeLeg
from ...data.models.options import OptionType, Greeks


class StrategyCategory(str, Enum):
    """Strategy categories for classification."""
    INCOME = "income"              # Credit strategies for income
    SPECULATION = "speculation"    # Directional bets
    HEDGING = "hedging"           # Risk mitigation
    ARBITRAGE = "arbitrage"       # Risk-free profit
    VOLATILITY = "volatility"     # Volatility plays


class MarketOutlook(str, Enum):
    """Market outlook for strategy selection."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


class Strategy(ABC):
    """
    Abstract base class for all options trading strategies.
    
    Defines the interface and common behavior for all strategy types.
    Each concrete strategy implements specific logic for construction,
    risk calculation, and profit/loss analysis.
    """
    
    def __init__(self, 
                 underlying_symbol: str,
                 strategy_type: StrategyType,
                 legs: List[TradeLeg],
                 underlying_price: Optional[Decimal] = None):
        """
        Initialize strategy.
        
        Args:
            underlying_symbol: Symbol of underlying asset
            strategy_type: Type of strategy
            legs: List of trade legs
            underlying_price: Current underlying price
        """
        self.underlying_symbol = underlying_symbol
        self.strategy_type = strategy_type
        self.legs = legs
        self.underlying_price = underlying_price
        self._cached_calculations: Dict[str, Any] = {}
        
        # Validate strategy
        self._validate_strategy()
    
    @abstractmethod
    def get_category(self) -> StrategyCategory:
        """Get strategy category."""
        pass
    
    @abstractmethod
    def get_market_outlook(self) -> MarketOutlook:
        """Get market outlook for this strategy."""
        pass
    
    @abstractmethod
    def calculate_max_profit(self) -> Optional[Decimal]:
        """Calculate maximum profit potential."""
        pass
    
    @abstractmethod
    def calculate_max_loss(self) -> Optional[Decimal]:
        """Calculate maximum loss potential."""
        pass
    
    @abstractmethod
    def calculate_breakeven_points(self) -> List[Decimal]:
        """Calculate breakeven points."""
        pass
    
    @abstractmethod
    def get_profit_loss_at_expiration(self, spot_prices: List[Decimal]) -> List[Tuple[Decimal, Decimal]]:
        """Calculate P&L at expiration for given spot prices."""
        pass
    
    def calculate_net_premium(self) -> Decimal:
        """Calculate net premium received (credit) or paid (debit)."""
        cache_key = "net_premium"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        total_premium = Decimal('0')
        
        for leg in self.legs:
            if leg.option.mark:
                leg_premium = leg.option.mark * leg.quantity * 100  # Standard multiplier
                if leg.direction == TradeDirection.SHORT:
                    total_premium += leg_premium  # Credit
                else:
                    total_premium -= leg_premium  # Debit
        
        self._cached_calculations[cache_key] = total_premium
        return total_premium
    
    def calculate_net_greeks(self) -> Greeks:
        """Calculate net Greeks for the strategy."""
        cache_key = "net_greeks"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        
        for leg in self.legs:
            if leg.option.greeks:
                multiplier = leg.quantity * 100  # Standard option multiplier
                sign = 1 if leg.direction == TradeDirection.LONG else -1
                
                if leg.option.greeks.delta:
                    total_delta += leg.option.greeks.delta * multiplier * sign
                if leg.option.greeks.gamma:
                    total_gamma += leg.option.greeks.gamma * multiplier * sign
                if leg.option.greeks.theta:
                    total_theta += leg.option.greeks.theta * multiplier * sign
                if leg.option.greeks.vega:
                    total_vega += leg.option.greeks.vega * multiplier * sign
                if leg.option.greeks.rho:
                    total_rho += leg.option.greeks.rho * multiplier * sign
        
        net_greeks = Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho
        )
        
        self._cached_calculations[cache_key] = net_greeks
        return net_greeks
    
    def calculate_probability_of_profit(self, volatility: Optional[float] = None) -> Optional[float]:
        """
        Calculate probability of profit using Monte Carlo simulation.
        
        Args:
            volatility: Annual volatility (use average IV if not provided)
            
        Returns:
            Probability between 0 and 1, or None if insufficient data
        """
        if not self.underlying_price:
            return None
        
        # Use average implied volatility if not provided
        if volatility is None:
            ivs = [leg.option.implied_volatility for leg in self.legs 
                   if leg.option.implied_volatility]
            if not ivs:
                return None
            volatility = sum(ivs) / len(ivs)
        
        cache_key = f"pop_{volatility}"
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        # Get the nearest expiration date
        expiration_dates = [leg.option.expiration for leg in self.legs]
        nearest_expiration = min(expiration_dates)
        days_to_expiration = (nearest_expiration - date.today()).days
        
        if days_to_expiration <= 0:
            return None
        
        # Monte Carlo simulation
        num_simulations = 1000
        profitable_outcomes = 0
        
        for _ in range(num_simulations):
            # Generate random price at expiration
            final_price = self._simulate_final_price(
                float(self.underlying_price), 
                volatility, 
                days_to_expiration / 365.0
            )
            
            # Calculate P&L at this price
            pnl = self._calculate_pnl_at_price(Decimal(str(final_price)))
            
            if pnl > 0:
                profitable_outcomes += 1
        
        prob = profitable_outcomes / num_simulations
        self._cached_calculations[cache_key] = prob
        return prob
    
    def get_days_to_expiration(self) -> int:
        """Get days to nearest expiration."""
        expiration_dates = [leg.option.expiration for leg in self.legs]
        nearest_expiration = min(expiration_dates)
        return (nearest_expiration - date.today()).days
    
    def get_primary_expiration(self) -> date:
        """Get the primary expiration date (usually the nearest)."""
        expiration_dates = [leg.option.expiration for leg in self.legs]
        return min(expiration_dates)
    
    def is_credit_strategy(self) -> bool:
        """Check if strategy receives net credit."""
        return self.calculate_net_premium() > 0
    
    def is_debit_strategy(self) -> bool:
        """Check if strategy pays net debit."""
        return self.calculate_net_premium() < 0
    
    def calculate_capital_required(self) -> Decimal:
        """Calculate capital/margin required for strategy."""
        net_premium = self.calculate_net_premium()
        max_loss = self.calculate_max_loss()
        
        if self.is_credit_strategy():
            # For credit strategies, capital required is max loss minus credit received
            return abs(max_loss or Decimal('0')) - net_premium
        else:
            # For debit strategies, capital required is the debit paid
            return abs(net_premium)
    
    def calculate_return_on_capital(self) -> Optional[float]:
        """Calculate annualized return on capital."""
        max_profit = self.calculate_max_profit()
        capital_required = self.calculate_capital_required()
        days_to_exp = self.get_days_to_expiration()
        
        if not max_profit or capital_required <= 0 or days_to_exp <= 0:
            return None
        
        # Annualized return
        daily_return = float(max_profit / capital_required)
        annual_return = daily_return * (365 / days_to_exp)
        
        return annual_return
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio (max_loss / max_profit)."""
        max_profit = self.calculate_max_profit()
        max_loss = self.calculate_max_loss()
        
        if not max_profit or not max_loss or max_profit <= 0:
            return None
        
        return float(abs(max_loss) / max_profit)
    
    def _validate_strategy(self) -> None:
        """Validate strategy construction."""
        if not self.legs:
            raise ValueError("Strategy must have at least one leg")
        
        # Check all legs have same underlying
        underlyings = {leg.option.underlying for leg in self.legs}
        if len(underlyings) > 1:
            raise ValueError("All legs must have same underlying")
        
        if self.underlying_symbol not in underlyings:
            raise ValueError("Leg underlyings don't match strategy underlying")
    
    def _simulate_final_price(self, current_price: float, volatility: float, time_to_expiry: float) -> float:
        """Simulate final price using geometric Brownian motion."""
        import random
        
        # Risk-free rate (simplified)
        r = 0.05
        
        # Generate random normal variable
        z = random.gauss(0, 1)
        
        # Geometric Brownian motion
        drift = (r - 0.5 * volatility**2) * time_to_expiry
        diffusion = volatility * math.sqrt(time_to_expiry) * z
        
        final_price = current_price * math.exp(drift + diffusion)
        return final_price
    
    def _calculate_pnl_at_price(self, spot_price: Decimal) -> Decimal:
        """Calculate P&L at expiration for a given spot price."""
        total_pnl = Decimal('0')
        
        for leg in self.legs:
            option = leg.option
            
            # Calculate intrinsic value at expiration
            if option.option_type == OptionType.CALL:
                intrinsic = max(Decimal('0'), spot_price - option.strike)
            else:  # PUT
                intrinsic = max(Decimal('0'), option.strike - spot_price)
            
            # Calculate leg P&L
            if leg.direction == TradeDirection.LONG:
                leg_pnl = (intrinsic - (option.mark or Decimal('0'))) * leg.quantity * 100
            else:  # SHORT
                leg_pnl = ((option.mark or Decimal('0')) - intrinsic) * leg.quantity * 100
            
            total_pnl += leg_pnl
        
        return total_pnl
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy summary."""
        return {
            'underlying_symbol': self.underlying_symbol,
            'strategy_type': self.strategy_type,
            'category': self.get_category(),
            'market_outlook': self.get_market_outlook(),
            'num_legs': len(self.legs),
            'is_credit': self.is_credit_strategy(),
            'net_premium': self.calculate_net_premium(),
            'max_profit': self.calculate_max_profit(),
            'max_loss': self.calculate_max_loss(),
            'breakeven_points': self.calculate_breakeven_points(),
            'probability_of_profit': self.calculate_probability_of_profit(),
            'days_to_expiration': self.get_days_to_expiration(),
            'capital_required': self.calculate_capital_required(),
            'return_on_capital': self.calculate_return_on_capital(),
            'risk_reward_ratio': self.get_risk_reward_ratio(),
            'net_greeks': self.calculate_net_greeks(),
        }


class CreditSpreadStrategy(Strategy):
    """Implementation of credit spread strategies (put and call credit spreads)."""
    
    def __init__(self, 
                 underlying_symbol: str,
                 short_leg: TradeLeg,
                 long_leg: TradeLeg,
                 underlying_price: Optional[Decimal] = None):
        """
        Initialize credit spread strategy.
        
        Args:
            underlying_symbol: Underlying symbol
            short_leg: Short option leg (sold)
            long_leg: Long option leg (bought for protection)
            underlying_price: Current underlying price
        """
        # Determine strategy type
        if short_leg.option.option_type == OptionType.PUT:
            strategy_type = StrategyType.PUT_CREDIT_SPREAD
        else:
            strategy_type = StrategyType.CALL_CREDIT_SPREAD
        
        super().__init__(underlying_symbol, strategy_type, [short_leg, long_leg], underlying_price)
        
        self.short_leg = short_leg
        self.long_leg = long_leg
        
        # Validate credit spread structure
        self._validate_credit_spread()
    
    def get_category(self) -> StrategyCategory:
        """Credit spreads are income strategies."""
        return StrategyCategory.INCOME
    
    def get_market_outlook(self) -> MarketOutlook:
        """Market outlook depends on spread type."""
        if self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            return MarketOutlook.BULLISH
        else:  # CALL_CREDIT_SPREAD
            return MarketOutlook.BEARISH
    
    def calculate_max_profit(self) -> Optional[Decimal]:
        """Maximum profit is the net credit received."""
        return self.calculate_net_premium()
    
    def calculate_max_loss(self) -> Optional[Decimal]:
        """Maximum loss is spread width minus credit received."""
        short_strike = self.short_leg.option.strike
        long_strike = self.long_leg.option.strike
        
        spread_width = abs(long_strike - short_strike) * 100  # Per contract
        net_credit = self.calculate_net_premium()
        
        max_loss = spread_width - net_credit
        return -max_loss  # Negative because it's a loss
    
    def calculate_breakeven_points(self) -> List[Decimal]:
        """Calculate breakeven point."""
        short_strike = self.short_leg.option.strike
        net_credit = self.calculate_net_premium() / 100  # Per share
        
        if self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            breakeven = short_strike - net_credit
        else:  # CALL_CREDIT_SPREAD
            breakeven = short_strike + net_credit
        
        return [breakeven]
    
    def get_profit_loss_at_expiration(self, spot_prices: List[Decimal]) -> List[Tuple[Decimal, Decimal]]:
        """Calculate P&L at expiration for given spot prices."""
        results = []
        
        for spot_price in spot_prices:
            pnl = self._calculate_pnl_at_price(spot_price)
            results.append((spot_price, pnl))
        
        return results
    
    def _validate_credit_spread(self) -> None:
        """Validate credit spread structure."""
        # Check directions
        if self.short_leg.direction != TradeDirection.SHORT:
            raise ValueError("Short leg must be short direction")
        if self.long_leg.direction != TradeDirection.LONG:
            raise ValueError("Long leg must be long direction")
        
        # Check option types match
        if self.short_leg.option.option_type != self.long_leg.option.option_type:
            raise ValueError("Both legs must be same option type")
        
        # Check quantities match
        if self.short_leg.quantity != self.long_leg.quantity:
            raise ValueError("Leg quantities must match")
        
        # Check expirations match
        if self.short_leg.option.expiration != self.long_leg.option.expiration:
            raise ValueError("Leg expirations must match")
        
        # Check strike relationship
        short_strike = self.short_leg.option.strike
        long_strike = self.long_leg.option.strike
        
        if self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            if short_strike <= long_strike:
                raise ValueError("Put credit spread: short strike must be > long strike")
        else:  # CALL_CREDIT_SPREAD
            if short_strike >= long_strike:
                raise ValueError("Call credit spread: short strike must be < long strike")


class IronCondorStrategy(Strategy):
    """Implementation of iron condor strategy."""
    
    def __init__(self,
                 underlying_symbol: str,
                 put_spread_short: TradeLeg,
                 put_spread_long: TradeLeg,
                 call_spread_short: TradeLeg,
                 call_spread_long: TradeLeg,
                 underlying_price: Optional[Decimal] = None):
        """Initialize iron condor strategy."""
        legs = [put_spread_short, put_spread_long, call_spread_short, call_spread_long]
        
        super().__init__(underlying_symbol, StrategyType.IRON_CONDOR, legs, underlying_price)
        
        self.put_spread_short = put_spread_short
        self.put_spread_long = put_spread_long
        self.call_spread_short = call_spread_short
        self.call_spread_long = call_spread_long
        
        self._validate_iron_condor()
    
    def get_category(self) -> StrategyCategory:
        """Iron condors are income strategies."""
        return StrategyCategory.INCOME
    
    def get_market_outlook(self) -> MarketOutlook:
        """Iron condors profit from low volatility/range-bound markets."""
        return MarketOutlook.NEUTRAL
    
    def calculate_max_profit(self) -> Optional[Decimal]:
        """Maximum profit is the net credit received."""
        return self.calculate_net_premium()
    
    def calculate_max_loss(self) -> Optional[Decimal]:
        """Maximum loss is the larger spread width minus credit received."""
        # Put spread width
        put_width = abs(self.put_spread_short.option.strike - self.put_spread_long.option.strike) * 100
        
        # Call spread width
        call_width = abs(self.call_spread_short.option.strike - self.call_spread_long.option.strike) * 100
        
        # Use larger width (they should be equal in standard iron condor)
        spread_width = max(put_width, call_width)
        net_credit = self.calculate_net_premium()
        
        max_loss = spread_width - net_credit
        return -max_loss  # Negative because it's a loss
    
    def calculate_breakeven_points(self) -> List[Decimal]:
        """Calculate both breakeven points."""
        net_credit_per_share = self.calculate_net_premium() / 100
        
        # Lower breakeven (put side)
        lower_breakeven = self.put_spread_short.option.strike - net_credit_per_share
        
        # Upper breakeven (call side)
        upper_breakeven = self.call_spread_short.option.strike + net_credit_per_share
        
        return [lower_breakeven, upper_breakeven]
    
    def get_profit_loss_at_expiration(self, spot_prices: List[Decimal]) -> List[Tuple[Decimal, Decimal]]:
        """Calculate P&L at expiration for given spot prices."""
        results = []
        
        for spot_price in spot_prices:
            pnl = self._calculate_pnl_at_price(spot_price)
            results.append((spot_price, pnl))
        
        return results
    
    def _validate_iron_condor(self) -> None:
        """Validate iron condor structure."""
        # Check we have exactly 4 legs
        if len(self.legs) != 4:
            raise ValueError("Iron condor must have exactly 4 legs")
        
        # Check option types
        if self.put_spread_short.option.option_type != OptionType.PUT:
            raise ValueError("Put spread short must be a put")
        if self.put_spread_long.option.option_type != OptionType.PUT:
            raise ValueError("Put spread long must be a put")
        if self.call_spread_short.option.option_type != OptionType.CALL:
            raise ValueError("Call spread short must be a call")
        if self.call_spread_long.option.option_type != OptionType.CALL:
            raise ValueError("Call spread long must be a call")
        
        # Check directions
        if self.put_spread_short.direction != TradeDirection.SHORT:
            raise ValueError("Put spread short must be short direction")
        if self.call_spread_short.direction != TradeDirection.SHORT:
            raise ValueError("Call spread short must be short direction")
        if self.put_spread_long.direction != TradeDirection.LONG:
            raise ValueError("Put spread long must be long direction")
        if self.call_spread_long.direction != TradeDirection.LONG:
            raise ValueError("Call spread long must be long direction")


class StrategyBuilder:
    """Builder for constructing various options strategies."""
    
    def __init__(self, underlying_symbol: str):
        self.underlying_symbol = underlying_symbol
        self.underlying_price: Optional[Decimal] = None
        self.legs: List[TradeLeg] = []
    
    def with_underlying_price(self, price: Decimal) -> 'StrategyBuilder':
        """Set underlying price."""
        self.underlying_price = price
        return self
    
    def add_leg(self, leg: TradeLeg) -> 'StrategyBuilder':
        """Add a leg to the strategy."""
        self.legs.append(leg)
        return self
    
    def build_credit_spread(self, short_leg: TradeLeg, long_leg: TradeLeg) -> CreditSpreadStrategy:
        """Build a credit spread strategy."""
        return CreditSpreadStrategy(
            self.underlying_symbol,
            short_leg,
            long_leg,
            self.underlying_price
        )
    
    def build_iron_condor(self,
                         put_short: TradeLeg,
                         put_long: TradeLeg,
                         call_short: TradeLeg,
                         call_long: TradeLeg) -> IronCondorStrategy:
        """Build an iron condor strategy."""
        return IronCondorStrategy(
            self.underlying_symbol,
            put_short,
            put_long,
            call_short,
            call_long,
            self.underlying_price
        )
    
    def build_generic_strategy(self, strategy_type: StrategyType) -> Strategy:
        """Build a generic strategy from accumulated legs."""
        if not self.legs:
            raise ValueError("No legs added to strategy")
        
        # For now, return a generic strategy (could be extended)
        class GenericStrategy(Strategy):
            def get_category(self) -> StrategyCategory:
                return StrategyCategory.SPECULATION
            
            def get_market_outlook(self) -> MarketOutlook:
                return MarketOutlook.NEUTRAL
            
            def calculate_max_profit(self) -> Optional[Decimal]:
                # Simplified calculation
                return None
            
            def calculate_max_loss(self) -> Optional[Decimal]:
                # Simplified calculation
                return None
            
            def calculate_breakeven_points(self) -> List[Decimal]:
                return []
            
            def get_profit_loss_at_expiration(self, spot_prices: List[Decimal]) -> List[Tuple[Decimal, Decimal]]:
                return [(price, self._calculate_pnl_at_price(price)) for price in spot_prices]
        
        return GenericStrategy(
            self.underlying_symbol,
            strategy_type,
            self.legs.copy(),
            self.underlying_price
        )