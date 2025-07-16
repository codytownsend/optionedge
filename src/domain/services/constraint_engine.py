"""
Hard constraint validation system for trade filtering and portfolio risk controls.
"""

from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ...data.models.options import OptionQuote, OptionType, Greeks
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, TradeLeg, TradeDirection,
    PortfolioGreeks, TradeFilterCriteria
)
from ...infrastructure.error_handling import (
    handle_errors, ConstraintViolationError, BusinessLogicError
)


class ConstraintType(Enum):
    """Types of constraints for validation."""
    HARD_CONSTRAINT = "hard_constraint"  # Must pass, rejects trade
    SOFT_CONSTRAINT = "soft_constraint"  # Warning only, affects scoring
    PORTFOLIO_LIMIT = "portfolio_limit"  # Portfolio-level constraint


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    CRITICAL = "critical"      # Immediate rejection
    WARNING = "warning"        # Score penalty
    INFO = "info"             # Informational only


@dataclass
class ConstraintViolation:
    """Constraint violation details."""
    constraint_name: str
    violation_type: ConstraintType
    severity: ConstraintSeverity
    message: str
    current_value: Any
    required_value: Any
    recommendation: Optional[str] = None


@dataclass
class ConstraintValidationResult:
    """Result of constraint validation."""
    is_valid: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[ConstraintViolation] = field(default_factory=list)
    score_penalty: float = 0.0
    validation_summary: str = ""


# GICS Sector Classification Mapping
GICS_SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology", 
    "GOOGL": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "JNJ": "Health Care",
    "XOM": "Energy",
    "PG": "Consumer Staples",
    "UNH": "Health Care",
    "V": "Information Technology",
    "WMT": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "MA": "Information Technology",
    "BAC": "Financials",
    "PFE": "Health Care",
    "KO": "Consumer Staples",
    "DIS": "Communication Services",
    "ADBE": "Information Technology",
    "NFLX": "Communication Services",
    "CRM": "Information Technology",
    "NVDA": "Information Technology",
    "PYPL": "Information Technology",
    "INTC": "Information Technology",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "MRK": "Health Care",
    "ABT": "Health Care",
    "CVX": "Energy",
    "WFC": "Financials",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "TMO": "Health Care",
    "ACN": "Information Technology",
    "COST": "Consumer Staples",
    "DHR": "Health Care",
    "NEE": "Utilities",
    "TXN": "Information Technology",
    "RTX": "Industrials",
    "HON": "Industrials",
    "QCOM": "Information Technology",
    "UPS": "Industrials",
    "IBM": "Information Technology",
    "MDT": "Health Care",
    "LIN": "Materials",
    "AMGN": "Health Care",
    "LOW": "Consumer Discretionary",
    "PM": "Consumer Staples",
    "BMY": "Health Care",
    "SPGI": "Financials",
    "CAT": "Industrials",
    "GS": "Financials"
}


class HardConstraintValidator:
    """
    Hard constraint validation system implementing all filtering rules from instructions.txt.
    
    Features:
    - Quote freshness validation (≤10 minutes)
    - Probability of profit enforcement (≥0.65)
    - Credit-to-max-loss ratio validation (≥0.33)
    - Maximum loss per trade limits (≤$500)
    - Capital availability verification
    - Liquidity requirements enforcement
    - Portfolio Greeks limits validation
    - GICS sector diversification (max 2 per sector)
    - Margin requirement calculations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Default constraint parameters (can be overridden)
        self.default_constraints = {
            'max_quote_age_minutes': 10,
            'min_probability_of_profit': 0.65,
            'min_credit_to_max_loss_ratio': 0.33,
            'max_loss_per_trade': 500,
            'max_loss_pct_of_nav': 0.005,  # 0.5% of NAV
            'min_open_interest': 50,
            'max_bid_ask_spread_pct': 0.05,  # 5%
            'min_daily_volume': 10,
            'max_delta_per_100k': 0.30,
            'min_vega_per_100k': -0.05,
            'max_trades_per_sector': 2
        }
    
    @handle_errors(operation_name="validate_trade_constraints")
    def validate_trade_constraints(
        self,
        trade_candidate: TradeCandidate,
        existing_trades: List[TradeCandidate],
        nav: Decimal,
        available_capital: Decimal,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> ConstraintValidationResult:
        """
        Validate all hard constraints for a trade candidate.
        
        Args:
            trade_candidate: Trade to validate
            existing_trades: Current portfolio trades
            nav: Portfolio net asset value
            available_capital: Available capital for new trades
            custom_constraints: Override default constraint parameters
            
        Returns:
            Validation result with violations and recommendations
        """
        self.logger.info(f"Validating constraints for {trade_candidate.strategy.strategy_type.value}")
        
        # Merge custom constraints with defaults
        constraints = {**self.default_constraints, **(custom_constraints or {})}
        
        violations = []
        warnings = []
        is_valid = True
        
        # 1. Quote Freshness Validation
        quote_result = self._validate_quote_freshness(
            trade_candidate.strategy, constraints['max_quote_age_minutes']
        )
        if quote_result:
            violations.append(quote_result)
            is_valid = False
        
        # 2. Probability of Profit Filter
        pop_result = self._validate_pop_constraint(
            trade_candidate.strategy, constraints['min_probability_of_profit']
        )
        if pop_result:
            violations.append(pop_result)
            is_valid = False
        
        # 3. Credit-to-Max-Loss Ratio
        credit_ratio_result = self._validate_credit_ratio(
            trade_candidate.strategy, constraints['min_credit_to_max_loss_ratio']
        )
        if credit_ratio_result:
            violations.append(credit_ratio_result)
            is_valid = False
        
        # 4. Maximum Loss Per Trade
        max_loss_result = self._validate_max_loss(
            trade_candidate.strategy, nav, constraints
        )
        if max_loss_result:
            violations.append(max_loss_result)
            is_valid = False
        
        # 5. Capital Availability Check
        capital_result = self._validate_capital_requirement(
            trade_candidate.strategy, available_capital
        )
        if capital_result:
            violations.append(capital_result)
            is_valid = False
        
        # 6. Liquidity Validation
        liquidity_result = self._validate_liquidity(
            trade_candidate.strategy.legs, constraints
        )
        if liquidity_result:
            violations.append(liquidity_result)
            is_valid = False
        
        # 7. Portfolio Greeks Validation
        greeks_result = self._validate_portfolio_greeks(
            trade_candidate, existing_trades, nav, constraints
        )
        if greeks_result:
            violations.append(greeks_result)
            is_valid = False
        
        # 8. Sector Diversification
        sector_result = self._validate_sector_diversification(
            trade_candidate.strategy, existing_trades, constraints['max_trades_per_sector']
        )
        if sector_result:
            violations.append(sector_result)
            is_valid = False
        
        # Generate summary
        summary = self._generate_validation_summary(is_valid, violations, warnings)
        
        return ConstraintValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            validation_summary=summary
        )
    
    def _validate_quote_freshness(
        self,
        strategy: StrategyDefinition,
        max_age_minutes: int
    ) -> Optional[ConstraintViolation]:
        """Reject quotes older than specified minutes."""
        
        current_time = datetime.now(timezone.utc)
        
        for leg in strategy.legs:
            if not leg.option.quote_timestamp:
                return ConstraintViolation(
                    constraint_name="quote_freshness",
                    violation_type=ConstraintType.HARD_CONSTRAINT,
                    severity=ConstraintSeverity.CRITICAL,
                    message="Missing quote timestamp",
                    current_value="None",
                    required_value=f"≤{max_age_minutes} minutes",
                    recommendation="Refresh market data and retry"
                )
            
            try:
                if isinstance(leg.option.quote_timestamp, str):
                    quote_time = datetime.fromisoformat(leg.option.quote_timestamp.replace('Z', '+00:00'))
                else:
                    quote_time = leg.option.quote_timestamp
                
                if quote_time.tzinfo is None:
                    quote_time = quote_time.replace(tzinfo=timezone.utc)
                
                age_minutes = (current_time - quote_time).total_seconds() / 60
                
                if age_minutes > max_age_minutes:
                    return ConstraintViolation(
                        constraint_name="quote_freshness",
                        violation_type=ConstraintType.HARD_CONSTRAINT,
                        severity=ConstraintSeverity.CRITICAL,
                        message=f"Quote too old: {age_minutes:.1f} minutes",
                        current_value=f"{age_minutes:.1f} minutes",
                        required_value=f"≤{max_age_minutes} minutes",
                        recommendation="Refresh market data and retry"
                    )
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse quote timestamp: {str(e)}")
                return ConstraintViolation(
                    constraint_name="quote_freshness",
                    violation_type=ConstraintType.HARD_CONSTRAINT,
                    severity=ConstraintSeverity.CRITICAL,
                    message="Invalid quote timestamp format",
                    current_value=str(leg.option.quote_timestamp),
                    required_value="Valid ISO timestamp",
                    recommendation="Fix quote timestamp format"
                )
        
        return None
    
    def _validate_pop_constraint(
        self,
        strategy: StrategyDefinition,
        min_pop: float
    ) -> Optional[ConstraintViolation]:
        """Ensure POP >= minimum threshold."""
        
        if not strategy.probability_of_profit:
            return ConstraintViolation(
                constraint_name="probability_of_profit",
                violation_type=ConstraintType.HARD_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                message="Missing probability of profit calculation",
                current_value="None",
                required_value=f"≥{min_pop:.1%}",
                recommendation="Calculate POP using Black-Scholes or Monte Carlo"
            )
        
        if strategy.probability_of_profit < min_pop:
            return ConstraintViolation(
                constraint_name="probability_of_profit",
                violation_type=ConstraintType.HARD_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"POP too low: {strategy.probability_of_profit:.1%}",
                current_value=f"{strategy.probability_of_profit:.1%}",
                required_value=f"≥{min_pop:.1%}",
                recommendation="Select higher probability strikes or different strategy"
            )
        
        return None
    
    def _validate_credit_ratio(
        self,
        strategy: StrategyDefinition,
        min_ratio: float
    ) -> Optional[ConstraintViolation]:
        """Ensure credit-to-max-loss ratio >= minimum threshold."""
        
        if not strategy.credit_to_max_loss_ratio:
            # Try to calculate if components are available
            if strategy.net_credit and strategy.max_loss and strategy.max_loss > 0:
                ratio = float(strategy.net_credit) / float(strategy.max_loss)
                strategy.credit_to_max_loss_ratio = ratio
            else:
                return ConstraintViolation(
                    constraint_name="credit_to_max_loss_ratio",
                    violation_type=ConstraintType.HARD_CONSTRAINT,
                    severity=ConstraintSeverity.CRITICAL,
                    message="Missing credit-to-max-loss ratio",
                    current_value="None",
                    required_value=f"≥{min_ratio:.1%}",
                    recommendation="Calculate strategy risk metrics"
                )
        
        if strategy.credit_to_max_loss_ratio < min_ratio:
            return ConstraintViolation(
                constraint_name="credit_to_max_loss_ratio",
                violation_type=ConstraintType.HARD_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Credit ratio too low: {strategy.credit_to_max_loss_ratio:.1%}",
                current_value=f"{strategy.credit_to_max_loss_ratio:.1%}",
                required_value=f"≥{min_ratio:.1%}",
                recommendation="Select strikes with better risk/reward profile"
            )
        
        return None
    
    def _validate_max_loss(
        self,
        strategy: StrategyDefinition,
        nav: Decimal,
        constraints: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Ensure max loss <= $500 or 0.5% of NAV, whichever is lower."""
        
        if not strategy.max_loss:
            return ConstraintViolation(
                constraint_name="maximum_loss",
                violation_type=ConstraintType.HARD_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                message="Missing maximum loss calculation",
                current_value="None",
                required_value=f"≤${constraints['max_loss_per_trade']}",
                recommendation="Calculate strategy maximum loss"
            )
        
        max_allowed_loss = min(
            constraints['max_loss_per_trade'],
            float(nav) * constraints['max_loss_pct_of_nav']
        )
        
        if float(strategy.max_loss) > max_allowed_loss:
            return ConstraintViolation(
                constraint_name="maximum_loss",
                violation_type=ConstraintType.HARD_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Max loss too high: ${strategy.max_loss}",
                current_value=f"${strategy.max_loss}",
                required_value=f"≤${max_allowed_loss:.0f}",
                recommendation="Reduce position size or select different strikes"
            )
        
        return None
    
    def _validate_capital_requirement(
        self,
        strategy: StrategyDefinition,
        available_capital: Decimal
    ) -> Optional[ConstraintViolation]:
        """Check if strategy fits within available capital."""
        
        required_capital = self._calculate_margin_requirement(strategy)
        
        if required_capital > available_capital:
            return ConstraintViolation(
                constraint_name="capital_requirement",
                violation_type=ConstraintType.HARD_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Insufficient capital: need ${required_capital}, have ${available_capital}",
                current_value=f"${available_capital}",
                required_value=f"≥${required_capital}",
                recommendation="Add capital or reduce position size"
            )
        
        return None
    
    def _validate_liquidity(
        self,
        option_legs: List[TradeLeg],
        constraints: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Check minimum liquidity requirements."""
        
        for i, leg in enumerate(option_legs):
            option = leg.option
            
            # Minimum open interest
            if not option.open_interest or option.open_interest < constraints['min_open_interest']:
                return ConstraintViolation(
                    constraint_name="liquidity",
                    violation_type=ConstraintType.HARD_CONSTRAINT,
                    severity=ConstraintSeverity.CRITICAL,
                    message=f"Low open interest on leg {i+1}: {option.open_interest}",
                    current_value=option.open_interest or 0,
                    required_value=f"≥{constraints['min_open_interest']}",
                    recommendation="Select more liquid options with higher OI"
                )
            
            # Maximum bid-ask spread
            if option.bid and option.ask and option.bid > 0 and option.ask > 0:
                mid_price = (option.bid + option.ask) / 2
                spread_pct = (option.ask - option.bid) / mid_price if mid_price > 0 else 1
                
                if spread_pct > constraints['max_bid_ask_spread_pct']:
                    return ConstraintViolation(
                        constraint_name="liquidity",
                        violation_type=ConstraintType.HARD_CONSTRAINT,
                        severity=ConstraintSeverity.CRITICAL,
                        message=f"Wide bid-ask spread on leg {i+1}: {spread_pct:.1%}",
                        current_value=f"{spread_pct:.1%}",
                        required_value=f"≤{constraints['max_bid_ask_spread_pct']:.1%}",
                        recommendation="Select options with tighter spreads"
                    )
            
            # Minimum daily volume
            if not option.volume or option.volume < constraints['min_daily_volume']:
                return ConstraintViolation(
                    constraint_name="liquidity",
                    violation_type=ConstraintType.HARD_CONSTRAINT,
                    severity=ConstraintSeverity.WARNING,  # Warning level for volume
                    message=f"Low volume on leg {i+1}: {option.volume}",
                    current_value=option.volume or 0,
                    required_value=f"≥{constraints['min_daily_volume']}",
                    recommendation="Consider more liquid options or proceed with caution"
                )
        
        return None
    
    def _validate_portfolio_greeks(
        self,
        new_trade: TradeCandidate,
        existing_trades: List[TradeCandidate],
        nav: Decimal,
        constraints: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Validate portfolio Greeks against limits."""
        
        # Calculate portfolio Greeks with new trade added
        portfolio_greeks = self._calculate_portfolio_greeks(existing_trades, new_trade, nav)
        
        nav_factor = float(nav) / 100000  # Reference NAV of $100k
        
        # Delta limits: [-0.30, +0.30] × (NAV / 100k)
        max_delta = constraints['max_delta_per_100k'] * nav_factor
        if not (-max_delta <= portfolio_greeks['delta'] <= max_delta):
            return ConstraintViolation(
                constraint_name="portfolio_delta",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Portfolio delta limit exceeded: {portfolio_greeks['delta']:.2f}",
                current_value=f"{portfolio_greeks['delta']:.2f}",
                required_value=f"[{-max_delta:.2f}, {max_delta:.2f}]",
                recommendation="Hedge delta exposure or reduce position size"
            )
        
        # Vega limit: >= -0.05 × (NAV / 100k)
        min_vega = constraints['min_vega_per_100k'] * nav_factor
        if portfolio_greeks['vega'] < min_vega:
            return ConstraintViolation(
                constraint_name="portfolio_vega",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Portfolio vega limit exceeded: {portfolio_greeks['vega']:.2f}",
                current_value=f"{portfolio_greeks['vega']:.2f}",
                required_value=f"≥{min_vega:.2f}",
                recommendation="Add long vega positions or reduce short vega exposure"
            )
        
        return None
    
    def _validate_sector_diversification(
        self,
        new_strategy: StrategyDefinition,
        existing_trades: List[TradeCandidate],
        max_trades_per_sector: int
    ) -> Optional[ConstraintViolation]:
        """Ensure max trades per GICS sector."""
        
        # Count trades by sector
        sector_count = {}
        
        for trade in existing_trades:
            sector = GICS_SECTORS.get(trade.strategy.underlying, "Unknown")
            sector_count[sector] = sector_count.get(sector, 0) + 1
        
        # Check new trade sector
        new_trade_sector = GICS_SECTORS.get(new_strategy.underlying, "Unknown")
        current_count = sector_count.get(new_trade_sector, 0)
        
        if current_count >= max_trades_per_sector:
            return ConstraintViolation(
                constraint_name="sector_diversification",
                violation_type=ConstraintType.PORTFOLIO_LIMIT,
                severity=ConstraintSeverity.CRITICAL,
                message=f"Sector limit exceeded for {new_trade_sector}: {current_count} trades",
                current_value=current_count,
                required_value=f"<{max_trades_per_sector}",
                recommendation=f"Select trades from different GICS sectors"
            )
        
        return None
    
    def _calculate_portfolio_greeks(
        self,
        existing_trades: List[TradeCandidate],
        new_trade: TradeCandidate,
        nav: Decimal
    ) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks including new trade."""
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        # Sum existing positions
        for trade in existing_trades:
            for leg in trade.strategy.legs:
                multiplier = 100  # Options multiplier
                position_sign = 1 if leg.direction == TradeDirection.BUY else -1
                
                if leg.option.greeks:
                    total_delta += float(leg.option.greeks.delta or 0) * leg.quantity * multiplier * position_sign
                    total_gamma += float(leg.option.greeks.gamma or 0) * leg.quantity * multiplier * position_sign
                    total_theta += float(leg.option.greeks.theta or 0) * leg.quantity * multiplier * position_sign
                    total_vega += float(leg.option.greeks.vega or 0) * leg.quantity * multiplier * position_sign
        
        # Add new trade Greeks
        for leg in new_trade.strategy.legs:
            multiplier = 100
            position_sign = 1 if leg.direction == TradeDirection.BUY else -1
            
            if leg.option.greeks:
                total_delta += float(leg.option.greeks.delta or 0) * leg.quantity * multiplier * position_sign
                total_gamma += float(leg.option.greeks.gamma or 0) * leg.quantity * multiplier * position_sign
                total_theta += float(leg.option.greeks.theta or 0) * leg.quantity * multiplier * position_sign
                total_vega += float(leg.option.greeks.vega or 0) * leg.quantity * multiplier * position_sign
        
        # Normalize by NAV factor
        nav_factor = float(nav) / 100000  # Reference NAV of $100k
        
        return {
            'delta': total_delta / nav_factor,
            'gamma': total_gamma / nav_factor,
            'theta': total_theta / nav_factor,
            'vega': total_vega / nav_factor
        }
    
    def _calculate_margin_requirement(self, strategy: StrategyDefinition) -> Decimal:
        """Calculate margin/capital requirement for strategy."""
        
        strategy_type = strategy.strategy_type.value
        
        if strategy_type == "PUT_CREDIT_SPREAD":
            # Margin = Strike width - Net credit
            if strategy.strike_width and strategy.net_credit:
                return (strategy.strike_width * 100) - strategy.net_credit
            elif strategy.max_loss:
                return strategy.max_loss
            
        elif strategy_type == "CALL_CREDIT_SPREAD":
            if strategy.strike_width and strategy.net_credit:
                return (strategy.strike_width * 100) - strategy.net_credit
            elif strategy.max_loss:
                return strategy.max_loss
            
        elif strategy_type == "IRON_CONDOR":
            # Margin = Larger of the two spread widths - Net credit
            if strategy.max_loss:
                return strategy.max_loss
            
        elif strategy_type == "COVERED_CALL":
            return Decimal('0')  # Assuming stock already owned
            
        elif strategy_type == "CASH_SECURED_PUT":
            # Cash secured amount = strike price × 100
            if strategy.legs:
                put_leg = next((leg for leg in strategy.legs if leg.option.option_type == OptionType.PUT), None)
                if put_leg:
                    return put_leg.option.strike * 100
            
        # Conservative default
        return strategy.max_loss if strategy.max_loss else Decimal('500')
    
    def _generate_validation_summary(
        self,
        is_valid: bool,
        violations: List[ConstraintViolation],
        warnings: List[ConstraintViolation]
    ) -> str:
        """Generate human-readable validation summary."""
        
        if is_valid and not warnings:
            return "✅ All constraints passed"
        
        summary_parts = []
        
        if not is_valid:
            critical_count = len([v for v in violations if v.severity == ConstraintSeverity.CRITICAL])
            summary_parts.append(f"❌ {critical_count} critical violation(s)")
        
        if warnings:
            warning_count = len(warnings)
            summary_parts.append(f"⚠️ {warning_count} warning(s)")
        
        if violations:
            constraint_names = [v.constraint_name for v in violations]
            summary_parts.append(f"Failed: {', '.join(constraint_names)}")
        
        return " | ".join(summary_parts)