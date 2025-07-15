"""
Options data validators for ensuring data quality and consistency.
"""

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from ..models.options import OptionQuote, OptionsChain, Greeks, OptionType

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue:
    """Represents a validation issue."""
    
    def __init__(self, severity: ValidationSeverity, message: str, field: Optional[str] = None):
        self.severity = severity
        self.message = message
        self.field = field
        self.timestamp = datetime.utcnow()
    
    def __str__(self):
        field_info = f" ({self.field})" if self.field else ""
        return f"{self.severity.value.upper()}: {self.message}{field_info}"


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.is_valid = True
    
    def add_issue(self, severity: ValidationSeverity, message: str, field: Optional[str] = None):
        """Add a validation issue."""
        issue = ValidationIssue(severity, message, field)
        self.issues.append(issue)
        
        # Mark as invalid if we have any errors
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False
    
    def add_error(self, message: str, field: Optional[str] = None):
        """Add an error issue."""
        self.add_issue(ValidationSeverity.ERROR, message, field)
    
    def add_warning(self, message: str, field: Optional[str] = None):
        """Add a warning issue."""
        self.add_issue(ValidationSeverity.WARNING, message, field)
    
    def add_info(self, message: str, field: Optional[str] = None):
        """Add an info issue."""
        self.add_issue(ValidationSeverity.INFO, message, field)
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get only error issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.get_errors()) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.get_warnings()) > 0
    
    def summary(self) -> str:
        """Get summary of validation results."""
        error_count = len(self.get_errors())
        warning_count = len(self.get_warnings())
        
        if self.is_valid:
            if warning_count > 0:
                return f"Valid with {warning_count} warnings"
            else:
                return "Valid"
        else:
            return f"Invalid: {error_count} errors, {warning_count} warnings"


class GreeksValidator:
    """Validator for options Greeks."""
    
    @staticmethod
    def validate(greeks: Greeks) -> ValidationResult:
        """Validate Greeks values."""
        result = ValidationResult()
        
        if greeks is None:
            result.add_warning("Greeks data is missing")
            return result
        
        # Delta validation
        if greeks.delta is not None:
            if not (-1.0 <= greeks.delta <= 1.0):
                result.add_error(f"Delta {greeks.delta} out of valid range [-1, 1]", "delta")
        
        # Gamma validation
        if greeks.gamma is not None:
            if greeks.gamma < 0:
                result.add_error(f"Gamma {greeks.gamma} cannot be negative", "gamma")
            elif greeks.gamma > 1.0:
                result.add_warning(f"Gamma {greeks.gamma} is unusually high", "gamma")
        
        # Theta validation
        if greeks.theta is not None:
            # Theta is typically negative for long positions
            if abs(greeks.theta) > 1.0:
                result.add_warning(f"Theta {greeks.theta} has unusually high magnitude", "theta")
        
        # Vega validation
        if greeks.vega is not None:
            if greeks.vega < 0:
                result.add_error(f"Vega {greeks.vega} cannot be negative", "vega")
            elif greeks.vega > 1.0:
                result.add_warning(f"Vega {greeks.vega} is unusually high", "vega")
        
        # Rho validation (can be positive or negative)
        if greeks.rho is not None:
            if abs(greeks.rho) > 1.0:
                result.add_warning(f"Rho {greeks.rho} has unusually high magnitude", "rho")
        
        return result


class OptionQuoteValidator:
    """Validator for individual option quotes."""
    
    @staticmethod
    def validate(option: OptionQuote, underlying_price: Optional[Decimal] = None) -> ValidationResult:
        """Validate option quote data."""
        result = ValidationResult()
        
        # Basic field validation
        if not option.symbol:
            result.add_error("Option symbol is required", "symbol")
        
        if not option.underlying:
            result.add_error("Underlying symbol is required", "underlying")
        
        if option.strike <= 0:
            result.add_error(f"Strike price {option.strike} must be positive", "strike")
        
        # Expiration validation
        if option.expiration < date.today():
            result.add_error(f"Option expired on {option.expiration}", "expiration")
        elif option.expiration == date.today():
            result.add_warning("Option expires today", "expiration")
        
        # Price validation
        OptionQuoteValidator._validate_prices(option, result)
        
        # Volume validation
        OptionQuoteValidator._validate_volume_oi(option, result)
        
        # Implied volatility validation
        if option.implied_volatility is not None:
            if option.implied_volatility < 0:
                result.add_error(f"Implied volatility {option.implied_volatility} cannot be negative", "implied_volatility")
            elif option.implied_volatility > 5.0:  # 500%
                result.add_warning(f"Implied volatility {option.implied_volatility:.1%} is extremely high", "implied_volatility")
            elif option.implied_volatility < 0.05:  # 5%
                result.add_warning(f"Implied volatility {option.implied_volatility:.1%} is extremely low", "implied_volatility")
        
        # Greeks validation
        if option.greeks:
            greeks_result = GreeksValidator.validate(option.greeks)
            result.issues.extend(greeks_result.issues)
            if not greeks_result.is_valid:
                result.is_valid = False
        
        # Quote age validation
        if option.quote_time:
            age_minutes = (datetime.utcnow() - option.quote_time).total_seconds() / 60
            if age_minutes > 60:
                result.add_warning(f"Quote is {age_minutes:.1f} minutes old", "quote_time")
            elif age_minutes > 10:
                result.add_info(f"Quote is {age_minutes:.1f} minutes old", "quote_time")
        
        # Intrinsic value validation (if underlying price available)
        if underlying_price:
            OptionQuoteValidator._validate_intrinsic_value(option, underlying_price, result)
        
        return result
    
    @staticmethod
    def _validate_prices(option: OptionQuote, result: ValidationResult):
        """Validate option price data."""
        # Check for negative prices
        for price_field in ['bid', 'ask', 'last', 'mark']:
            price = getattr(option, price_field)
            if price is not None and price < 0:
                result.add_error(f"{price_field} price {price} cannot be negative", price_field)
        
        # Bid/Ask relationship
        if option.bid is not None and option.ask is not None:
            if option.bid > option.ask:
                result.add_error(f"Bid {option.bid} cannot be greater than ask {option.ask}", "bid_ask")
            
            # Check spread width
            spread = option.ask - option.bid
            if option.mark and option.mark > 0:
                spread_pct = spread / option.mark
                if spread_pct > 0.5:  # 50%
                    result.add_warning(f"Wide bid-ask spread: {spread_pct:.1%}", "bid_ask_spread")
        
        # Mark price validation
        if option.mark is not None and option.bid is not None and option.ask is not None:
            if not (option.bid <= option.mark <= option.ask):
                result.add_warning(f"Mark price {option.mark} outside bid-ask range", "mark")
    
    @staticmethod
    def _validate_volume_oi(option: OptionQuote, result: ValidationResult):
        """Validate volume and open interest."""
        if option.volume is not None:
            if option.volume < 0:
                result.add_error(f"Volume {option.volume} cannot be negative", "volume")
            elif option.volume == 0:
                result.add_info("No volume traded today", "volume")
        
        if option.open_interest is not None:
            if option.open_interest < 0:
                result.add_error(f"Open interest {option.open_interest} cannot be negative", "open_interest")
            elif option.open_interest == 0:
                result.add_warning("No open interest", "open_interest")
            elif option.open_interest < 10:
                result.add_warning(f"Low open interest: {option.open_interest}", "open_interest")
    
    @staticmethod
    def _validate_intrinsic_value(option: OptionQuote, underlying_price: Decimal, result: ValidationResult):
        """Validate option pricing relative to intrinsic value."""
        intrinsic_value = option.get_intrinsic_value(underlying_price)
        
        if option.mark is not None:
            if option.mark < intrinsic_value:
                diff = intrinsic_value - option.mark
                result.add_warning(f"Option trading below intrinsic value by {diff}", "intrinsic_value")
            
            time_value = option.get_time_value(underlying_price)
            if time_value is not None and time_value < 0:
                result.add_warning("Negative time value detected", "time_value")


class OptionsChainValidator:
    """Validator for complete options chains."""
    
    @staticmethod
    def validate(chain: OptionsChain) -> ValidationResult:
        """Validate options chain data."""
        result = ValidationResult()
        
        # Basic validation
        if not chain.underlying:
            result.add_error("Options chain missing underlying symbol", "underlying")
            return result
        
        if not chain.options:
            result.add_warning("Options chain is empty", "options")
            return result
        
        # Validate underlying price
        if chain.underlying_price is not None:
            if chain.underlying_price <= 0:
                result.add_error(f"Underlying price {chain.underlying_price} must be positive", "underlying_price")
        
        # Validate chain timestamp
        if chain.chain_timestamp:
            age_minutes = (datetime.utcnow() - chain.chain_timestamp).total_seconds() / 60
            if age_minutes > 30:
                result.add_warning(f"Options chain is {age_minutes:.1f} minutes old", "chain_timestamp")
        
        # Validate individual options
        option_symbols = set()
        total_options = len(chain.options)
        valid_options = 0
        
        for i, option in enumerate(chain.options):
            # Check for duplicate options
            option_key = (option.strike, option.expiration, option.option_type)
            if option_key in option_symbols:
                result.add_warning(f"Duplicate option found: {option_key}", f"options[{i}]")
            else:
                option_symbols.add(option_key)
            
            # Validate underlying consistency
            if option.underlying != chain.underlying:
                result.add_error(f"Option underlying {option.underlying} doesn't match chain {chain.underlying}", f"options[{i}].underlying")
            
            # Validate individual option
            option_result = OptionQuoteValidator.validate(option, chain.underlying_price)
            
            # Add option-specific issues with index
            for issue in option_result.issues:
                field = f"options[{i}].{issue.field}" if issue.field else f"options[{i}]"
                result.add_issue(issue.severity, issue.message, field)
            
            if option_result.is_valid:
                valid_options += 1
        
        # Chain-level statistics
        valid_ratio = valid_options / total_options if total_options > 0 else 0
        if valid_ratio < 0.8:
            result.add_warning(f"Only {valid_ratio:.1%} of options are valid", "options")
        
        # Validate chain structure
        OptionsChainValidator._validate_chain_structure(chain, result)
        
        return result
    
    @staticmethod
    def _validate_chain_structure(chain: OptionsChain, result: ValidationResult):
        """Validate the structure and completeness of the options chain."""
        expirations = chain.get_expiration_dates()
        
        if not expirations:
            result.add_warning("No valid expirations found in chain", "expirations")
            return
        
        # Check for reasonable expiration spread
        nearest_expiry = min(expirations)
        farthest_expiry = max(expirations)
        
        days_to_nearest = (nearest_expiry - date.today()).days
        days_to_farthest = (farthest_expiry - date.today()).days
        
        if days_to_nearest < 0:
            result.add_warning("Chain contains expired options", "expirations")
        
        if days_to_farthest > 365 * 2:  # More than 2 years
            result.add_info("Chain contains long-dated options (>2 years)", "expirations")
        
        # Check strike distribution for each expiration
        for expiration in expirations:
            strikes = chain.get_strikes_for_expiration(expiration)
            
            if len(strikes) < 5:
                result.add_warning(f"Limited strikes available for {expiration}", f"strikes_{expiration}")
            
            # Check for reasonable strike spacing
            if len(strikes) >= 2:
                strike_diffs = [strikes[i] - strikes[i-1] for i in range(1, len(strikes))]
                max_diff = max(strike_diffs)
                min_diff = min(strike_diffs)
                
                if max_diff / min_diff > 5:  # Inconsistent spacing
                    result.add_info(f"Inconsistent strike spacing for {expiration}", f"strikes_{expiration}")


class OptionsValidator:
    """Main validator class for options-related validation."""
    
    @staticmethod
    def validate_option_quote(option: OptionQuote, 
                            underlying_price: Optional[Decimal] = None,
                            strict: bool = False) -> ValidationResult:
        """
        Validate option quote with configurable strictness.
        
        Args:
            option: Option quote to validate
            underlying_price: Current underlying price for intrinsic value checks
            strict: If True, treat warnings as errors
            
        Returns:
            ValidationResult
        """
        result = OptionQuoteValidator.validate(option, underlying_price)
        
        if strict:
            # Convert warnings to errors in strict mode
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    issue.severity = ValidationSeverity.ERROR
                    result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_options_chain(chain: OptionsChain, strict: bool = False) -> ValidationResult:
        """
        Validate options chain with configurable strictness.
        
        Args:
            chain: Options chain to validate
            strict: If True, treat warnings as errors
            
        Returns:
            ValidationResult
        """
        result = OptionsChainValidator.validate(chain)
        
        if strict:
            # Convert warnings to errors in strict mode
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    issue.severity = ValidationSeverity.ERROR
                    result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_greeks(greeks: Greeks) -> ValidationResult:
        """Validate Greeks values."""
        return GreeksValidator.validate(greeks)
    
    @staticmethod
    def validate_liquidity(option: OptionQuote,
                          min_volume: int = 10,
                          min_open_interest: int = 100,
                          max_spread_pct: float = 0.5) -> ValidationResult:
        """
        Validate option liquidity criteria.
        
        Args:
            option: Option quote to validate
            min_volume: Minimum daily volume
            min_open_interest: Minimum open interest
            max_spread_pct: Maximum bid-ask spread percentage
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Volume check
        if option.volume is not None:
            if option.volume < min_volume:
                result.add_warning(f"Volume {option.volume} below minimum {min_volume}", "volume")
        else:
            result.add_warning("Volume data not available", "volume")
        
        # Open interest check
        if option.open_interest is not None:
            if option.open_interest < min_open_interest:
                result.add_warning(f"Open interest {option.open_interest} below minimum {min_open_interest}", "open_interest")
        else:
            result.add_warning("Open interest data not available", "open_interest")
        
        # Spread check
        spread_pct = option.bid_ask_spread_percent
        if spread_pct is not None:
            if spread_pct > max_spread_pct:
                result.add_warning(f"Bid-ask spread {spread_pct:.1%} above maximum {max_spread_pct:.1%}", "bid_ask_spread")
        else:
            result.add_warning("Cannot calculate bid-ask spread", "bid_ask_spread")
        
        return result
    
    @staticmethod
    def validate_quote_freshness(option: OptionQuote, max_age_minutes: float = 10.0) -> ValidationResult:
        """
        Validate quote freshness.
        
        Args:
            option: Option quote to validate
            max_age_minutes: Maximum allowed quote age in minutes
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        if not option.quote_time:
            result.add_error("Quote timestamp is missing", "quote_time")
            return result
        
        age_minutes = (datetime.utcnow() - option.quote_time).total_seconds() / 60
        
        if age_minutes > max_age_minutes:
            result.add_error(f"Quote age {age_minutes:.1f} minutes exceeds maximum {max_age_minutes}", "quote_time")
        elif age_minutes > max_age_minutes * 0.8:
            result.add_warning(f"Quote is getting stale: {age_minutes:.1f} minutes old", "quote_time")
        
        return result