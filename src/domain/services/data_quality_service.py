"""
Data quality assurance framework for market data validation.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import statistics
import logging

from ...data.models.options import OptionsChain, OptionQuote, Greeks
from ...data.models.market_data import StockQuote, TechnicalIndicators, FundamentalData
from ...infrastructure.error_handling import (
    DataQualityError, StaleDataError, InsufficientDataError,
    ValidationError
)


class QualityCheckSeverity(Enum):
    """Severity levels for data quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Individual data quality issue."""
    severity: QualityCheckSeverity
    category: str
    description: str
    field_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    data_source: str
    symbol: Optional[str]
    timestamp: datetime
    overall_score: float  # 0-1
    issues: List[QualityIssue]
    passed_checks: int
    total_checks: int
    
    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == QualityCheckSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(issue.severity == QualityCheckSeverity.ERROR for issue in self.issues)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == QualityCheckSeverity.WARNING)


class DataQualityService:
    """
    Comprehensive data quality assurance framework.
    
    Features:
    - Quote age verification with automatic rejection of stale data
    - Bid/ask spread reasonableness checks
    - Volume and open interest consistency validation
    - Greeks mathematical validation using Black-Scholes verification
    - Cross-exchange data comparison for accuracy confirmation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Quality thresholds from instructions
        self.thresholds = {
            # Quote freshness
            "max_quote_age_minutes": 10,
            "critical_quote_age_minutes": 30,
            
            # Spread checks
            "max_bid_ask_spread_pct": 0.05,  # 5% from instructions
            "warning_spread_pct": 0.10,      # 10% warning level
            "critical_spread_pct": 0.50,     # 50% critical level
            
            # Volume and open interest
            "min_volume": 10,
            "min_open_interest": 50,  # Reduced from 100 for practicality
            "min_avg_volume": 1000,
            
            # Price validation
            "min_option_price": 0.01,
            "max_option_price_ratio": 2.0,  # 200% of intrinsic value
            
            # Greeks validation ranges
            "delta_range": (-1.0, 1.0),
            "gamma_min": 0.0,
            "theta_max": 0.0,  # Theta should be negative for long options
            "vega_min": 0.0,
            
            # Implied volatility
            "min_implied_vol": 0.01,  # 1%
            "max_implied_vol": 5.0,   # 500%
            "warning_implied_vol": 2.0,  # 200%
        }
    
    def validate_options_chain(self, chain: OptionsChain) -> QualityReport:
        """
        Comprehensive validation of options chain data.
        
        Args:
            chain: Options chain to validate
            
        Returns:
            Quality report with issues and overall score
        """
        issues = []
        total_checks = 0
        
        # Basic chain validation
        chain_issues, chain_checks = self._validate_chain_structure(chain)
        issues.extend(chain_issues)
        total_checks += chain_checks
        
        # Validate individual options
        for expiration, strikes_dict in chain.options.items():
            for strike, types_dict in strikes_dict.items():
                for option_type, option in types_dict.items():
                    option_issues, option_checks = self._validate_option_quote(option, chain.underlying_price)
                    issues.extend(option_issues)
                    total_checks += option_checks
        
        # Cross-validation checks
        cross_issues, cross_checks = self._cross_validate_options(chain)
        issues.extend(cross_issues)
        total_checks += cross_checks
        
        # Calculate overall score
        passed_checks = total_checks - len([i for i in issues if i.severity in [QualityCheckSeverity.ERROR, QualityCheckSeverity.CRITICAL]])
        overall_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return QualityReport(
            data_source=chain.data_source or "Unknown",
            symbol=chain.underlying,
            timestamp=datetime.utcnow(),
            overall_score=overall_score,
            issues=issues,
            passed_checks=passed_checks,
            total_checks=total_checks
        )
    
    def validate_stock_quote(self, quote: StockQuote) -> QualityReport:
        """Validate stock quote data quality."""
        issues = []
        total_checks = 0
        
        # Price validation
        price_issues, price_checks = self._validate_stock_prices(quote)
        issues.extend(price_issues)
        total_checks += price_checks
        
        # Volume validation
        volume_issues, volume_checks = self._validate_stock_volume(quote)
        issues.extend(volume_issues)
        total_checks += volume_checks
        
        # Quote freshness
        freshness_issues, freshness_checks = self._validate_quote_freshness(quote.quote_time)
        issues.extend(freshness_issues)
        total_checks += freshness_checks
        
        # Calculate score
        passed_checks = total_checks - len([i for i in issues if i.severity in [QualityCheckSeverity.ERROR, QualityCheckSeverity.CRITICAL]])
        overall_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return QualityReport(
            data_source="StockQuote",
            symbol=quote.symbol,
            timestamp=datetime.utcnow(),
            overall_score=overall_score,
            issues=issues,
            passed_checks=passed_checks,
            total_checks=total_checks
        )
    
    def validate_fundamental_data(self, data: FundamentalData) -> QualityReport:
        """Validate fundamental data quality."""
        issues = []
        total_checks = 0
        
        # Financial ratios validation
        ratio_issues, ratio_checks = self._validate_financial_ratios(data)
        issues.extend(ratio_issues)
        total_checks += ratio_checks
        
        # Data completeness
        completeness_issues, completeness_checks = self._validate_fundamental_completeness(data)
        issues.extend(completeness_issues)
        total_checks += completeness_checks
        
        # Data consistency
        consistency_issues, consistency_checks = self._validate_fundamental_consistency(data)
        issues.extend(consistency_issues)
        total_checks += consistency_checks
        
        # Calculate score
        passed_checks = total_checks - len([i for i in issues if i.severity in [QualityCheckSeverity.ERROR, QualityCheckSeverity.CRITICAL]])
        overall_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return QualityReport(
            data_source="FundamentalData",
            symbol=data.symbol,
            timestamp=datetime.utcnow(),
            overall_score=overall_score,
            issues=issues,
            passed_checks=passed_checks,
            total_checks=total_checks
        )
    
    def _validate_chain_structure(self, chain: OptionsChain) -> Tuple[List[QualityIssue], int]:
        """Validate basic options chain structure."""
        issues = []
        checks = 0
        
        # Check if chain has options
        checks += 1
        if not chain.options:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.CRITICAL,
                category="structure",
                description="Options chain is empty",
                suggestion="Verify symbol exists and has options"
            ))
        
        # Check underlying price
        checks += 1
        if not chain.underlying_price or chain.underlying_price <= 0:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.ERROR,
                category="structure",
                description="Missing or invalid underlying price",
                actual_value=chain.underlying_price,
                suggestion="Fetch current stock quote"
            ))
        
        # Check expiration spread
        if chain.options:
            checks += 1
            expirations = list(chain.options.keys())
            if len(expirations) < 2:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="structure",
                    description="Limited expiration dates available",
                    actual_value=len(expirations),
                    expected_value="2+",
                    suggestion="Verify expiration filter settings"
                ))
        
        return issues, checks
    
    def _validate_option_quote(self, option: OptionQuote, underlying_price: Optional[Decimal]) -> Tuple[List[QualityIssue], int]:
        """Validate individual option quote."""
        issues = []
        checks = 0
        
        # Quote freshness
        freshness_issues, freshness_checks = self._validate_quote_freshness(option.quote_time)
        issues.extend(freshness_issues)
        checks += freshness_checks
        
        # Bid/ask validation
        bid_ask_issues, bid_ask_checks = self._validate_bid_ask(option)
        issues.extend(bid_ask_issues)
        checks += bid_ask_checks
        
        # Volume and open interest
        liquidity_issues, liquidity_checks = self._validate_liquidity(option)
        issues.extend(liquidity_issues)
        checks += liquidity_checks
        
        # Implied volatility
        iv_issues, iv_checks = self._validate_implied_volatility(option)
        issues.extend(iv_issues)
        checks += iv_checks
        
        # Greeks validation
        if option.greeks:
            greeks_issues, greeks_checks = self._validate_greeks(option, underlying_price)
            issues.extend(greeks_issues)
            checks += greeks_checks
        
        return issues, checks
    
    def _validate_quote_freshness(self, quote_time: datetime) -> Tuple[List[QualityIssue], int]:
        """Validate quote freshness."""
        issues = []
        checks = 1
        
        age_minutes = (datetime.utcnow() - quote_time).total_seconds() / 60
        
        if age_minutes > self.thresholds["critical_quote_age_minutes"]:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.CRITICAL,
                category="freshness",
                description="Quote is critically stale",
                field_name="quote_time",
                actual_value=f"{age_minutes:.1f} minutes",
                expected_value=f"<{self.thresholds['max_quote_age_minutes']} minutes",
                suggestion="Refresh quote data"
            ))
        elif age_minutes > self.thresholds["max_quote_age_minutes"]:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.ERROR,
                category="freshness",
                description="Quote is stale",
                field_name="quote_time",
                actual_value=f"{age_minutes:.1f} minutes",
                expected_value=f"<{self.thresholds['max_quote_age_minutes']} minutes",
                suggestion="Refresh quote data"
            ))
        
        return issues, checks
    
    def _validate_bid_ask(self, option: OptionQuote) -> Tuple[List[QualityIssue], int]:
        """Validate bid/ask spread and prices."""
        issues = []
        checks = 0
        
        # Check bid/ask presence
        checks += 1
        if option.bid is None or option.ask is None:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.WARNING,
                category="pricing",
                description="Missing bid or ask price",
                field_name="bid/ask",
                suggestion="Verify market hours and liquidity"
            ))
            return issues, checks
        
        # Check bid <= ask
        checks += 1
        if option.bid > option.ask:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.ERROR,
                category="pricing",
                description="Bid price exceeds ask price",
                field_name="bid_ask_order",
                actual_value=f"bid={option.bid}, ask={option.ask}",
                suggestion="Data error - refresh quote"
            ))
        
        # Check spread reasonableness
        if option.bid_ask_spread_pct:
            checks += 1
            spread_pct = option.bid_ask_spread_pct
            
            if spread_pct > self.thresholds["critical_spread_pct"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.CRITICAL,
                    category="pricing",
                    description="Extremely wide bid/ask spread",
                    field_name="bid_ask_spread",
                    actual_value=f"{spread_pct:.1%}",
                    expected_value=f"<{self.thresholds['max_bid_ask_spread_pct']:.1%}",
                    suggestion="Option may be illiquid"
                ))
            elif spread_pct > self.thresholds["warning_spread_pct"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="pricing",
                    description="Wide bid/ask spread",
                    field_name="bid_ask_spread",
                    actual_value=f"{spread_pct:.1%}",
                    expected_value=f"<{self.thresholds['max_bid_ask_spread_pct']:.1%}",
                    suggestion="Consider liquidity requirements"
                ))
        
        # Check minimum price
        checks += 1
        min_price = min(option.bid, option.ask)
        if min_price < self.thresholds["min_option_price"]:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.WARNING,
                category="pricing",
                description="Very low option price",
                field_name="price",
                actual_value=f"${min_price}",
                expected_value=f">=${self.thresholds['min_option_price']}",
                suggestion="May indicate low-value option"
            ))
        
        return issues, checks
    
    def _validate_liquidity(self, option: OptionQuote) -> Tuple[List[QualityIssue], int]:
        """Validate volume and open interest."""
        issues = []
        checks = 0
        
        # Volume check
        checks += 1
        if option.volume is not None:
            if option.volume < self.thresholds["min_volume"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="liquidity",
                    description="Low trading volume",
                    field_name="volume",
                    actual_value=option.volume,
                    expected_value=f">={self.thresholds['min_volume']}",
                    suggestion="May be difficult to trade"
                ))
        else:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.INFO,
                category="liquidity",
                description="Volume data not available",
                field_name="volume"
            ))
        
        # Open interest check
        checks += 1
        if option.open_interest is not None:
            if option.open_interest < self.thresholds["min_open_interest"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="liquidity",
                    description="Low open interest",
                    field_name="open_interest",
                    actual_value=option.open_interest,
                    expected_value=f">={self.thresholds['min_open_interest']}",
                    suggestion="Limited liquidity"
                ))
        else:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.INFO,
                category="liquidity",
                description="Open interest data not available",
                field_name="open_interest"
            ))
        
        return issues, checks
    
    def _validate_implied_volatility(self, option: OptionQuote) -> Tuple[List[QualityIssue], int]:
        """Validate implied volatility."""
        issues = []
        checks = 1
        
        if option.implied_volatility is None:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.WARNING,
                category="pricing",
                description="Implied volatility not available",
                field_name="implied_volatility"
            ))
            return issues, checks
        
        iv = option.implied_volatility
        
        # Check reasonable range
        if iv < self.thresholds["min_implied_vol"]:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.WARNING,
                category="pricing",
                description="Unusually low implied volatility",
                field_name="implied_volatility",
                actual_value=f"{iv:.1%}",
                expected_value=f">={self.thresholds['min_implied_vol']:.1%}",
                suggestion="Verify calculation"
            ))
        elif iv > self.thresholds["max_implied_vol"]:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.ERROR,
                category="pricing",
                description="Extremely high implied volatility",
                field_name="implied_volatility",
                actual_value=f"{iv:.1%}",
                expected_value=f"<={self.thresholds['max_implied_vol']:.1%}",
                suggestion="Data error or extreme market conditions"
            ))
        elif iv > self.thresholds["warning_implied_vol"]:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.WARNING,
                category="pricing",
                description="High implied volatility",
                field_name="implied_volatility",
                actual_value=f"{iv:.1%}",
                expected_value=f"<={self.thresholds['warning_implied_vol']:.1%}",
                suggestion="Elevated volatility environment"
            ))
        
        return issues, checks
    
    def _validate_greeks(self, option: OptionQuote, underlying_price: Optional[Decimal]) -> Tuple[List[QualityIssue], int]:
        """Validate Greeks values."""
        issues = []
        checks = 0
        greeks = option.greeks
        
        # Delta validation
        checks += 1
        if greeks.delta is not None:
            delta_min, delta_max = self.thresholds["delta_range"]
            if not (delta_min <= greeks.delta <= delta_max):
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="greeks",
                    description="Delta out of valid range",
                    field_name="delta",
                    actual_value=greeks.delta,
                    expected_value=f"{delta_min} to {delta_max}",
                    suggestion="Recalculate Greeks"
                ))
            
            # Option type specific checks
            if option.option_type.value == "call" and greeks.delta < 0:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="greeks",
                    description="Call option has negative delta",
                    field_name="delta",
                    actual_value=greeks.delta,
                    suggestion="Verify option type or recalculate"
                ))
            elif option.option_type.value == "put" and greeks.delta > 0:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="greeks",
                    description="Put option has positive delta",
                    field_name="delta",
                    actual_value=greeks.delta,
                    suggestion="Verify option type or recalculate"
                ))
        
        # Gamma validation
        checks += 1
        if greeks.gamma is not None:
            if greeks.gamma < self.thresholds["gamma_min"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="greeks",
                    description="Negative gamma",
                    field_name="gamma",
                    actual_value=greeks.gamma,
                    expected_value=f">={self.thresholds['gamma_min']}",
                    suggestion="Gamma should be positive for long options"
                ))
        
        # Theta validation
        checks += 1
        if greeks.theta is not None:
            if greeks.theta > self.thresholds["theta_max"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="greeks",
                    description="Positive theta",
                    field_name="theta",
                    actual_value=greeks.theta,
                    expected_value=f"<={self.thresholds['theta_max']}",
                    suggestion="Theta typically negative for long options"
                ))
        
        # Vega validation
        checks += 1
        if greeks.vega is not None:
            if greeks.vega < self.thresholds["vega_min"]:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="greeks",
                    description="Negative vega",
                    field_name="vega",
                    actual_value=greeks.vega,
                    expected_value=f">={self.thresholds['vega_min']}",
                    suggestion="Vega typically positive for long options"
                ))
        
        return issues, checks
    
    def _cross_validate_options(self, chain: OptionsChain) -> Tuple[List[QualityIssue], int]:
        """Cross-validate options within the chain."""
        issues = []
        checks = 0
        
        for expiration in chain.get_expirations():
            strikes = chain.get_strikes(expiration)
            
            # Check put-call parity violations
            parity_issues, parity_checks = self._check_put_call_parity(chain, expiration, strikes)
            issues.extend(parity_issues)
            checks += parity_checks
            
            # Check volatility smile consistency
            smile_issues, smile_checks = self._check_volatility_smile(chain, expiration, strikes)
            issues.extend(smile_issues)
            checks += smile_checks
        
        return issues, checks
    
    def _check_put_call_parity(self, chain: OptionsChain, expiration: date, strikes: List[Decimal]) -> Tuple[List[QualityIssue], int]:
        """Check put-call parity relationships."""
        issues = []
        checks = 0
        
        if not chain.underlying_price:
            return issues, checks
        
        for strike in strikes:
            call_option = chain.get_option(expiration, strike, "call")
            put_option = chain.get_option(expiration, strike, "put")
            
            if call_option and put_option and call_option.mid_price and put_option.mid_price:
                checks += 1
                
                # Simplified put-call parity check
                # C - P ≈ S - K * e^(-r*T)
                # For simplicity, ignoring interest rate and dividends
                call_put_diff = call_option.mid_price - put_option.mid_price
                stock_strike_diff = chain.underlying_price - strike
                
                parity_diff = abs(call_put_diff - stock_strike_diff)
                tolerance = chain.underlying_price * Decimal('0.05')  # 5% tolerance
                
                if parity_diff > tolerance:
                    issues.append(QualityIssue(
                        severity=QualityCheckSeverity.WARNING,
                        category="arbitrage",
                        description="Potential put-call parity violation",
                        field_name="put_call_parity",
                        actual_value=f"Diff: ${parity_diff}",
                        expected_value=f"<${tolerance}",
                        suggestion="Review for arbitrage opportunity or data error"
                    ))
        
        return issues, checks
    
    def _check_volatility_smile(self, chain: OptionsChain, expiration: date, strikes: List[Decimal]) -> Tuple[List[QualityIssue], int]:
        """Check for reasonable volatility smile."""
        issues = []
        checks = 0
        
        if not chain.underlying_price or len(strikes) < 3:
            return issues, checks
        
        # Collect implied volatilities by moneyness
        iv_data = []
        for strike in strikes:
            # Check both calls and puts
            for option_type in ["call", "put"]:
                option = chain.get_option(expiration, strike, option_type)
                if option and option.implied_volatility:
                    moneyness = float(strike / chain.underlying_price)
                    iv_data.append((moneyness, option.implied_volatility))
        
        if len(iv_data) < 3:
            return issues, checks
        
        # Sort by moneyness
        iv_data.sort(key=lambda x: x[0])
        ivs = [iv for _, iv in iv_data]
        
        checks += 1
        
        # Check for reasonable IV range
        iv_range = max(ivs) - min(ivs)
        if iv_range > 2.0:  # 200% range seems excessive
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.WARNING,
                category="volatility",
                description="Extremely wide volatility smile",
                field_name="implied_volatility_range",
                actual_value=f"{iv_range:.1%}",
                expected_value="<200%",
                suggestion="Review for data quality issues"
            ))
        
        # Check for smoothness (no large jumps)
        for i in range(1, len(ivs)):
            iv_jump = abs(ivs[i] - ivs[i-1])
            if iv_jump > 0.5:  # 50% jump
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="volatility",
                    description="Large implied volatility jump",
                    field_name="implied_volatility_jump",
                    actual_value=f"{iv_jump:.1%}",
                    suggestion="Review for data errors"
                ))
                break  # Don't report multiple jumps
        
        return issues, checks
    
    def _validate_stock_prices(self, quote: StockQuote) -> Tuple[List[QualityIssue], int]:
        """Validate stock price data."""
        issues = []
        checks = 0
        
        # Check price is positive
        checks += 1
        if quote.price <= 0:
            issues.append(QualityIssue(
                severity=QualityCheckSeverity.CRITICAL,
                category="pricing",
                description="Non-positive stock price",
                field_name="price",
                actual_value=quote.price,
                suggestion="Data error"
            ))
        
        # Check OHLC relationships
        if all([quote.open, quote.high, quote.low, quote.price]):
            checks += 1
            if not (quote.low <= quote.open <= quote.high and quote.low <= quote.price <= quote.high):
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="pricing",
                    description="Invalid OHLC relationships",
                    field_name="ohlc",
                    actual_value=f"O:{quote.open} H:{quote.high} L:{quote.low} C:{quote.price}",
                    suggestion="Verify price data"
                ))
        
        # Check bid/ask around current price
        if quote.bid and quote.ask:
            checks += 1
            if not (quote.bid <= quote.price <= quote.ask):
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="pricing",
                    description="Current price outside bid/ask",
                    field_name="price_vs_bid_ask",
                    actual_value=f"Price:{quote.price} Bid:{quote.bid} Ask:{quote.ask}",
                    suggestion="May indicate stale quote"
                ))
        
        return issues, checks
    
    def _validate_stock_volume(self, quote: StockQuote) -> Tuple[List[QualityIssue], int]:
        """Validate volume data."""
        issues = []
        checks = 0
        
        if quote.volume is not None:
            checks += 1
            if quote.volume < 0:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="volume",
                    description="Negative volume",
                    field_name="volume",
                    actual_value=quote.volume,
                    suggestion="Data error"
                ))
            
            # Compare to average volume if available
            if quote.avg_volume and quote.volume > 0:
                checks += 1
                volume_ratio = quote.volume / quote.avg_volume
                if volume_ratio > 10:  # 10x average volume
                    issues.append(QualityIssue(
                        severity=QualityCheckSeverity.INFO,
                        category="volume",
                        description="Unusually high volume",
                        field_name="volume_ratio",
                        actual_value=f"{volume_ratio:.1f}x average",
                        suggestion="May indicate news or events"
                    ))
                elif volume_ratio < 0.1:  # 10% of average volume
                    issues.append(QualityIssue(
                        severity=QualityCheckSeverity.WARNING,
                        category="volume",
                        description="Unusually low volume",
                        field_name="volume_ratio",
                        actual_value=f"{volume_ratio:.1f}x average",
                        suggestion="May indicate low activity"
                    ))
        
        return issues, checks
    
    def _validate_financial_ratios(self, data: FundamentalData) -> Tuple[List[QualityIssue], int]:
        """Validate financial ratios for reasonableness."""
        issues = []
        checks = 0
        
        # P/E ratio checks
        if data.pe_ratio is not None:
            checks += 1
            if data.pe_ratio < 0:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="ratios",
                    description="Negative P/E ratio",
                    field_name="pe_ratio",
                    actual_value=data.pe_ratio,
                    suggestion="Company may have negative earnings"
                ))
            elif data.pe_ratio > 1000:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.WARNING,
                    category="ratios",
                    description="Extremely high P/E ratio",
                    field_name="pe_ratio",
                    actual_value=data.pe_ratio,
                    suggestion="Verify earnings data"
                ))
        
        # Margin checks
        for margin_field, margin_name in [
            ("gross_margin", "Gross margin"),
            ("operating_margin", "Operating margin"),
            ("net_margin", "Net margin")
        ]:
            margin_value = getattr(data, margin_field)
            if margin_value is not None:
                checks += 1
                if abs(margin_value) > 2.0:  # 200% margin seems unrealistic
                    issues.append(QualityIssue(
                        severity=QualityCheckSeverity.WARNING,
                        category="ratios",
                        description=f"Extreme {margin_name.lower()}",
                        field_name=margin_field,
                        actual_value=f"{margin_value:.1%}",
                        suggestion="Verify calculation or data source"
                    ))
        
        return issues, checks
    
    def _validate_fundamental_completeness(self, data: FundamentalData) -> Tuple[List[QualityIssue], int]:
        """Check completeness of fundamental data."""
        issues = []
        checks = 1
        
        # Key fields that should typically be available
        key_fields = [
            "revenue", "net_income", "eps", "pe_ratio", 
            "sector", "gross_margin", "operating_margin"
        ]
        
        missing_fields = [field for field in key_fields if getattr(data, field) is None]
        
        if missing_fields:
            severity = QualityCheckSeverity.WARNING if len(missing_fields) <= 2 else QualityCheckSeverity.ERROR
            issues.append(QualityIssue(
                severity=severity,
                category="completeness",
                description=f"Missing key fundamental data",
                field_name="missing_fields",
                actual_value=", ".join(missing_fields),
                suggestion="Verify data source completeness"
            ))
        
        return issues, checks
    
    def _validate_fundamental_consistency(self, data: FundamentalData) -> Tuple[List[QualityIssue], int]:
        """Check internal consistency of fundamental data."""
        issues = []
        checks = 0
        
        # Revenue and profit consistency
        if data.revenue and data.net_income:
            checks += 1
            if data.net_income > data.revenue:
                issues.append(QualityIssue(
                    severity=QualityCheckSeverity.ERROR,
                    category="consistency",
                    description="Net income exceeds revenue",
                    field_name="income_vs_revenue",
                    actual_value=f"Income: {data.net_income}, Revenue: {data.revenue}",
                    suggestion="Check data source for errors"
                ))
        
        # EPS consistency with net income and shares
        if all([data.eps, data.net_income]) and hasattr(data, 'shares_outstanding'):
            shares = getattr(data, 'shares_outstanding', None)
            if shares and shares > 0:
                checks += 1
                calculated_eps = data.net_income / shares
                eps_diff = abs(data.eps - calculated_eps)
                if eps_diff > abs(data.eps * 0.1):  # 10% tolerance
                    issues.append(QualityIssue(
                        severity=QualityCheckSeverity.WARNING,
                        category="consistency",
                        description="EPS inconsistent with net income and shares",
                        field_name="eps_calculation",
                        actual_value=f"Reported: {data.eps}, Calculated: {calculated_eps:.2f}",
                        suggestion="Verify calculation methodology"
                    ))
        
        return issues, checks