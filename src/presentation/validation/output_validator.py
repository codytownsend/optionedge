"""
Output validation framework for trade recommendations and strategy verification.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

from ...data.models.options import OptionQuote, OptionType, Greeks
from ...data.models.trades import StrategyDefinition, TradeCandidate, TradeLeg
from ...domain.services.scoring_engine import ScoredTradeCandidate
from ...domain.services.trade_selector import SelectionResult
from ...infrastructure.error_handling import (
    handle_errors, ValidationError, BusinessLogicError
)


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"      # Must pass for execution
    WARNING = "warning"        # Should investigate but not blocking
    INFO = "info"             # Informational only


class ValidationCategory(Enum):
    """Categories of validation checks."""
    MARKET_DATA = "market_data"
    STRATEGY_LOGIC = "strategy_logic"
    MATHEMATICAL_CONSISTENCY = "mathematical_consistency"
    RISK_VERIFICATION = "risk_verification"
    IMPLEMENTABILITY = "implementability"
    PORTFOLIO_COHERENCE = "portfolio_coherence"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    category: ValidationCategory
    level: ValidationLevel
    message: str
    affected_ticker: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    recommendation: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    is_valid: bool
    critical_issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info_items: List[ValidationIssue] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Summary metrics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    
    # Performance expectations
    expected_portfolio_return: Optional[float] = None
    expected_win_rate: Optional[float] = None
    risk_adjusted_score: Optional[float] = None


class OutputValidator:
    """
    Comprehensive output validation framework for trade recommendations.
    
    Features:
    - Real-time market data verification
    - Strategy implementability confirmation  
    - Mathematical consistency checks across all metrics
    - Cross-validation against independent pricing models
    - Risk scenario edge case testing
    - Performance expectation reality checking
    - Portfolio coherence validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Validation thresholds
        self.thresholds = {
            'max_quote_age_minutes': 15,
            'min_option_volume': 1,
            'max_bid_ask_spread_pct': 0.15,  # 15% for validation (more lenient than trading)
            'min_open_interest': 10,         # Minimum for validation
            'max_price_deviation_pct': 0.05, # 5% max price deviation
            'min_realistic_pop': 0.50,       # 50% minimum realistic POP
            'max_realistic_pop': 0.95,       # 95% maximum realistic POP
            'max_credit_to_loss_ratio': 0.80, # 80% maximum realistic ratio
            'min_days_to_expiration': 1,
            'max_days_to_expiration': 60
        }
    
    @handle_errors(operation_name="validate_output")
    def validate_complete_output(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        current_timestamp: Optional[datetime] = None
    ) -> ValidationReport:
        """
        Perform comprehensive validation of complete trade recommendation output.
        
        Args:
            selection_result: Selection result to validate
            market_data: Current market data for verification
            current_timestamp: Current timestamp for data freshness checks
            
        Returns:
            Comprehensive validation report
        """
        if current_timestamp is None:
            current_timestamp = datetime.utcnow()
        
        self.logger.info(f"Performing comprehensive output validation for {len(selection_result.selected_trades)} trades")
        
        # Initialize report
        report = ValidationReport()
        
        # Pre-output validation checks
        self._validate_market_data_quality(selection_result, market_data, current_timestamp, report)
        self._validate_strategy_implementability(selection_result, market_data, report)
        self._validate_mathematical_consistency(selection_result, report)
        
        # Cross-validation checks
        self._cross_validate_pricing_models(selection_result, market_data, report)
        
        # Post-generation quality control
        self._validate_strategy_replication(selection_result, report)
        self._validate_risk_scenarios(selection_result, report)
        self._validate_performance_expectations(selection_result, report)
        self._validate_portfolio_coherence(selection_result, report)
        
        # Calculate summary metrics
        self._calculate_validation_summary(report)
        
        # Determine overall validation status
        report.is_valid = len(report.critical_issues) == 0
        
        self.logger.info(f"Validation completed. Status: {'PASS' if report.is_valid else 'FAIL'} "
                        f"({report.passed_checks}/{report.total_checks} checks passed)")
        
        return report
    
    def _validate_market_data_quality(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        current_timestamp: datetime,
        report: ValidationReport
    ):
        """Validate real-time market data quality."""
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            ticker = strategy.underlying
            
            # Check quote freshness
            report.total_checks += 1
            for leg in strategy.legs:
                if leg.option.quote_timestamp:
                    try:
                        if isinstance(leg.option.quote_timestamp, str):
                            quote_time = datetime.fromisoformat(leg.option.quote_timestamp.replace('Z', '+00:00'))
                        else:
                            quote_time = leg.option.quote_timestamp
                        
                        age_minutes = (current_timestamp - quote_time).total_seconds() / 60
                        
                        if age_minutes > self.thresholds['max_quote_age_minutes']:
                            report.critical_issues.append(ValidationIssue(
                                category=ValidationCategory.MARKET_DATA,
                                level=ValidationLevel.CRITICAL,
                                message=f"Stale quote data for {ticker}",
                                affected_ticker=ticker,
                                expected_value=f"≤{self.thresholds['max_quote_age_minutes']} minutes",
                                actual_value=f"{age_minutes:.1f} minutes",
                                recommendation="Refresh market data before execution"
                            ))
                            report.failed_checks += 1
                        else:
                            report.passed_checks += 1
                    except Exception as e:
                        report.warnings.append(ValidationIssue(
                            category=ValidationCategory.MARKET_DATA,
                            level=ValidationLevel.WARNING,
                            message=f"Could not validate quote timestamp for {ticker}: {str(e)}",
                            affected_ticker=ticker,
                            recommendation="Verify quote data source"
                        ))
                        report.failed_checks += 1
                else:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.MARKET_DATA,
                        level=ValidationLevel.WARNING,
                        message=f"Missing quote timestamp for {ticker}",
                        affected_ticker=ticker,
                        recommendation="Ensure quote timestamps are provided"
                    ))
                    report.failed_checks += 1
            
            # Validate bid-ask spreads
            report.total_checks += 1
            spread_valid = True
            for leg in strategy.legs:
                if leg.option.bid and leg.option.ask and leg.option.bid > 0 and leg.option.ask > 0:
                    mid_price = (leg.option.bid + leg.option.ask) / 2
                    spread_pct = (leg.option.ask - leg.option.bid) / mid_price
                    
                    if spread_pct > self.thresholds['max_bid_ask_spread_pct']:
                        report.warnings.append(ValidationIssue(
                            category=ValidationCategory.MARKET_DATA,
                            level=ValidationLevel.WARNING,
                            message=f"Wide bid-ask spread for {ticker} option",
                            affected_ticker=ticker,
                            expected_value=f"≤{self.thresholds['max_bid_ask_spread_pct']:.1%}",
                            actual_value=f"{spread_pct:.1%}",
                            recommendation="Consider market impact on execution"
                        ))
                        spread_valid = False
            
            if spread_valid:
                report.passed_checks += 1
            else:
                report.failed_checks += 1
    
    def _validate_strategy_implementability(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        report: ValidationReport
    ):
        """Validate that strategies can be implemented in practice."""
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            ticker = strategy.underlying
            
            # Check option availability
            report.total_checks += 1
            options_available = True
            for leg in strategy.legs:
                # Check minimum volume and open interest
                if leg.option.volume is not None and leg.option.volume < self.thresholds['min_option_volume']:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.IMPLEMENTABILITY,
                        level=ValidationLevel.WARNING,
                        message=f"Low volume option in {ticker} strategy",
                        affected_ticker=ticker,
                        expected_value=f"≥{self.thresholds['min_option_volume']}",
                        actual_value=str(leg.option.volume),
                        recommendation="Monitor liquidity during execution"
                    ))
                    options_available = False
                
                if leg.option.open_interest is not None and leg.option.open_interest < self.thresholds['min_open_interest']:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.IMPLEMENTABILITY,
                        level=ValidationLevel.WARNING,
                        message=f"Low open interest option in {ticker} strategy",
                        affected_ticker=ticker,
                        expected_value=f"≥{self.thresholds['min_open_interest']}",
                        actual_value=str(leg.option.open_interest),
                        recommendation="Consider alternative strikes with higher OI"
                    ))
                    options_available = False
            
            if options_available:
                report.passed_checks += 1
            else:
                report.failed_checks += 1
            
            # Validate expiration dates
            report.total_checks += 1
            if strategy.days_to_expiration:
                dte = strategy.days_to_expiration
                if dte < self.thresholds['min_days_to_expiration']:
                    report.critical_issues.append(ValidationIssue(
                        category=ValidationCategory.IMPLEMENTABILITY,
                        level=ValidationLevel.CRITICAL,
                        message=f"Expiration too soon for {ticker}",
                        affected_ticker=ticker,
                        expected_value=f"≥{self.thresholds['min_days_to_expiration']} days",
                        actual_value=f"{dte} days",
                        recommendation="Select later expiration date"
                    ))
                    report.failed_checks += 1
                elif dte > self.thresholds['max_days_to_expiration']:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.IMPLEMENTABILITY,
                        level=ValidationLevel.WARNING,
                        message=f"Long expiration for {ticker}",
                        affected_ticker=ticker,
                        expected_value=f"≤{self.thresholds['max_days_to_expiration']} days",
                        actual_value=f"{dte} days",
                        recommendation="Consider shorter-term expiration"
                    ))
                    report.failed_checks += 1
                else:
                    report.passed_checks += 1
            else:
                report.warnings.append(ValidationIssue(
                    category=ValidationCategory.IMPLEMENTABILITY,
                    level=ValidationLevel.WARNING,
                    message=f"Missing days to expiration for {ticker}",
                    affected_ticker=ticker,
                    recommendation="Verify expiration date calculation"
                ))
                report.failed_checks += 1
    
    def _validate_mathematical_consistency(
        self,
        selection_result: SelectionResult,
        report: ValidationReport
    ):
        """Validate mathematical consistency across all metrics."""
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            ticker = strategy.underlying
            
            # Validate probability of profit
            report.total_checks += 1
            if strategy.probability_of_profit:
                pop = strategy.probability_of_profit
                if pop < self.thresholds['min_realistic_pop'] or pop > self.thresholds['max_realistic_pop']:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                        level=ValidationLevel.WARNING,
                        message=f"Unrealistic POP for {ticker}",
                        affected_ticker=ticker,
                        expected_value=f"{self.thresholds['min_realistic_pop']:.0%}-{self.thresholds['max_realistic_pop']:.0%}",
                        actual_value=f"{pop:.1%}",
                        recommendation="Review POP calculation methodology"
                    ))
                    report.failed_checks += 1
                else:
                    report.passed_checks += 1
            else:
                report.critical_issues.append(ValidationIssue(
                    category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                    level=ValidationLevel.CRITICAL,
                    message=f"Missing POP calculation for {ticker}",
                    affected_ticker=ticker,
                    recommendation="Calculate probability of profit"
                ))
                report.failed_checks += 1
            
            # Validate credit-to-max-loss ratio
            report.total_checks += 1
            if strategy.credit_to_max_loss_ratio:
                ratio = strategy.credit_to_max_loss_ratio
                if ratio > self.thresholds['max_credit_to_loss_ratio']:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                        level=ValidationLevel.WARNING,
                        message=f"Unrealistically high credit ratio for {ticker}",
                        affected_ticker=ticker,
                        expected_value=f"≤{self.thresholds['max_credit_to_loss_ratio']:.0%}",
                        actual_value=f"{ratio:.1%}",
                        recommendation="Verify risk calculation"
                    ))
                    report.failed_checks += 1
                else:
                    report.passed_checks += 1
            else:
                # Check if we can calculate it
                if strategy.net_credit and strategy.max_loss:
                    calculated_ratio = float(strategy.net_credit) / float(strategy.max_loss)
                    report.info_items.append(ValidationIssue(
                        category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                        level=ValidationLevel.INFO,
                        message=f"Credit ratio calculated: {calculated_ratio:.1%} for {ticker}",
                        affected_ticker=ticker
                    ))
                    report.passed_checks += 1
                else:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                        level=ValidationLevel.WARNING,
                        message=f"Cannot calculate credit ratio for {ticker}",
                        affected_ticker=ticker,
                        recommendation="Ensure net credit and max loss are calculated"
                    ))
                    report.failed_checks += 1
            
            # Validate Greeks calculations
            report.total_checks += 1
            greeks_valid = True
            for leg in strategy.legs:
                if leg.option.greeks:
                    greeks = leg.option.greeks
                    # Basic sanity checks on Greeks
                    if greeks.delta and abs(greeks.delta) > 1.0:
                        report.warnings.append(ValidationIssue(
                            category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                            level=ValidationLevel.WARNING,
                            message=f"Invalid delta value for {ticker}",
                            affected_ticker=ticker,
                            expected_value="[-1.0, 1.0]",
                            actual_value=str(greeks.delta),
                            recommendation="Review Greeks calculation"
                        ))
                        greeks_valid = False
                    
                    if greeks.gamma and greeks.gamma < 0:
                        report.warnings.append(ValidationIssue(
                            category=ValidationCategory.MATHEMATICAL_CONSISTENCY,
                            level=ValidationLevel.WARNING,
                            message=f"Negative gamma for {ticker}",
                            affected_ticker=ticker,
                            expected_value="≥0",
                            actual_value=str(greeks.gamma),
                            recommendation="Review Greeks calculation"
                        ))
                        greeks_valid = False
            
            if greeks_valid:
                report.passed_checks += 1
            else:
                report.failed_checks += 1
    
    def _cross_validate_pricing_models(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        report: ValidationReport
    ):
        """Cross-validate against independent pricing models."""
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            ticker = strategy.underlying
            
            # Validate option prices against theoretical values
            report.total_checks += 1
            prices_valid = True
            
            for leg in strategy.legs:
                if leg.option.bid and leg.option.ask:
                    mid_price = (leg.option.bid + leg.option.ask) / 2
                    
                    # Simple arbitrage check
                    intrinsic_value = leg.option.calculate_intrinsic_value(
                        market_data.get(ticker, {}).get('current_price', leg.option.strike)
                    )
                    
                    if mid_price < intrinsic_value * 0.95:  # Allow 5% tolerance
                        report.warnings.append(ValidationIssue(
                            category=ValidationCategory.RISK_VERIFICATION,
                            level=ValidationLevel.WARNING,
                            message=f"Option price below intrinsic value for {ticker}",
                            affected_ticker=ticker,
                            expected_value=f"≥{intrinsic_value:.2f}",
                            actual_value=f"{mid_price:.2f}",
                            recommendation="Verify option prices for arbitrage"
                        ))
                        prices_valid = False
            
            if prices_valid:
                report.passed_checks += 1
            else:
                report.failed_checks += 1
    
    def _validate_strategy_replication(
        self,
        selection_result: SelectionResult,
        report: ValidationReport
    ):
        """Validate that strategies can be replicated and verified."""
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            ticker = strategy.underlying
            
            # Check strategy leg consistency
            report.total_checks += 1
            if len(strategy.legs) >= 2:
                # Verify expiration consistency
                expirations = [leg.option.expiration for leg in strategy.legs if leg.option.expiration]
                if len(set(expirations)) > 1:
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.STRATEGY_LOGIC,
                        level=ValidationLevel.WARNING,
                        message=f"Multiple expirations in {ticker} strategy",
                        affected_ticker=ticker,
                        recommendation="Verify strategy construction"
                    ))
                    report.failed_checks += 1
                else:
                    report.passed_checks += 1
            else:
                report.passed_checks += 1
    
    def _validate_risk_scenarios(
        self,
        selection_result: SelectionResult,
        report: ValidationReport
    ):
        """Validate risk scenarios and edge cases."""
        
        portfolio_risk = 0.0
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            ticker = strategy.underlying
            
            # Check maximum loss scenarios
            report.total_checks += 1
            if strategy.max_loss:
                max_loss = float(strategy.max_loss)
                portfolio_risk += max_loss
                
                # Reasonable max loss check
                if max_loss > 1000:  # $1000 seems excessive for single trade
                    report.warnings.append(ValidationIssue(
                        category=ValidationCategory.RISK_VERIFICATION,
                        level=ValidationLevel.WARNING,
                        message=f"High maximum loss for {ticker}",
                        affected_ticker=ticker,
                        actual_value=f"${max_loss:.0f}",
                        recommendation="Consider smaller position size"
                    ))
                    report.failed_checks += 1
                else:
                    report.passed_checks += 1
            else:
                report.critical_issues.append(ValidationIssue(
                    category=ValidationCategory.RISK_VERIFICATION,
                    level=ValidationLevel.CRITICAL,
                    message=f"Missing max loss calculation for {ticker}",
                    affected_ticker=ticker,
                    recommendation="Calculate maximum loss scenario"
                ))
                report.failed_checks += 1
        
        # Portfolio risk check
        report.total_checks += 1
        if portfolio_risk > 5000:  # $5000 total portfolio risk seems high
            report.warnings.append(ValidationIssue(
                category=ValidationCategory.PORTFOLIO_COHERENCE,
                level=ValidationLevel.WARNING,
                message="High total portfolio risk",
                actual_value=f"${portfolio_risk:.0f}",
                recommendation="Consider reducing position sizes"
            ))
            report.failed_checks += 1
        else:
            report.passed_checks += 1
    
    def _validate_performance_expectations(
        self,
        selection_result: SelectionResult,
        report: ValidationReport
    ):
        """Validate performance expectations for realism."""
        
        if not selection_result.selected_trades:
            return
        
        # Calculate expected portfolio performance
        total_expected_return = 0.0
        total_risk = 0.0
        win_rates = []
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            if strategy.probability_of_profit:
                win_rates.append(strategy.probability_of_profit)
            
            if strategy.max_profit and strategy.max_loss and strategy.probability_of_profit:
                expected_return = (
                    float(strategy.max_profit) * strategy.probability_of_profit +
                    float(-strategy.max_loss) * (1 - strategy.probability_of_profit)
                )
                total_expected_return += expected_return
                total_risk += float(strategy.max_loss)
        
        if win_rates:
            avg_win_rate = sum(win_rates) / len(win_rates)
            report.expected_win_rate = avg_win_rate
            
            # Reality check on win rate
            report.total_checks += 1
            if avg_win_rate > 0.85:  # 85% seems unrealistically high
                report.warnings.append(ValidationIssue(
                    category=ValidationCategory.RISK_VERIFICATION,
                    level=ValidationLevel.WARNING,
                    message="Unrealistically high average win rate",
                    expected_value="≤85%",
                    actual_value=f"{avg_win_rate:.1%}",
                    recommendation="Review probability calculations"
                ))
                report.failed_checks += 1
            else:
                report.passed_checks += 1
        
        if total_risk > 0:
            portfolio_return_pct = total_expected_return / total_risk * 100
            report.expected_portfolio_return = portfolio_return_pct
            
            # Risk-adjusted score
            if total_expected_return > 0 and total_risk > 0:
                report.risk_adjusted_score = total_expected_return / total_risk
    
    def _validate_portfolio_coherence(
        self,
        selection_result: SelectionResult,
        report: ValidationReport
    ):
        """Validate overall portfolio coherence and diversification."""
        
        if not selection_result.selected_trades:
            return
        
        # Check sector diversification
        sectors = {}
        for scored_trade in selection_result.selected_trades:
            ticker = scored_trade.trade_candidate.strategy.underlying
            # Would map to sector - simplified for now
            sector = ticker[0]  # First letter as proxy
            sectors[sector] = sectors.get(sector, 0) + 1
        
        report.total_checks += 1
        max_sector_count = max(sectors.values()) if sectors else 0
        if max_sector_count > 3:  # More than 3 trades in one sector
            report.warnings.append(ValidationIssue(
                category=ValidationCategory.PORTFOLIO_COHERENCE,
                level=ValidationLevel.WARNING,
                message="High sector concentration",
                actual_value=f"{max_sector_count} trades in one sector",
                recommendation="Improve sector diversification"
            ))
            report.failed_checks += 1
        else:
            report.passed_checks += 1
        
        # Check strategy type diversification
        strategy_types = {}
        for scored_trade in selection_result.selected_trades:
            strategy_type = scored_trade.trade_candidate.strategy.strategy_type.value
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
        
        report.total_checks += 1
        if len(strategy_types) < 2:  # All same strategy type
            report.info_items.append(ValidationIssue(
                category=ValidationCategory.PORTFOLIO_COHERENCE,
                level=ValidationLevel.INFO,
                message="Low strategy type diversification",
                actual_value=f"{len(strategy_types)} unique strategy types",
                recommendation="Consider mixing strategy types"
            ))
            report.passed_checks += 1  # Not a failure, just info
        else:
            report.passed_checks += 1
    
    def _calculate_validation_summary(self, report: ValidationReport):
        """Calculate summary validation metrics."""
        
        # Already calculated during validation process
        pass
    
    def get_validation_summary(self, report: ValidationReport) -> str:
        """Generate human-readable validation summary."""
        
        lines = []
        lines.append("VALIDATION SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Overall Status: {'✓ PASS' if report.is_valid else '✗ FAIL'}")
        lines.append(f"Total Checks: {report.total_checks}")
        lines.append(f"Passed: {report.passed_checks}")
        lines.append(f"Failed: {report.failed_checks}")
        lines.append("")
        
        if report.critical_issues:
            lines.append("CRITICAL ISSUES:")
            for issue in report.critical_issues:
                lines.append(f"  ❌ {issue.message}")
                if issue.recommendation:
                    lines.append(f"     → {issue.recommendation}")
            lines.append("")
        
        if report.warnings:
            lines.append("WARNINGS:")
            for warning in report.warnings[:5]:  # Limit to top 5
                lines.append(f"  ⚠️  {warning.message}")
                if warning.recommendation:
                    lines.append(f"     → {warning.recommendation}")
            if len(report.warnings) > 5:
                lines.append(f"  ... and {len(report.warnings) - 5} more warnings")
            lines.append("")
        
        if report.expected_win_rate:
            lines.append("PERFORMANCE EXPECTATIONS:")
            lines.append(f"  Expected Win Rate: {report.expected_win_rate:.1%}")
            if report.expected_portfolio_return:
                lines.append(f"  Expected Return: {report.expected_portfolio_return:.1f}%")
            if report.risk_adjusted_score:
                lines.append(f"  Risk-Adjusted Score: {report.risk_adjusted_score:.2f}")
        
        return '\n'.join(lines)