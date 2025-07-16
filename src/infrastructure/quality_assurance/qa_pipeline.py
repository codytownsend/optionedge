"""
Quality Assurance Pipeline for trade recommendations.
Implements comprehensive validation checks for all recommendations.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from contextlib import contextmanager

from ...data.models.trades import TradeCandidate, StrategyDefinition
from ...domain.services.scoring_engine import ScoredTradeCandidate
from ...domain.services.trade_selector import SelectionResult
from ...presentation.validation.output_validator import OutputValidator, ValidationReport
from ...infrastructure.error_handling import handle_errors, QualityAssuranceError


class QACheckLevel(Enum):
    """Quality assurance check levels."""
    CRITICAL = "critical"      # Must pass for production
    HIGH = "high"             # Should pass for quality
    MEDIUM = "medium"         # Important but not blocking
    LOW = "low"               # Nice to have


class QACheckCategory(Enum):
    """Categories of QA checks."""
    STRATEGY_VALIDATION = "strategy_validation"
    MARGIN_CALCULATIONS = "margin_calculations"
    COMMISSION_IMPACT = "commission_impact"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    EXECUTION_TIMING = "execution_timing"
    MONITORING_ALERTS = "monitoring_alerts"
    ACCURACY_AUDIT = "accuracy_audit"


@dataclass
class QACheckResult:
    """Result of a single QA check."""
    check_name: str
    category: QACheckCategory
    level: QACheckLevel
    passed: bool
    message: str
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QAPipelineResult:
    """Complete QA pipeline result."""
    overall_pass: bool
    execution_time: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    check_results: List[QACheckResult] = field(default_factory=list)
    critical_failures: List[QACheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class QualityAssurancePipeline:
    """
    Comprehensive quality assurance pipeline for trade recommendations.
    
    Features:
    - Strategy leg availability across exchanges
    - Margin requirement calculations
    - Commission impact analysis
    - Regulatory compliance for retail trading
    - Output consistency verification
    - Accuracy audit trail for all recommendations
    - Execution timing recommendations
    - Trade expiration and monitoring alerts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.output_validator = OutputValidator()
        
        # QA thresholds
        self.thresholds = {
            'max_execution_time': 300.0,  # 5 minutes
            'min_margin_accuracy': 0.95,   # 95% accuracy
            'max_commission_impact': 0.05, # 5% of trade value
            'max_bid_ask_impact': 0.10,    # 10% slippage
            'min_liquidity_score': 70,     # 70/100 minimum
            'max_strategy_complexity': 4,  # Max 4 legs
        }
        
        # Register QA checks
        self.qa_checks = self._register_qa_checks()
    
    def _register_qa_checks(self) -> List[Callable]:
        """Register all QA check functions."""
        return [
            self._check_strategy_leg_availability,
            self._check_margin_requirements,
            self._check_commission_impact,
            self._check_regulatory_compliance,
            self._check_output_consistency,
            self._check_execution_timing,
            self._check_monitoring_alerts,
            self._check_accuracy_audit_trail
        ]
    
    @handle_errors(operation_name="run_qa_pipeline")
    def run_complete_pipeline(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]] = None
    ) -> QAPipelineResult:
        """
        Run complete quality assurance pipeline.
        
        Args:
            selection_result: Selection result to validate
            market_data: Current market data
            execution_context: Execution context parameters
            
        Returns:
            Complete QA pipeline result
        """
        self.logger.info(f"Starting QA pipeline for {len(selection_result.selected_trades)} trades")
        
        start_time = time.time()
        check_results = []
        critical_failures = []
        warnings = []
        
        # Run output validation first
        validation_report = self.output_validator.validate_complete_output(
            selection_result, market_data
        )
        
        if not validation_report.is_valid:
            critical_failures.extend([
                QACheckResult(
                    check_name="output_validation",
                    category=QACheckCategory.STRATEGY_VALIDATION,
                    level=QACheckLevel.CRITICAL,
                    passed=False,
                    message=f"Output validation failed: {len(validation_report.critical_issues)} critical issues",
                    execution_time=0.0,
                    details={'validation_report': validation_report}
                )
            ])
        
        # Run all QA checks
        for check_func in self.qa_checks:
            try:
                check_start = time.time()
                check_result = check_func(selection_result, market_data, execution_context)
                check_result.execution_time = time.time() - check_start
                
                check_results.append(check_result)
                
                if check_result.level == QACheckLevel.CRITICAL and not check_result.passed:
                    critical_failures.append(check_result)
                elif not check_result.passed:
                    warnings.append(check_result.message)
                    
            except Exception as e:
                self.logger.error(f"QA check {check_func.__name__} failed: {str(e)}")
                check_results.append(QACheckResult(
                    check_name=check_func.__name__,
                    category=QACheckCategory.ACCURACY_AUDIT,
                    level=QACheckLevel.HIGH,
                    passed=False,
                    message=f"Check execution failed: {str(e)}",
                    execution_time=0.0
                ))
        
        # Calculate results
        total_time = time.time() - start_time
        total_checks = len(check_results)
        passed_checks = sum(1 for r in check_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Overall pass determination
        overall_pass = (
            len(critical_failures) == 0 and
            validation_report.is_valid and
            total_time < self.thresholds['max_execution_time']
        )
        
        result = QAPipelineResult(
            overall_pass=overall_pass,
            execution_time=total_time,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            check_results=check_results,
            critical_failures=critical_failures,
            warnings=warnings,
            validation_report=validation_report
        )
        
        self.logger.info(f"QA pipeline completed in {total_time:.2f}s. "
                        f"Status: {'PASS' if overall_pass else 'FAIL'} "
                        f"({passed_checks}/{total_checks} checks passed)")
        
        return result
    
    def _check_strategy_leg_availability(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check strategy leg availability across exchanges."""
        
        unavailable_legs = []
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            for leg in strategy.legs:
                # Check if option contract exists and is tradeable
                if not leg.option.bid or not leg.option.ask:
                    unavailable_legs.append(f"{strategy.underlying} {leg.option.strike} {leg.option.option_type}")
                elif leg.option.bid <= 0 or leg.option.ask <= 0:
                    unavailable_legs.append(f"{strategy.underlying} {leg.option.strike} {leg.option.option_type} (invalid prices)")
                elif leg.option.volume is not None and leg.option.volume == 0:
                    unavailable_legs.append(f"{strategy.underlying} {leg.option.strike} {leg.option.option_type} (no volume)")
        
        passed = len(unavailable_legs) == 0
        
        return QACheckResult(
            check_name="strategy_leg_availability",
            category=QACheckCategory.STRATEGY_VALIDATION,
            level=QACheckLevel.CRITICAL,
            passed=passed,
            message=f"Strategy leg availability check: {len(unavailable_legs)} unavailable legs" if not passed else "All strategy legs available",
            execution_time=0.0,
            details={'unavailable_legs': unavailable_legs},
            recommendations=["Verify option chain data freshness", "Check market hours"] if not passed else []
        )
    
    def _check_margin_requirements(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check margin requirement calculations."""
        
        margin_errors = []
        total_margin = 0
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Calculate expected margin requirement
            expected_margin = self._calculate_expected_margin(strategy)
            calculated_margin = strategy.margin_requirement or 0
            
            # Check accuracy
            if expected_margin > 0:
                accuracy = min(calculated_margin, expected_margin) / max(calculated_margin, expected_margin)
                if accuracy < self.thresholds['min_margin_accuracy']:
                    margin_errors.append(f"{strategy.underlying}: Expected {expected_margin:.2f}, Got {calculated_margin:.2f}")
            
            total_margin += calculated_margin
        
        passed = len(margin_errors) == 0
        
        return QACheckResult(
            check_name="margin_requirements",
            category=QACheckCategory.MARGIN_CALCULATIONS,
            level=QACheckLevel.HIGH,
            passed=passed,
            message=f"Margin calculation check: {len(margin_errors)} errors" if not passed else f"Margin calculations accurate (${total_margin:.2f} total)",
            execution_time=0.0,
            details={'margin_errors': margin_errors, 'total_margin': total_margin},
            recommendations=["Review margin calculation methodology"] if not passed else []
        )
    
    def _check_commission_impact(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check commission impact analysis."""
        
        high_impact_trades = []
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Estimate commission cost
            commission_cost = self._estimate_commission_cost(strategy)
            trade_value = float(strategy.net_credit or strategy.net_debit or 100)
            
            # Calculate impact
            if trade_value > 0:
                impact = commission_cost / trade_value
                if impact > self.thresholds['max_commission_impact']:
                    high_impact_trades.append(f"{strategy.underlying}: {impact:.1%} impact")
        
        passed = len(high_impact_trades) == 0
        
        return QACheckResult(
            check_name="commission_impact",
            category=QACheckCategory.COMMISSION_IMPACT,
            level=QACheckLevel.MEDIUM,
            passed=passed,
            message=f"Commission impact check: {len(high_impact_trades)} high-impact trades" if not passed else "Commission impact acceptable",
            execution_time=0.0,
            details={'high_impact_trades': high_impact_trades},
            recommendations=["Consider broker with lower commissions", "Increase trade size"] if not passed else []
        )
    
    def _check_regulatory_compliance(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check regulatory compliance for retail trading."""
        
        compliance_issues = []
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Check for naked short positions (not allowed for retail)
            naked_shorts = self._check_naked_short_positions(strategy)
            if naked_shorts:
                compliance_issues.extend(naked_shorts)
            
            # Check for pattern day trading implications
            if strategy.days_to_expiration and strategy.days_to_expiration < 5:
                compliance_issues.append(f"{strategy.underlying}: Short-term trade may impact PDT status")
            
            # Check for complex strategies (broker approval required)
            if len(strategy.legs) > self.thresholds['max_strategy_complexity']:
                compliance_issues.append(f"{strategy.underlying}: Complex strategy may require broker approval")
        
        passed = len(compliance_issues) == 0
        
        return QACheckResult(
            check_name="regulatory_compliance",
            category=QACheckCategory.REGULATORY_COMPLIANCE,
            level=QACheckLevel.HIGH,
            passed=passed,
            message=f"Regulatory compliance check: {len(compliance_issues)} issues" if not passed else "Regulatory compliance verified",
            execution_time=0.0,
            details={'compliance_issues': compliance_issues},
            recommendations=["Review broker requirements", "Consider strategy modifications"] if not passed else []
        )
    
    def _check_output_consistency(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check output consistency verification."""
        
        consistency_errors = []
        
        # Check that all required fields are present
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Required fields check
            if not strategy.probability_of_profit:
                consistency_errors.append(f"{strategy.underlying}: Missing probability of profit")
            if not strategy.max_loss:
                consistency_errors.append(f"{strategy.underlying}: Missing max loss")
            if not strategy.legs:
                consistency_errors.append(f"{strategy.underlying}: Missing strategy legs")
            
            # Consistency checks
            if strategy.net_credit and strategy.max_loss:
                if float(strategy.net_credit) > float(strategy.max_loss):
                    consistency_errors.append(f"{strategy.underlying}: Net credit exceeds max loss")
        
        passed = len(consistency_errors) == 0
        
        return QACheckResult(
            check_name="output_consistency",
            category=QACheckCategory.ACCURACY_AUDIT,
            level=QACheckLevel.HIGH,
            passed=passed,
            message=f"Output consistency check: {len(consistency_errors)} errors" if not passed else "Output consistency verified",
            execution_time=0.0,
            details={'consistency_errors': consistency_errors},
            recommendations=["Review data validation logic"] if not passed else []
        )
    
    def _check_execution_timing(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check execution timing recommendations."""
        
        timing_recommendations = []
        
        # Check market hours
        current_time = datetime.now()
        if current_time.weekday() >= 5:  # Weekend
            timing_recommendations.append("Consider waiting for market open (weekend)")
        elif current_time.hour < 9 or current_time.hour >= 16:  # Outside market hours
            timing_recommendations.append("Consider executing during market hours")
        
        # Check earnings proximity
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Check if expiration is too close to earnings (would need earnings data)
            if strategy.days_to_expiration and strategy.days_to_expiration < 7:
                timing_recommendations.append(f"{strategy.underlying}: Check earnings calendar before execution")
        
        passed = len(timing_recommendations) == 0
        
        return QACheckResult(
            check_name="execution_timing",
            category=QACheckCategory.EXECUTION_TIMING,
            level=QACheckLevel.MEDIUM,
            passed=passed,
            message=f"Execution timing check: {len(timing_recommendations)} recommendations" if not passed else "Execution timing optimal",
            execution_time=0.0,
            details={'timing_recommendations': timing_recommendations},
            recommendations=timing_recommendations
        )
    
    def _check_monitoring_alerts(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check monitoring and alert requirements."""
        
        monitoring_needs = []
        
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Check for early assignment risk
            if any(leg.option.option_type.value == "PUT" for leg in strategy.legs):
                monitoring_needs.append(f"{strategy.underlying}: Monitor for early assignment risk")
            
            # Check for high theta strategies
            if strategy.days_to_expiration and strategy.days_to_expiration < 21:
                monitoring_needs.append(f"{strategy.underlying}: High theta - monitor daily")
            
            # Check for earnings proximity
            if strategy.days_to_expiration and strategy.days_to_expiration < 30:
                monitoring_needs.append(f"{strategy.underlying}: Set earnings alert")
        
        passed = True  # Monitoring is always recommended
        
        return QACheckResult(
            check_name="monitoring_alerts",
            category=QACheckCategory.MONITORING_ALERTS,
            level=QACheckLevel.LOW,
            passed=passed,
            message=f"Monitoring requirements: {len(monitoring_needs)} items",
            execution_time=0.0,
            details={'monitoring_needs': monitoring_needs},
            recommendations=monitoring_needs
        )
    
    def _check_accuracy_audit_trail(
        self,
        selection_result: SelectionResult,
        market_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]]
    ) -> QACheckResult:
        """Check accuracy audit trail for all recommendations."""
        
        audit_items = []
        
        # Check data sources
        for scored_trade in selection_result.selected_trades:
            strategy = scored_trade.trade_candidate.strategy
            
            # Verify data source timestamps
            for leg in strategy.legs:
                if not leg.option.quote_timestamp:
                    audit_items.append(f"{strategy.underlying}: Missing quote timestamp")
            
            # Check calculation audit trail
            if not hasattr(scored_trade, 'calculation_metadata'):
                audit_items.append(f"{strategy.underlying}: Missing calculation metadata")
        
        passed = len(audit_items) == 0
        
        return QACheckResult(
            check_name="accuracy_audit_trail",
            category=QACheckCategory.ACCURACY_AUDIT,
            level=QACheckLevel.MEDIUM,
            passed=passed,
            message=f"Audit trail check: {len(audit_items)} items" if not passed else "Audit trail complete",
            execution_time=0.0,
            details={'audit_items': audit_items},
            recommendations=["Enhance audit trail logging"] if not passed else []
        )
    
    def _calculate_expected_margin(self, strategy: StrategyDefinition) -> float:
        """Calculate expected margin requirement."""
        
        # Simplified margin calculation
        if strategy.strategy_type.value == "PUT_CREDIT_SPREAD":
            return float(strategy.max_loss or 0)
        elif strategy.strategy_type.value == "CALL_CREDIT_SPREAD":
            return float(strategy.max_loss or 0)
        elif strategy.strategy_type.value == "IRON_CONDOR":
            return float(strategy.max_loss or 0)
        else:
            return float(strategy.net_debit or 0)
    
    def _estimate_commission_cost(self, strategy: StrategyDefinition) -> float:
        """Estimate commission cost."""
        
        # Typical retail commission: $0.65 per contract
        commission_per_contract = 0.65
        num_contracts = len(strategy.legs)
        
        return commission_per_contract * num_contracts
    
    def _check_naked_short_positions(self, strategy: StrategyDefinition) -> List[str]:
        """Check for naked short positions."""
        
        naked_shorts = []
        
        # This is a simplified check - real implementation would be more complex
        sell_legs = [leg for leg in strategy.legs if leg.direction.value == "SELL"]
        buy_legs = [leg for leg in strategy.legs if leg.direction.value == "BUY"]
        
        if len(sell_legs) > len(buy_legs):
            naked_shorts.append(f"{strategy.underlying}: Potential naked short position")
        
        return naked_shorts
    
    def get_qa_summary(self, result: QAPipelineResult) -> str:
        """Generate human-readable QA summary."""
        
        lines = []
        lines.append("QUALITY ASSURANCE SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Overall Status: {'✓ PASS' if result.overall_pass else '✗ FAIL'}")
        lines.append(f"Execution Time: {result.execution_time:.2f}s")
        lines.append(f"Total Checks: {result.total_checks}")
        lines.append(f"Passed: {result.passed_checks}")
        lines.append(f"Failed: {result.failed_checks}")
        lines.append("")
        
        if result.critical_failures:
            lines.append("CRITICAL FAILURES:")
            for failure in result.critical_failures:
                lines.append(f"  ❌ {failure.message}")
                if failure.recommendations:
                    for rec in failure.recommendations:
                        lines.append(f"     → {rec}")
            lines.append("")
        
        if result.warnings:
            lines.append("WARNINGS:")
            for warning in result.warnings[:5]:  # Limit to top 5
                lines.append(f"  ⚠️  {warning}")
            if len(result.warnings) > 5:
                lines.append(f"  ... and {len(result.warnings) - 5} more warnings")
            lines.append("")
        
        # Check category summary
        category_summary = {}
        for check in result.check_results:
            if check.category not in category_summary:
                category_summary[check.category] = {'passed': 0, 'failed': 0}
            if check.passed:
                category_summary[check.category]['passed'] += 1
            else:
                category_summary[check.category]['failed'] += 1
        
        lines.append("CHECK CATEGORY SUMMARY:")
        for category, counts in category_summary.items():
            total = counts['passed'] + counts['failed']
            lines.append(f"  {category.value}: {counts['passed']}/{total} passed")
        
        return '\n'.join(lines)