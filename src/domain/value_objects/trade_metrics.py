"""
Domain value objects for trade metrics and performance analysis.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from .greeks import DomainGreeks


class PerformanceRating(str, Enum):
    """Performance rating levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    TERRIBLE = "terrible"


class RiskCategory(str, Enum):
    """Risk category classifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"


@dataclass(frozen=True)
class ProfitabilityMetrics:
    """
    Immutable value object for profitability analysis.
    
    Contains comprehensive profitability calculations and assessments
    for trading strategies and positions.
    """
    
    # Basic profitability
    net_premium: Optional[Decimal] = None
    max_profit: Optional[Decimal] = None
    max_loss: Optional[Decimal] = None
    current_pnl: Optional[Decimal] = None
    
    # Probability metrics
    probability_of_profit: Optional[float] = None
    probability_of_max_profit: Optional[float] = None
    probability_of_max_loss: Optional[float] = None
    
    # Efficiency metrics
    return_on_capital: Optional[float] = None
    return_on_margin: Optional[float] = None
    credit_to_max_loss_ratio: Optional[float] = None
    profit_efficiency: Optional[float] = None
    
    # Risk-adjusted returns
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        object.__setattr__(self, '_profit_factor', self._calculate_profit_factor())
        object.__setattr__(self, '_risk_reward_ratio', self._calculate_risk_reward_ratio())
        object.__setattr__(self, '_breakeven_probability', self._calculate_breakeven_probability())
    
    def _calculate_profit_factor(self) -> Optional[float]:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.max_profit and self.max_loss and self.max_loss != 0:
            return float(abs(self.max_profit) / abs(self.max_loss))
        return None
    
    def _calculate_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio."""
        if self.max_profit and self.max_loss and self.max_profit != 0:
            return float(abs(self.max_loss) / abs(self.max_profit))
        return None
    
    def _calculate_breakeven_probability(self) -> Optional[float]:
        """Estimate probability needed to break even."""
        if self._risk_reward_ratio:
            # Simplified calculation: P(win) > Risk / (Risk + Reward)
            return self._risk_reward_ratio / (1 + self._risk_reward_ratio)
        return None
    
    @property
    def profit_factor(self) -> Optional[float]:
        """Get profit factor."""
        return self._profit_factor
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Get risk-reward ratio."""
        return self._risk_reward_ratio
    
    @property
    def breakeven_probability(self) -> Optional[float]:
        """Get minimum probability needed to break even."""
        return self._breakeven_probability
    
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.current_pnl is not None and self.current_pnl > 0
    
    def is_favorable_odds(self) -> bool:
        """Check if trade has favorable probability vs required breakeven."""
        if self.probability_of_profit and self.breakeven_probability:
            return self.probability_of_profit > self.breakeven_probability
        return False
    
    def get_profitability_rating(self) -> PerformanceRating:
        """Get overall profitability rating."""
        score = 0
        factors = 0
        
        # Probability of profit component (30%)
        if self.probability_of_profit:
            if self.probability_of_profit >= 0.75:
                score += 30
            elif self.probability_of_profit >= 0.65:
                score += 25
            elif self.probability_of_profit >= 0.55:
                score += 20
            elif self.probability_of_profit >= 0.45:
                score += 15
            else:
                score += 10
            factors += 1
        
        # Return on capital component (25%)
        if self.return_on_capital:
            # Annualized return thresholds
            if self.return_on_capital >= 0.50:  # 50%+
                score += 25
            elif self.return_on_capital >= 0.30:  # 30%+
                score += 20
            elif self.return_on_capital >= 0.15:  # 15%+
                score += 15
            elif self.return_on_capital >= 0.05:  # 5%+
                score += 10
            else:
                score += 5
            factors += 1
        
        # Risk-reward ratio component (25%)
        if self.risk_reward_ratio:
            if self.risk_reward_ratio <= 1.5:
                score += 25
            elif self.risk_reward_ratio <= 2.5:
                score += 20
            elif self.risk_reward_ratio <= 4.0:
                score += 15
            elif self.risk_reward_ratio <= 6.0:
                score += 10
            else:
                score += 5
            factors += 1
        
        # Profit efficiency component (20%)
        if self.profit_efficiency:
            if self.profit_efficiency >= 0.8:
                score += 20
            elif self.profit_efficiency >= 0.6:
                score += 16
            elif self.profit_efficiency >= 0.4:
                score += 12
            elif self.profit_efficiency >= 0.2:
                score += 8
            else:
                score += 4
            factors += 1
        
        # Calculate average score
        if factors == 0:
            return PerformanceRating.AVERAGE
        
        avg_score = score / factors
        
        if avg_score >= 25:
            return PerformanceRating.EXCELLENT
        elif avg_score >= 20:
            return PerformanceRating.GOOD
        elif avg_score >= 15:
            return PerformanceRating.AVERAGE
        elif avg_score >= 10:
            return PerformanceRating.POOR
        else:
            return PerformanceRating.TERRIBLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'net_premium': float(self.net_premium) if self.net_premium else None,
            'max_profit': float(self.max_profit) if self.max_profit else None,
            'max_loss': float(self.max_loss) if self.max_loss else None,
            'current_pnl': float(self.current_pnl) if self.current_pnl else None,
            'probability_of_profit': self.probability_of_profit,
            'probability_of_max_profit': self.probability_of_max_profit,
            'probability_of_max_loss': self.probability_of_max_loss,
            'return_on_capital': self.return_on_capital,
            'return_on_margin': self.return_on_margin,
            'credit_to_max_loss_ratio': self.credit_to_max_loss_ratio,
            'profit_efficiency': self.profit_efficiency,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'profit_factor': self.profit_factor,
            'risk_reward_ratio': self.risk_reward_ratio,
            'breakeven_probability': self.breakeven_probability,
            'is_profitable': self.is_profitable(),
            'is_favorable_odds': self.is_favorable_odds(),
            'profitability_rating': self.get_profitability_rating().value
        }


@dataclass(frozen=True)
class RiskMetrics:
    """
    Immutable value object for risk analysis.
    
    Contains comprehensive risk calculations and assessments
    for trading strategies and positions.
    """
    
    # Basic risk measures
    value_at_risk_95: Optional[Decimal] = None
    conditional_var_95: Optional[Decimal] = None
    maximum_drawdown: Optional[Decimal] = None
    
    # Greeks-based risk
    delta_risk: Optional[float] = None
    gamma_risk: Optional[float] = None
    vega_risk: Optional[float] = None
    theta_risk: Optional[float] = None
    
    # Time-based risk
    time_to_expiration: Optional[int] = None
    time_decay_risk: Optional[float] = None
    
    # Liquidity risk
    liquidity_score: Optional[float] = None
    bid_ask_spread_risk: Optional[float] = None
    
    # Concentration risk
    position_size_risk: Optional[float] = None
    sector_concentration: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived risk metrics."""
        object.__setattr__(self, '_overall_risk_score', self._calculate_overall_risk_score())
        object.__setattr__(self, '_risk_category', self._determine_risk_category())
    
    def _calculate_overall_risk_score(self) -> float:
        """Calculate overall risk score (0-100, higher is riskier)."""
        score = 0
        components = 0
        
        # VaR component (25%)
        if self.value_at_risk_95:
            var_score = min(100, abs(float(self.value_at_risk_95)) / 1000 * 100)
            score += var_score * 0.25
            components += 1
        
        # Greeks risk component (30%)
        greeks_risks = [self.delta_risk, self.gamma_risk, self.vega_risk]
        valid_greeks = [r for r in greeks_risks if r is not None]
        if valid_greeks:
            avg_greeks_risk = sum(valid_greeks) / len(valid_greeks)
            score += avg_greeks_risk * 100 * 0.30
            components += 1
        
        # Time decay risk component (20%)
        if self.time_decay_risk:
            score += self.time_decay_risk * 100 * 0.20
            components += 1
        
        # Liquidity risk component (15%)
        if self.liquidity_score:
            # Invert liquidity score (lower liquidity = higher risk)
            liquidity_risk = 1 - self.liquidity_score
            score += liquidity_risk * 100 * 0.15
            components += 1
        
        # Position size risk component (10%)
        if self.position_size_risk:
            score += self.position_size_risk * 100 * 0.10
            components += 1
        
        return score / components if components > 0 else 50  # Default moderate risk
    
    def _determine_risk_category(self) -> RiskCategory:
        """Determine risk category based on overall risk score."""
        risk_score = self._overall_risk_score
        
        if risk_score <= 25:
            return RiskCategory.CONSERVATIVE
        elif risk_score <= 50:
            return RiskCategory.MODERATE
        elif risk_score <= 75:
            return RiskCategory.AGGRESSIVE
        else:
            return RiskCategory.SPECULATIVE
    
    @property
    def overall_risk_score(self) -> float:
        """Get overall risk score."""
        return self._overall_risk_score
    
    @property
    def risk_category(self) -> RiskCategory:
        """Get risk category."""
        return self._risk_category
    
    def is_high_risk(self) -> bool:
        """Check if position is considered high risk."""
        return self.risk_category in [RiskCategory.AGGRESSIVE, RiskCategory.SPECULATIVE]
    
    def exceeds_risk_tolerance(self, max_risk_score: float = 60) -> bool:
        """Check if risk exceeds specified tolerance."""
        return self.overall_risk_score > max_risk_score
    
    def get_risk_warnings(self) -> List[str]:
        """Get list of risk warnings."""
        warnings = []
        
        if self.overall_risk_score > 75:
            warnings.append("Overall risk level is very high")
        
        if self.delta_risk and self.delta_risk > 0.8:
            warnings.append("High directional risk exposure")
        
        if self.gamma_risk and self.gamma_risk > 0.8:
            warnings.append("High gamma risk - position sensitive to large moves")
        
        if self.vega_risk and self.vega_risk > 0.8:
            warnings.append("High volatility risk exposure")
        
        if self.liquidity_score and self.liquidity_score < 0.3:
            warnings.append("Poor liquidity - may be difficult to exit position")
        
        if self.time_to_expiration and self.time_to_expiration < 7:
            warnings.append("Position expires within one week")
        
        if self.bid_ask_spread_risk and self.bid_ask_spread_risk > 0.5:
            warnings.append("Wide bid-ask spreads increase execution risk")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'value_at_risk_95': float(self.value_at_risk_95) if self.value_at_risk_95 else None,
            'conditional_var_95': float(self.conditional_var_95) if self.conditional_var_95 else None,
            'maximum_drawdown': float(self.maximum_drawdown) if self.maximum_drawdown else None,
            'delta_risk': self.delta_risk,
            'gamma_risk': self.gamma_risk,
            'vega_risk': self.vega_risk,
            'theta_risk': self.theta_risk,
            'time_to_expiration': self.time_to_expiration,
            'time_decay_risk': self.time_decay_risk,
            'liquidity_score': self.liquidity_score,
            'bid_ask_spread_risk': self.bid_ask_spread_risk,
            'position_size_risk': self.position_size_risk,
            'sector_concentration': self.sector_concentration,
            'overall_risk_score': self.overall_risk_score,
            'risk_category': self.risk_category.value,
            'is_high_risk': self.is_high_risk(),
            'risk_warnings': self.get_risk_warnings()
        }


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Immutable value object for performance tracking and analysis.
    
    Contains historical performance data and derived performance statistics.
    """
    
    # Return metrics
    total_return: Optional[float] = None
    annualized_return: Optional[float] = None
    excess_return: Optional[float] = None
    
    # Risk-adjusted performance
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    treynor_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Win/loss statistics
    win_rate: Optional[float] = None
    average_win: Optional[Decimal] = None
    average_loss: Optional[Decimal] = None
    largest_win: Optional[Decimal] = None
    largest_loss: Optional[Decimal] = None
    
    # Consistency metrics
    volatility: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Drawdown analysis
    max_drawdown: Optional[float] = None
    max_drawdown_duration: Optional[int] = None
    current_drawdown: Optional[float] = None
    
    # Trade frequency
    total_trades: Optional[int] = None
    trades_per_month: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived performance metrics."""
        object.__setattr__(self, '_profit_factor', self._calculate_profit_factor())
        object.__setattr__(self, '_kelly_criterion', self._calculate_kelly_criterion())
        object.__setattr__(self, '_performance_rating', self._calculate_performance_rating())
    
    def _calculate_profit_factor(self) -> Optional[float]:
        """Calculate profit factor."""
        if (self.average_win and self.average_loss and 
            self.win_rate and self.average_loss != 0):
            
            avg_win = float(self.average_win)
            avg_loss = float(abs(self.average_loss))
            gross_profit = avg_win * self.win_rate
            gross_loss = avg_loss * (1 - self.win_rate)
            
            if gross_loss > 0:
                return gross_profit / gross_loss
        
        return None
    
    def _calculate_kelly_criterion(self) -> Optional[float]:
        """Calculate optimal Kelly fraction."""
        if (self.win_rate and self.average_win and self.average_loss and 
            self.average_loss != 0):
            
            win_prob = self.win_rate
            loss_prob = 1 - win_prob
            win_amount = float(self.average_win)
            loss_amount = float(abs(self.average_loss))
            
            if loss_amount > 0:
                kelly = (win_prob * win_amount - loss_prob * loss_amount) / win_amount
                return max(0, min(0.25, kelly))  # Cap at 25% for safety
        
        return None
    
    def _calculate_performance_rating(self) -> PerformanceRating:
        """Calculate overall performance rating."""
        score = 0
        factors = 0
        
        # Sharpe ratio component (30%)
        if self.sharpe_ratio:
            if self.sharpe_ratio >= 2.0:
                score += 30
            elif self.sharpe_ratio >= 1.5:
                score += 25
            elif self.sharpe_ratio >= 1.0:
                score += 20
            elif self.sharpe_ratio >= 0.5:
                score += 15
            else:
                score += 10
            factors += 1
        
        # Win rate component (25%)
        if self.win_rate:
            if self.win_rate >= 0.70:
                score += 25
            elif self.win_rate >= 0.60:
                score += 20
            elif self.win_rate >= 0.50:
                score += 15
            elif self.win_rate >= 0.40:
                score += 10
            else:
                score += 5
            factors += 1
        
        # Profit factor component (25%)
        if self._profit_factor:
            if self._profit_factor >= 2.0:
                score += 25
            elif self._profit_factor >= 1.5:
                score += 20
            elif self._profit_factor >= 1.2:
                score += 15
            elif self._profit_factor >= 1.0:
                score += 10
            else:
                score += 5
            factors += 1
        
        # Max drawdown component (20%)
        if self.max_drawdown:
            if self.max_drawdown <= 0.05:  # 5%
                score += 20
            elif self.max_drawdown <= 0.10:  # 10%
                score += 16
            elif self.max_drawdown <= 0.20:  # 20%
                score += 12
            elif self.max_drawdown <= 0.30:  # 30%
                score += 8
            else:
                score += 4
            factors += 1
        
        if factors == 0:
            return PerformanceRating.AVERAGE
        
        avg_score = score / factors
        
        if avg_score >= 22:
            return PerformanceRating.EXCELLENT
        elif avg_score >= 18:
            return PerformanceRating.GOOD
        elif avg_score >= 14:
            return PerformanceRating.AVERAGE
        elif avg_score >= 10:
            return PerformanceRating.POOR
        else:
            return PerformanceRating.TERRIBLE
    
    @property
    def profit_factor(self) -> Optional[float]:
        """Get profit factor."""
        return self._profit_factor
    
    @property
    def kelly_criterion(self) -> Optional[float]:
        """Get Kelly criterion optimal fraction."""
        return self._kelly_criterion
    
    @property
    def performance_rating(self) -> PerformanceRating:
        """Get overall performance rating."""
        return self._performance_rating
    
    def is_outperforming(self, benchmark_return: float = 0.05) -> bool:
        """Check if performance exceeds benchmark."""
        return (self.annualized_return is not None and 
                self.annualized_return > benchmark_return)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'excess_return': self.excess_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'treynor_ratio': self.treynor_ratio,
            'information_ratio': self.information_ratio,
            'win_rate': self.win_rate,
            'average_win': float(self.average_win) if self.average_win else None,
            'average_loss': float(self.average_loss) if self.average_loss else None,
            'largest_win': float(self.largest_win) if self.largest_win else None,
            'largest_loss': float(self.largest_loss) if self.largest_loss else None,
            'volatility': self.volatility,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'current_drawdown': self.current_drawdown,
            'total_trades': self.total_trades,
            'trades_per_month': self.trades_per_month,
            'profit_factor': self.profit_factor,
            'kelly_criterion': self.kelly_criterion,
            'performance_rating': self.performance_rating.value,
            'is_outperforming': self.is_outperforming()
        }


@dataclass(frozen=True)
class TradeMetrics:
    """
    Comprehensive trade metrics value object.
    
    Combines profitability, risk, and performance metrics into
    a unified view of trade quality and characteristics.
    """
    
    profitability: ProfitabilityMetrics
    risk: RiskMetrics
    performance: Optional[PerformanceMetrics] = None
    greeks: Optional[DomainGreeks] = None
    
    # Trade characteristics
    trade_id: Optional[str] = None
    underlying_symbol: Optional[str] = None
    strategy_type: Optional[str] = None
    entry_date: Optional[date] = None
    exit_date: Optional[date] = None
    
    def __post_init__(self):
        """Calculate overall trade quality score."""
        object.__setattr__(self, '_quality_score', self._calculate_quality_score())
        object.__setattr__(self, '_trade_grade', self._assign_trade_grade())
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall trade quality score (0-100)."""
        score = 0
        components = 0
        
        # Profitability component (40%)
        prof_rating = self.profitability.get_profitability_rating()
        prof_scores = {
            PerformanceRating.EXCELLENT: 40,
            PerformanceRating.GOOD: 32,
            PerformanceRating.AVERAGE: 24,
            PerformanceRating.POOR: 16,
            PerformanceRating.TERRIBLE: 8
        }
        score += prof_scores.get(prof_rating, 24)
        components += 1
        
        # Risk component (35%) - inverted (lower risk = higher score)
        risk_score = 35 - (self.risk.overall_risk_score * 0.35)
        score += max(0, risk_score)
        components += 1
        
        # Performance component (25%) - if available
        if self.performance:
            perf_rating = self.performance.performance_rating
            perf_scores = {
                PerformanceRating.EXCELLENT: 25,
                PerformanceRating.GOOD: 20,
                PerformanceRating.AVERAGE: 15,
                PerformanceRating.POOR: 10,
                PerformanceRating.TERRIBLE: 5
            }
            score += perf_scores.get(perf_rating, 15)
            components += 1
        
        return score / components if components > 0 else 50
    
    def _assign_trade_grade(self) -> str:
        """Assign letter grade based on quality score."""
        score = self._quality_score
        
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 45:
            return "D+"
        elif score >= 40:
            return "D"
        elif score >= 35:
            return "D-"
        else:
            return "F"
    
    @property
    def quality_score(self) -> float:
        """Get overall trade quality score."""
        return self._quality_score
    
    @property
    def trade_grade(self) -> str:
        """Get trade letter grade."""
        return self._trade_grade
    
    def is_high_quality_trade(self) -> bool:
        """Check if trade is considered high quality."""
        return self.quality_score >= 70
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get comprehensive trade summary."""
        return {
            'trade_id': self.trade_id,
            'underlying_symbol': self.underlying_symbol,
            'strategy_type': self.strategy_type,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'quality_score': self.quality_score,
            'trade_grade': self.trade_grade,
            'is_high_quality': self.is_high_quality_trade(),
            'profitability_rating': self.profitability.get_profitability_rating().value,
            'risk_category': self.risk.risk_category.value,
            'performance_rating': self.performance.performance_rating.value if self.performance else None,
            'key_metrics': {
                'probability_of_profit': self.profitability.probability_of_profit,
                'return_on_capital': self.profitability.return_on_capital,
                'risk_reward_ratio': self.profitability.risk_reward_ratio,
                'overall_risk_score': self.risk.overall_risk_score,
                'max_profit': float(self.profitability.max_profit) if self.profitability.max_profit else None,
                'max_loss': float(self.profitability.max_loss) if self.profitability.max_loss else None
            }
        }
    
    def compare_to(self, other: 'TradeMetrics') -> Dict[str, Any]:
        """Compare this trade to another trade."""
        comparison = {
            'quality_score_diff': self.quality_score - other.quality_score,
            'grade_comparison': f"{self.trade_grade} vs {other.trade_grade}",
            'better_trade': self.trade_id if self.quality_score > other.quality_score else other.trade_id,
            'profitability_comparison': {},
            'risk_comparison': {}
        }
        
        # Profitability comparison
        if self.profitability.probability_of_profit and other.profitability.probability_of_profit:
            comparison['profitability_comparison']['pop_diff'] = (
                self.profitability.probability_of_profit - other.profitability.probability_of_profit
            )
        
        if self.profitability.return_on_capital and other.profitability.return_on_capital:
            comparison['profitability_comparison']['roc_diff'] = (
                self.profitability.return_on_capital - other.profitability.return_on_capital
            )
        
        # Risk comparison
        comparison['risk_comparison']['risk_score_diff'] = (
            self.risk.overall_risk_score - other.risk.overall_risk_score
        )
        comparison['risk_comparison']['lower_risk'] = (
            self.trade_id if self.risk.overall_risk_score < other.risk.overall_risk_score else other.trade_id
        )
        
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        result = {
            'trade_metrics': {
                'quality_score': self.quality_score,
                'trade_grade': self.trade_grade,
                'is_high_quality': self.is_high_quality_trade()
            },
            'profitability': self.profitability.to_dict(),
            'risk': self.risk.to_dict()
        }
        
        if self.performance:
            result['performance'] = self.performance.to_dict()
        
        if self.greeks:
            result['greeks'] = self.greeks.get_risk_summary()
        
        return result


class TradeAnalysis:
    """
    Utility class for trade metrics analysis and calculations.
    
    Provides static methods for analyzing trade metrics across
    multiple trades and generating insights.
    """
    
    @staticmethod
    def analyze_trade_portfolio(trades: List[TradeMetrics]) -> Dict[str, Any]:
        """Analyze a portfolio of trades."""
        if not trades:
            return {}
        
        # Basic statistics
        total_trades = len(trades)
        quality_scores = [trade.quality_score for trade in trades]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Grade distribution
        grade_counts = {}
        for trade in trades:
            grade = trade.trade_grade
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        # High quality trade percentage
        high_quality_trades = sum(1 for trade in trades if trade.is_high_quality_trade())
        high_quality_pct = high_quality_trades / total_trades
        
        # Risk distribution
        risk_categories = {}
        for trade in trades:
            risk_cat = trade.risk.risk_category.value
            risk_categories[risk_cat] = risk_categories.get(risk_cat, 0) + 1
        
        return {
            'total_trades': total_trades,
            'average_quality_score': avg_quality,
            'high_quality_percentage': high_quality_pct,
            'grade_distribution': grade_counts,
            'risk_distribution': risk_categories,
            'best_trade_id': max(trades, key=lambda t: t.quality_score).trade_id,
            'worst_trade_id': min(trades, key=lambda t: t.quality_score).trade_id
        }
    
    @staticmethod
    def find_trade_patterns(trades: List[TradeMetrics]) -> Dict[str, Any]:
        """Find patterns in trade performance."""
        if len(trades) < 5:
            return {'message': 'Insufficient trades for pattern analysis'}
        
        patterns = {}
        
        # Strategy performance by type
        strategy_performance = {}
        for trade in trades:
            if trade.strategy_type:
                if trade.strategy_type not in strategy_performance:
                    strategy_performance[trade.strategy_type] = []
                strategy_performance[trade.strategy_type].append(trade.quality_score)
        
        # Calculate average performance by strategy
        for strategy, scores in strategy_performance.items():
            patterns[f'{strategy}_avg_quality'] = sum(scores) / len(scores)
        
        # Risk vs return analysis
        high_risk_trades = [t for t in trades if t.risk.is_high_risk()]
        if high_risk_trades:
            high_risk_avg_quality = sum(t.quality_score for t in high_risk_trades) / len(high_risk_trades)
            patterns['high_risk_avg_quality'] = high_risk_avg_quality
        
        return patterns