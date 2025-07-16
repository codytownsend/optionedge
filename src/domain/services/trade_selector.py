"""
Hierarchical ranking and trade selection algorithm with portfolio constraints.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ...data.models.options import OptionQuote, OptionType
from ...data.models.market_data import StockQuote, TechnicalIndicators
from ...data.models.trades import (
    StrategyDefinition, TradeCandidate, TradeLeg, StrategyType,
    PortfolioGreeks
)
from ...infrastructure.error_handling import (
    handle_errors, BusinessLogicError, SelectionError
)

from .scoring_engine import ScoredTradeCandidate, ComponentScores
from .constraint_engine import HardConstraintValidator, ConstraintValidationResult, GICS_SECTORS
from .portfolio_risk_controller import PortfolioRiskController


class SelectionCriteria(Enum):
    """Selection criteria for trade ranking."""
    MODEL_SCORE = "model_score"
    MOMENTUM_Z = "momentum_z"
    FLOW_Z = "flow_z"
    PROBABILITY_OF_PROFIT = "probability_of_profit"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    LIQUIDITY_SCORE = "liquidity_score"


@dataclass
class SelectionResult:
    """Result of trade selection process."""
    selected_trades: List[ScoredTradeCandidate]
    rejected_trades: List[Tuple[ScoredTradeCandidate, str]]
    selection_summary: Dict[str, Any]
    portfolio_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    execution_ready: bool = True


@dataclass
class RankingHierarchy:
    """Hierarchical ranking criteria with tie-breaking rules."""
    primary: SelectionCriteria = SelectionCriteria.MODEL_SCORE
    secondary: SelectionCriteria = SelectionCriteria.MOMENTUM_Z
    tertiary: SelectionCriteria = SelectionCriteria.FLOW_Z
    quaternary: SelectionCriteria = SelectionCriteria.PROBABILITY_OF_PROFIT
    final_tiebreaker: str = "ticker"  # Alphabetical for true ties


class TradeSelector:
    """
    Hierarchical ranking and selection algorithm implementing exact specifications from instructions.txt.
    
    Features:
    - Hierarchical ranking with specific tie-breaking rules
    - Portfolio constraint validation during selection
    - GICS sector diversification enforcement (max 2 per sector)
    - Portfolio Greeks limits validation
    - Minimum 5 trades requirement with execution blocking
    - Dynamic ranking criteria based on market conditions
    - Comprehensive rejection tracking and analysis
    """
    
    def __init__(self, nav: Decimal = Decimal('100000')):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.nav = nav
        
        # Initialize constraint validator and portfolio controller
        self.constraint_validator = HardConstraintValidator()
        self.portfolio_controller = PortfolioRiskController()
        
        # Selection parameters
        self.max_trades = 5
        self.min_trades_for_execution = 5
        self.max_trades_per_sector = 2
        
        # Default ranking hierarchy
        self.default_ranking = RankingHierarchy()
    
    @handle_errors(operation_name="select_final_trades")
    def select_final_trades(
        self,
        scored_candidates: List[ScoredTradeCandidate],
        current_trades: List[TradeCandidate],
        available_capital: Decimal,
        ranking_hierarchy: Optional[RankingHierarchy] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> SelectionResult:
        """
        Select top 5 trades using hierarchical ranking with portfolio constraints.
        
        Args:
            scored_candidates: Scored trade candidates
            current_trades: Current portfolio positions
            available_capital: Available capital for new trades
            ranking_hierarchy: Custom ranking criteria
            custom_constraints: Custom constraint parameters
            
        Returns:
            Selection result with exactly 5 trades or execution block
        """
        self.logger.info(f"Selecting final trades from {len(scored_candidates)} candidates")
        
        if not scored_candidates:
            return SelectionResult(
                selected_trades=[],
                rejected_trades=[],
                selection_summary={'message': "No trade candidates provided"},
                portfolio_metrics={},
                warnings=["No trade candidates available"],
                execution_ready=False
            )
        
        ranking = ranking_hierarchy or self.default_ranking
        
        # Step 1: Filter trades that pass all hard constraints
        valid_trades, constraint_rejects = self._filter_by_hard_constraints(
            scored_candidates, current_trades, available_capital, custom_constraints
        )
        
        self.logger.info(f"Hard constraint filtering: {len(valid_trades)} valid, {len(constraint_rejects)} rejected")
        
        # Check if we have enough valid trades
        if len(valid_trades) < self.min_trades_for_execution:
            return SelectionResult(
                selected_trades=valid_trades,
                rejected_trades=constraint_rejects,
                selection_summary={
                    'message': f"Fewer than {self.min_trades_for_execution} trades meet criteria, do not execute.",
                    'valid_trades_count': len(valid_trades),
                    'required_trades': self.min_trades_for_execution
                },
                portfolio_metrics={},
                warnings=[f"Only {len(valid_trades)} trades passed constraints (need {self.min_trades_for_execution})"],
                execution_ready=False
            )
        
        # Step 2: Apply hierarchical ranking
        ranked_trades = self._apply_hierarchical_ranking(valid_trades, ranking)
        
        # Step 3: Select trades ensuring portfolio constraints
        selected_trades, portfolio_rejects = self._select_with_portfolio_constraints(
            ranked_trades, current_trades
        )
        
        self.logger.info(f"Portfolio constraint selection: {len(selected_trades)} selected, {len(portfolio_rejects)} rejected")
        
        # Step 4: Final validation
        if len(selected_trades) < self.min_trades_for_execution:
            execution_ready = False
            message = f"Fewer than {self.min_trades_for_execution} trades meet criteria, do not execute."
            warnings = [f"Portfolio constraints reduced selection to {len(selected_trades)} trades"]
        else:
            execution_ready = True
            message = f"Successfully selected {len(selected_trades)} trades for execution"
            warnings = []
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_selection_portfolio_metrics(selected_trades, current_trades)
        
        # Generate selection summary
        selection_summary = {
            'message': message,
            'selected_count': len(selected_trades),
            'total_candidates': len(scored_candidates),
            'constraint_rejects': len(constraint_rejects),
            'portfolio_rejects': len(portfolio_rejects),
            'execution_ready': execution_ready
        }
        
        # Combine all rejections
        all_rejects = constraint_rejects + portfolio_rejects
        
        return SelectionResult(
            selected_trades=selected_trades,
            rejected_trades=all_rejects,
            selection_summary=selection_summary,
            portfolio_metrics=portfolio_metrics,
            warnings=warnings,
            execution_ready=execution_ready
        )
    
    def _filter_by_hard_constraints(
        self,
        candidates: List[ScoredTradeCandidate],
        current_trades: List[TradeCandidate],
        available_capital: Decimal,
        custom_constraints: Optional[Dict[str, Any]]
    ) -> Tuple[List[ScoredTradeCandidate], List[Tuple[ScoredTradeCandidate, str]]]:
        """Filter candidates by hard constraints."""
        
        valid_trades = []
        rejected_trades = []
        
        for scored_candidate in candidates:
            # Validate all hard constraints
            validation_result = self.constraint_validator.validate_trade_constraints(
                scored_candidate.trade_candidate,
                current_trades,
                self.nav,
                available_capital,
                custom_constraints
            )
            
            if validation_result.is_valid:
                valid_trades.append(scored_candidate)
            else:
                # Create rejection reason
                violation_messages = [v.message for v in validation_result.violations]
                rejection_reason = f"Hard constraint violations: {'; '.join(violation_messages)}"
                rejected_trades.append((scored_candidate, rejection_reason))
        
        return valid_trades, rejected_trades
    
    def _apply_hierarchical_ranking(
        self,
        candidates: List[ScoredTradeCandidate],
        ranking: RankingHierarchy
    ) -> List[ScoredTradeCandidate]:
        """Apply hierarchical ranking with tie-breaking rules."""
        
        def ranking_key(scored_candidate: ScoredTradeCandidate) -> Tuple:
            """Generate ranking key tuple for sorting."""
            
            candidate = scored_candidate.trade_candidate
            scores = scored_candidate.component_scores
            
            # Get primary criterion value (negative for descending sort)
            primary_value = self._get_criterion_value(scores, candidate, ranking.primary, descending=True)
            secondary_value = self._get_criterion_value(scores, candidate, ranking.secondary, descending=True)
            tertiary_value = self._get_criterion_value(scores, candidate, ranking.tertiary, descending=True)
            quaternary_value = self._get_criterion_value(scores, candidate, ranking.quaternary, descending=True)
            
            # Final tiebreaker (alphabetical)
            final_tiebreaker = candidate.strategy.underlying
            
            return (primary_value, secondary_value, tertiary_value, quaternary_value, final_tiebreaker)
        
        # Sort by ranking criteria
        sorted_candidates = sorted(candidates, key=ranking_key)
        
        self.logger.debug(f"Applied hierarchical ranking. Top candidate: {sorted_candidates[0].trade_candidate.strategy.underlying} (Score: {sorted_candidates[0].component_scores.model_score})")
        
        return sorted_candidates
    
    def _select_with_portfolio_constraints(
        self,
        ranked_candidates: List[ScoredTradeCandidate],
        current_trades: List[TradeCandidate]
    ) -> Tuple[List[ScoredTradeCandidate], List[Tuple[ScoredTradeCandidate, str]]]:
        """Select trades while enforcing portfolio constraints."""
        
        selected_trades = []
        rejected_trades = []
        
        # Track portfolio state
        portfolio_greeks = self._calculate_current_portfolio_greeks(current_trades)
        sector_count = self._calculate_current_sector_counts(current_trades)
        
        for candidate in ranked_candidates:
            if len(selected_trades) >= self.max_trades:
                rejected_trades.append((candidate, "Maximum trade limit reached"))
                continue
            
            # Check sector diversification
            sector = GICS_SECTORS.get(candidate.trade_candidate.strategy.underlying, "Unknown")
            if sector_count.get(sector, 0) >= self.max_trades_per_sector:
                rejected_trades.append((candidate, f"Sector limit exceeded for {sector} (max {self.max_trades_per_sector})"))
                continue
            
            # Check portfolio Greeks after adding this trade
            projected_trades = [t.trade_candidate for t in selected_trades] + [candidate.trade_candidate]
            violations = self.portfolio_controller.check_portfolio_limits_compliance(
                candidate.trade_candidate, 
                current_trades + [t.trade_candidate for t in selected_trades],
                self.nav
            )
            
            if violations:
                violation_messages = [v.message for v in violations]
                rejected_trades.append((candidate, f"Portfolio limits: {'; '.join(violation_messages)}"))
                continue
            
            # Trade passes all portfolio constraints
            selected_trades.append(candidate)
            sector_count[sector] = sector_count.get(sector, 0) + 1
            
            self.logger.debug(f"Selected trade {len(selected_trades)}: {candidate.trade_candidate.strategy.underlying}")
        
        return selected_trades, rejected_trades
    
    def _get_criterion_value(
        self,
        scores: ComponentScores,
        candidate: TradeCandidate,
        criterion: SelectionCriteria,
        descending: bool = True
    ) -> float:
        """Get value for a specific ranking criterion."""
        
        if criterion == SelectionCriteria.MODEL_SCORE:
            value = scores.model_score
        elif criterion == SelectionCriteria.MOMENTUM_Z:
            value = scores.momentum_z
        elif criterion == SelectionCriteria.FLOW_Z:
            value = scores.flow_z
        elif criterion == SelectionCriteria.PROBABILITY_OF_PROFIT:
            value = candidate.strategy.probability_of_profit or 0.0
        elif criterion == SelectionCriteria.RISK_REWARD_RATIO:
            if candidate.strategy.credit_to_max_loss_ratio:
                value = candidate.strategy.credit_to_max_loss_ratio
            elif candidate.strategy.max_profit and candidate.strategy.max_loss:
                value = float(candidate.strategy.max_profit) / float(candidate.strategy.max_loss)
            else:
                value = 0.0
        elif criterion == SelectionCriteria.LIQUIDITY_SCORE:
            value = scores.liquidity_score
        else:
            value = 0.0
        
        # Return negative for descending sort
        return -value if descending else value
    
    def _calculate_current_portfolio_greeks(self, current_trades: List[TradeCandidate]) -> Dict[str, float]:
        """Calculate current portfolio Greeks."""
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        
        for trade in current_trades:
            for leg in trade.strategy.legs:
                multiplier = 100  # Options multiplier
                position_sign = 1 if leg.direction.value == "BUY" else -1
                
                if leg.option.greeks:
                    total_delta += float(leg.option.greeks.delta or 0) * leg.quantity * multiplier * position_sign
                    total_gamma += float(leg.option.greeks.gamma or 0) * leg.quantity * multiplier * position_sign
                    total_theta += float(leg.option.greeks.theta or 0) * leg.quantity * multiplier * position_sign
                    total_vega += float(leg.option.greeks.vega or 0) * leg.quantity * multiplier * position_sign
                    total_rho += float(leg.option.greeks.rho or 0) * leg.quantity * multiplier * position_sign
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }
    
    def _calculate_current_sector_counts(self, current_trades: List[TradeCandidate]) -> Dict[str, int]:
        """Calculate current sector trade counts."""
        
        sector_count = {}
        
        for trade in current_trades:
            sector = GICS_SECTORS.get(trade.strategy.underlying, "Unknown")
            sector_count[sector] = sector_count.get(sector, 0) + 1
        
        return sector_count
    
    def _calculate_selection_portfolio_metrics(
        self,
        selected_trades: List[ScoredTradeCandidate],
        current_trades: List[TradeCandidate]
    ) -> Dict[str, float]:
        """Calculate portfolio metrics for selected trades."""
        
        if not selected_trades:
            return {}
        
        # Calculate projected portfolio Greeks
        all_trades = current_trades + [t.trade_candidate for t in selected_trades]
        projected_greeks = self._calculate_current_portfolio_greeks(all_trades)
        
        # Calculate sector diversification
        sector_counts = {}
        for trade in selected_trades:
            sector = GICS_SECTORS.get(trade.trade_candidate.strategy.underlying, "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Calculate average metrics
        avg_model_score = sum(t.component_scores.model_score for t in selected_trades) / len(selected_trades)
        avg_pop = sum(t.trade_candidate.strategy.probability_of_profit or 0 for t in selected_trades) / len(selected_trades)
        
        # Calculate total risk
        total_risk = sum(float(t.trade_candidate.strategy.max_loss or 0) for t in selected_trades)
        
        return {
            'portfolio_delta': projected_greeks['delta'],
            'portfolio_vega': projected_greeks['vega'],
            'portfolio_theta': projected_greeks['theta'],
            'avg_model_score': avg_model_score,
            'avg_probability_of_profit': avg_pop,
            'total_risk_amount': total_risk,
            'risk_percentage_of_nav': total_risk / float(self.nav) * 100,
            'unique_sectors': len(sector_counts),
            'max_sector_concentration': max(sector_counts.values()) if sector_counts else 0
        }
    
    @handle_errors(operation_name="validate_all_constraints")
    def validate_all_constraints(
        self,
        candidate: TradeCandidate,
        current_trades: List[TradeCandidate],
        available_capital: Decimal,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Master validation function for all hard constraints as specified in instructions.txt.
        
        Args:
            candidate: Trade candidate to validate
            current_trades: Current portfolio trades
            available_capital: Available capital
            custom_constraints: Custom constraint parameters
            
        Returns:
            True if all constraints pass, False otherwise
        """
        # Use the constraint validator
        validation_result = self.constraint_validator.validate_trade_constraints(
            candidate, current_trades, self.nav, available_capital, custom_constraints
        )
        
        return validation_result.is_valid
    
    def get_selection_analytics(
        self,
        selection_result: SelectionResult
    ) -> Dict[str, Any]:
        """Get detailed analytics on the selection process."""
        
        analytics = {
            'selection_efficiency': len(selection_result.selected_trades) / (len(selection_result.selected_trades) + len(selection_result.rejected_trades)) if (len(selection_result.selected_trades) + len(selection_result.rejected_trades)) > 0 else 0,
            'rejection_analysis': {},
            'score_distribution': {},
            'sector_distribution': {},
            'strategy_distribution': {}
        }
        
        # Analyze rejections by reason
        rejection_reasons = {}
        for _, reason in selection_result.rejected_trades:
            category = self._categorize_rejection_reason(reason)
            rejection_reasons[category] = rejection_reasons.get(category, 0) + 1
        analytics['rejection_analysis'] = rejection_reasons
        
        # Analyze selected trades
        if selection_result.selected_trades:
            scores = [t.component_scores.model_score for t in selection_result.selected_trades]
            analytics['score_distribution'] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'range': max(scores) - min(scores)
            }
            
            # Sector distribution
            sectors = {}
            strategies = {}
            for trade in selection_result.selected_trades:
                sector = GICS_SECTORS.get(trade.trade_candidate.strategy.underlying, "Unknown")
                sectors[sector] = sectors.get(sector, 0) + 1
                
                strategy_type = trade.trade_candidate.strategy.strategy_type.value
                strategies[strategy_type] = strategies.get(strategy_type, 0) + 1
            
            analytics['sector_distribution'] = sectors
            analytics['strategy_distribution'] = strategies
        
        return analytics
    
    def _categorize_rejection_reason(self, reason: str) -> str:
        """Categorize rejection reason for analytics."""
        
        reason_lower = reason.lower()
        
        if 'quote' in reason_lower or 'freshness' in reason_lower:
            return 'Quote Freshness'
        elif 'probability' in reason_lower or 'pop' in reason_lower:
            return 'Probability of Profit'
        elif 'credit' in reason_lower and 'ratio' in reason_lower:
            return 'Credit-to-Max-Loss Ratio'
        elif 'max loss' in reason_lower or 'maximum loss' in reason_lower:
            return 'Maximum Loss'
        elif 'capital' in reason_lower:
            return 'Capital Requirements'
        elif 'liquidity' in reason_lower:
            return 'Liquidity'
        elif 'delta' in reason_lower:
            return 'Portfolio Delta'
        elif 'vega' in reason_lower:
            return 'Portfolio Vega'
        elif 'sector' in reason_lower:
            return 'Sector Diversification'
        elif 'portfolio' in reason_lower:
            return 'Portfolio Constraints'
        else:
            return 'Other'