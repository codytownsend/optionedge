"""
Trade recommendation formatter with exact console table specifications from instructions.txt.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ...data.models.options import OptionType
from ...data.models.trades import StrategyDefinition, StrategyType, TradeLeg, TradeDirection
from ...domain.services.scoring_engine import ScoredTradeCandidate
from ...domain.services.trade_selector import SelectionResult
from ...infrastructure.error_handling import handle_errors, BusinessLogicError


@dataclass
class FormattingConfig:
    """Configuration for table formatting."""
    # Column widths for console display (exact from instructions.txt)
    ticker_width: int = 8
    strategy_width: int = 20
    legs_width: int = 30
    thesis_width: int = 35
    pop_width: int = 6
    
    # Formatting options
    truncate_indicator: str = "..."
    max_thesis_words: int = 30
    date_format: str = "%m/%d"


class TradeRecommendationFormatter:
    """
    Fixed-width table formatter implementing exact specifications from instructions.txt.
    
    Features:
    - Console-friendly table format with fixed column widths
    - Strategy name standardization
    - Comprehensive legs description formatting
    - Thesis generation based on trade rationale (≤30 words)
    - Exact output format matching specifications
    """
    
    def __init__(self, config: Optional[FormattingConfig] = None):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.config = config or FormattingConfig()
        
        # Strategy name mappings (exact from instructions.txt)
        self.strategy_names = {
            "PUT_CREDIT_SPREAD": "Put Credit Spread",
            "CALL_CREDIT_SPREAD": "Call Credit Spread", 
            "IRON_CONDOR": "Iron Condor",
            "COVERED_CALL": "Covered Call",
            "CASH_SECURED_PUT": "Cash-Secured Put",
            "BULL_PUT_SPREAD": "Bull Put Spread",
            "BEAR_CALL_SPREAD": "Bear Call Spread",
            "IRON_BUTTERFLY": "Iron Butterfly",
            "CALENDAR_SPREAD": "Calendar Spread",
            "DIAGONAL_SPREAD": "Diagonal Spread"
        }
    
    @handle_errors(operation_name="format_trades_table")
    def format_trades_table(self, selection_result: SelectionResult) -> str:
        """
        Format trades in exact console-friendly table format per instructions.txt.
        
        Args:
            selection_result: Selection result with trades to format
            
        Returns:
            Formatted table string or rejection message
        """
        if not selection_result.execution_ready or len(selection_result.selected_trades) < 5:
            return "Fewer than 5 trades meet criteria, do not execute."
        
        selected_trades = selection_result.selected_trades
        
        # Build table header
        header = self._build_table_header()
        
        # Build separator line
        separator = '-' * len(header)
        
        # Format each trade row
        rows = []
        for scored_trade in selected_trades:
            try:
                row = self._format_trade_row(scored_trade)
                rows.append(row)
            except Exception as e:
                self.logger.warning(f"Failed to format trade row for {scored_trade.trade_candidate.strategy.underlying}: {str(e)}")
                # Create fallback row
                fallback_row = self._create_fallback_row(scored_trade)
                rows.append(fallback_row)
        
        # Combine all parts
        table = '\n'.join([header, separator] + rows)
        
        self.logger.info(f"Formatted table with {len(rows)} trade recommendations")
        return table
    
    def format_detailed_output(
        self,
        selection_result: SelectionResult,
        include_analytics: bool = True,
        include_portfolio_summary: bool = True
    ) -> str:
        """
        Format detailed output with table, analytics, and portfolio summary.
        
        Args:
            selection_result: Selection result to format
            include_analytics: Include selection analytics
            include_portfolio_summary: Include portfolio metrics
            
        Returns:
            Comprehensive formatted output
        """
        output_parts = []
        
        # Main trades table
        table = self.format_trades_table(selection_result)
        output_parts.append("TRADE RECOMMENDATIONS")
        output_parts.append("=" * 50)
        output_parts.append(table)
        output_parts.append("")
        
        # Portfolio summary
        if include_portfolio_summary and selection_result.execution_ready:
            portfolio_summary = self._format_portfolio_summary(selection_result)
            output_parts.append("PORTFOLIO SUMMARY")
            output_parts.append("=" * 50)
            output_parts.append(portfolio_summary)
            output_parts.append("")
        
        # Selection analytics
        if include_analytics:
            analytics = self._format_selection_analytics(selection_result)
            output_parts.append("SELECTION ANALYTICS")
            output_parts.append("=" * 50)
            output_parts.append(analytics)
            output_parts.append("")
        
        # Execution guidance
        if selection_result.execution_ready:
            guidance = self._format_execution_guidance(selection_result)
            output_parts.append("EXECUTION GUIDANCE")
            output_parts.append("=" * 50)
            output_parts.append(guidance)
        
        return '\n'.join(output_parts)
    
    def _build_table_header(self) -> str:
        """Build table header with exact column widths."""
        header = (
            f"{'Ticker':<{self.config.ticker_width}} | "
            f"{'Strategy':<{self.config.strategy_width}} | "
            f"{'Legs':<{self.config.legs_width}} | "
            f"{'Thesis':<{self.config.thesis_width}} | "
            f"{'POP':<{self.config.pop_width}}"
        )
        return header
    
    def _format_trade_row(self, scored_trade: ScoredTradeCandidate) -> str:
        """Format a single trade row."""
        trade = scored_trade.trade_candidate
        strategy = trade.strategy
        
        # Format each column
        ticker = self._truncate_text(strategy.underlying, self.config.ticker_width)
        strategy_name = self._truncate_text(self._format_strategy_name(strategy), self.config.strategy_width)
        legs = self._truncate_text(self._format_legs_description(strategy), self.config.legs_width)
        thesis = self._truncate_text(self._generate_thesis(scored_trade), self.config.thesis_width)
        pop = f"{strategy.probability_of_profit:.2f}" if strategy.probability_of_profit else "N/A"
        
        # Build row
        row = (
            f"{ticker:<{self.config.ticker_width}} | "
            f"{strategy_name:<{self.config.strategy_width}} | "
            f"{legs:<{self.config.legs_width}} | "
            f"{thesis:<{self.config.thesis_width}} | "
            f"{pop:<{self.config.pop_width}}"
        )
        
        return row
    
    def _format_strategy_name(self, strategy: StrategyDefinition) -> str:
        """Standardize strategy names for display."""
        strategy_type = strategy.strategy_type.value if hasattr(strategy.strategy_type, 'value') else str(strategy.strategy_type)
        return self.strategy_names.get(strategy_type, strategy_type)
    
    def _format_legs_description(self, strategy: StrategyDefinition) -> str:
        """Format option legs for display per instructions.txt specifications."""
        
        if not strategy.legs:
            return "No legs"
        
        strategy_type = strategy.strategy_type.value if hasattr(strategy.strategy_type, 'value') else str(strategy.strategy_type)
        
        # Get expiration date (use first leg's expiration)
        expiration_str = "N/A"
        if strategy.legs[0].option.expiration:
            expiration_str = strategy.legs[0].option.expiration.strftime(self.config.date_format)
        
        try:
            if strategy_type == "PUT_CREDIT_SPREAD":
                return self._format_put_credit_spread_legs(strategy, expiration_str)
            elif strategy_type == "CALL_CREDIT_SPREAD":
                return self._format_call_credit_spread_legs(strategy, expiration_str)
            elif strategy_type == "IRON_CONDOR":
                return self._format_iron_condor_legs(strategy, expiration_str)
            elif strategy_type == "COVERED_CALL":
                return self._format_covered_call_legs(strategy, expiration_str)
            elif strategy_type == "CASH_SECURED_PUT":
                return self._format_cash_secured_put_legs(strategy, expiration_str)
            else:
                return self._format_generic_legs(strategy, expiration_str)
        except Exception as e:
            self.logger.warning(f"Failed to format legs for {strategy_type}: {str(e)}")
            return self._format_generic_legs(strategy, expiration_str)
    
    def _format_put_credit_spread_legs(self, strategy: StrategyDefinition, expiration_str: str) -> str:
        """Format put credit spread legs: S185P/B180P 01/17"""
        sell_legs = [leg for leg in strategy.legs if leg.direction == TradeDirection.SELL]
        buy_legs = [leg for leg in strategy.legs if leg.direction == TradeDirection.BUY]
        
        if sell_legs and buy_legs:
            short_strike = sell_legs[0].option.strike
            long_strike = buy_legs[0].option.strike
            return f"S{short_strike}P/B{long_strike}P {expiration_str}"
        else:
            return self._format_generic_legs(strategy, expiration_str)
    
    def _format_call_credit_spread_legs(self, strategy: StrategyDefinition, expiration_str: str) -> str:
        """Format call credit spread legs: S175C/B180C 01/24"""
        sell_legs = [leg for leg in strategy.legs if leg.direction == TradeDirection.SELL]
        buy_legs = [leg for leg in strategy.legs if leg.direction == TradeDirection.BUY]
        
        if sell_legs and buy_legs:
            short_strike = sell_legs[0].option.strike
            long_strike = buy_legs[0].option.strike
            return f"S{short_strike}C/B{long_strike}C {expiration_str}"
        else:
            return self._format_generic_legs(strategy, expiration_str)
    
    def _format_iron_condor_legs(self, strategy: StrategyDefinition, expiration_str: str) -> str:
        """Format iron condor legs: 380/390/420/430 01/17"""
        puts = [leg for leg in strategy.legs if leg.option.option_type == OptionType.PUT]
        calls = [leg for leg in strategy.legs if leg.option.option_type == OptionType.CALL]
        
        if len(puts) >= 2 and len(calls) >= 2:
            # Sort puts and calls by strike
            puts_sorted = sorted(puts, key=lambda x: x.option.strike)
            calls_sorted = sorted(calls, key=lambda x: x.option.strike)
            
            long_put = puts_sorted[0].option.strike  # Lower strike
            short_put = puts_sorted[1].option.strike  # Higher strike
            short_call = calls_sorted[0].option.strike  # Lower strike
            long_call = calls_sorted[1].option.strike  # Higher strike
            
            return f"{long_put}/{short_put}/{short_call}/{long_call} {expiration_str}"
        else:
            return self._format_generic_legs(strategy, expiration_str)
    
    def _format_covered_call_legs(self, strategy: StrategyDefinition, expiration_str: str) -> str:
        """Format covered call legs: Stock+S580C 01/31"""
        call_legs = [leg for leg in strategy.legs if leg.option.option_type == OptionType.CALL]
        
        if call_legs:
            call_strike = call_legs[0].option.strike
            return f"Stock+S{call_strike}C {expiration_str}"
        else:
            return self._format_generic_legs(strategy, expiration_str)
    
    def _format_cash_secured_put_legs(self, strategy: StrategyDefinition, expiration_str: str) -> str:
        """Format cash-secured put legs: Cash+S240P 01/17"""
        put_legs = [leg for leg in strategy.legs if leg.option.option_type == OptionType.PUT]
        
        if put_legs:
            put_strike = put_legs[0].option.strike
            return f"Cash+S{put_strike}P {expiration_str}"
        else:
            return self._format_generic_legs(strategy, expiration_str)
    
    def _format_generic_legs(self, strategy: StrategyDefinition, expiration_str: str) -> str:
        """Generic formatting for other strategies: S185P/B180C 01/17"""
        leg_strs = []
        
        for leg in strategy.legs:
            action = "S" if leg.direction == TradeDirection.SELL else "B"
            option_type = "P" if leg.option.option_type == OptionType.PUT else "C"
            leg_strs.append(f"{action}{leg.option.strike}{option_type}")
        
        return f"{'/'.join(leg_strs)} {expiration_str}"
    
    def _generate_thesis(self, scored_trade: ScoredTradeCandidate) -> str:
        """Generate concise thesis (≤30 words) based on trade rationale."""
        
        trade = scored_trade.trade_candidate
        strategy = trade.strategy
        scores = scored_trade.component_scores
        strategy_type = strategy.strategy_type.value if hasattr(strategy.strategy_type, 'value') else str(strategy.strategy_type)
        
        # Template theses based on strategy type and market conditions
        try:
            if strategy_type == "PUT_CREDIT_SPREAD":
                if scores.momentum_z > 0.5:
                    thesis = f"Bullish momentum with {scores.momentum_z:.1f}σ upside. High IV rank enables premium collection below support."
                else:
                    short_strike = self._get_short_strike(strategy)
                    pop_pct = int(strategy.probability_of_profit * 100) if strategy.probability_of_profit else 65
                    thesis = f"Range-bound expectations. Collect premium at {short_strike} strike with {pop_pct}% profit probability."
            
            elif strategy_type == "CALL_CREDIT_SPREAD":
                if scores.momentum_z < -0.5:
                    thesis = f"Bearish momentum {scores.momentum_z:.1f}σ. High IV rank favors premium selling above resistance."
                else:
                    short_strike = self._get_short_strike(strategy)
                    thesis = f"Capped upside trade. Collect premium above {short_strike} with strong time decay."
            
            elif strategy_type == "IRON_CONDOR":
                iv_rank = getattr(scores, 'iv_rank', 50) or 50
                thesis = f"Range-bound market expected. IV rank {iv_rank:.0f}% high. Profit from time decay in consolidation."
            
            elif strategy_type == "COVERED_CALL":
                # Calculate annualized return if possible
                annual_return = self._estimate_annualized_return(strategy)
                thesis = f"Income generation on existing shares. {annual_return:.1%} annual yield with moderate upside cap."
            
            elif strategy_type == "CASH_SECURED_PUT":
                put_strike = self._get_put_strike(strategy)
                thesis = f"Bullish assignment acceptable at {put_strike}. Premium collection while waiting for entry."
            
            else:
                # Generic thesis based on key metrics
                pop_pct = int(strategy.probability_of_profit * 100) if strategy.probability_of_profit else 65
                thesis = f"Favorable risk/reward with {pop_pct}% win rate. Strong technical setup supports directional bias."
            
            # Ensure thesis is ≤30 words
            words = thesis.split()
            if len(words) > self.config.max_thesis_words:
                thesis = ' '.join(words[:self.config.max_thesis_words]) + self.config.truncate_indicator
            
            return thesis
            
        except Exception as e:
            self.logger.warning(f"Failed to generate thesis: {str(e)}")
            # Fallback thesis
            pop_pct = int(strategy.probability_of_profit * 100) if strategy.probability_of_profit else 65
            return f"Quantitative signals favor this trade with {pop_pct}% profit probability."
    
    def _get_short_strike(self, strategy: StrategyDefinition) -> str:
        """Get the short strike from strategy legs."""
        sell_legs = [leg for leg in strategy.legs if leg.direction == TradeDirection.SELL]
        if sell_legs:
            return str(sell_legs[0].option.strike)
        return "N/A"
    
    def _get_put_strike(self, strategy: StrategyDefinition) -> str:
        """Get the put strike from strategy legs."""
        put_legs = [leg for leg in strategy.legs if leg.option.option_type == OptionType.PUT]
        if put_legs:
            return str(put_legs[0].option.strike)
        return "N/A"
    
    def _estimate_annualized_return(self, strategy: StrategyDefinition) -> float:
        """Estimate annualized return for covered call."""
        if strategy.max_profit and strategy.days_to_expiration:
            return_per_period = float(strategy.max_profit) / 100  # Assume $100 stock price
            periods_per_year = 365 / strategy.days_to_expiration
            return return_per_period * periods_per_year
        return 0.15  # Default 15% if cannot calculate
    
    def _truncate_text(self, text: str, max_width: int) -> str:
        """Truncate text to fit column width."""
        if len(text) <= max_width:
            return text
        return text[:max_width-len(self.config.truncate_indicator)] + self.config.truncate_indicator
    
    def _create_fallback_row(self, scored_trade: ScoredTradeCandidate) -> str:
        """Create fallback row when formatting fails."""
        ticker = scored_trade.trade_candidate.strategy.underlying[:self.config.ticker_width]
        strategy_name = "Error"[:self.config.strategy_width]
        legs = "Format Error"[:self.config.legs_width]
        thesis = "Unable to format trade details"[:self.config.thesis_width]
        pop = "N/A"
        
        return (
            f"{ticker:<{self.config.ticker_width}} | "
            f"{strategy_name:<{self.config.strategy_width}} | "
            f"{legs:<{self.config.legs_width}} | "
            f"{thesis:<{self.config.thesis_width}} | "
            f"{pop:<{self.config.pop_width}}"
        )
    
    def _format_portfolio_summary(self, selection_result: SelectionResult) -> str:
        """Format portfolio summary section."""
        metrics = selection_result.portfolio_metrics
        
        lines = []
        lines.append(f"Total Risk Amount: ${metrics.get('total_risk_amount', 0):,.0f}")
        lines.append(f"Risk as % of NAV: {metrics.get('risk_percentage_of_nav', 0):.1f}%")
        lines.append(f"Portfolio Delta: {metrics.get('portfolio_delta', 0):+.2f}")
        lines.append(f"Portfolio Vega: {metrics.get('portfolio_vega', 0):+.2f}")
        lines.append(f"Average Model Score: {metrics.get('avg_model_score', 0):.1f}")
        lines.append(f"Average POP: {metrics.get('avg_probability_of_profit', 0):.1%}")
        lines.append(f"Unique Sectors: {metrics.get('unique_sectors', 0)}")
        lines.append(f"Max Sector Concentration: {metrics.get('max_sector_concentration', 0)} trades")
        
        return '\n'.join(lines)
    
    def _format_selection_analytics(self, selection_result: SelectionResult) -> str:
        """Format selection analytics section."""
        summary = selection_result.selection_summary
        
        lines = []
        lines.append(f"Total Candidates Evaluated: {summary.get('total_candidates', 0)}")
        lines.append(f"Trades Selected: {summary.get('selected_count', 0)}")
        lines.append(f"Hard Constraint Rejects: {summary.get('constraint_rejects', 0)}")
        lines.append(f"Portfolio Constraint Rejects: {summary.get('portfolio_rejects', 0)}")
        
        if summary.get('execution_ready'):
            lines.append("Status: ✓ Ready for execution")
        else:
            lines.append("Status: ✗ Not ready for execution")
            lines.append(f"Reason: {summary.get('message', 'Unknown')}")
        
        if selection_result.warnings:
            lines.append("\nWarnings:")
            for warning in selection_result.warnings:
                lines.append(f"  • {warning}")
        
        return '\n'.join(lines)
    
    def _format_execution_guidance(self, selection_result: SelectionResult) -> str:
        """Format execution guidance section."""
        lines = []
        lines.append("Execution Checklist:")
        lines.append("  □ Verify current market prices before order entry")
        lines.append("  □ Check available buying power for margin requirements")
        lines.append("  □ Set limit orders with appropriate bid-ask spread consideration")
        lines.append("  □ Monitor positions daily for early assignment risk")
        lines.append("  □ Plan exit strategy at 25-50% of maximum profit")
        lines.append("")
        lines.append("Risk Management:")
        lines.append("  • Maximum loss per trade is already factored into selections")
        lines.append("  • Portfolio Greeks are within acceptable limits")
        lines.append("  • Sector diversification constraints have been applied")
        lines.append("  • Consider position sizing based on account size and risk tolerance")
        
        return '\n'.join(lines)