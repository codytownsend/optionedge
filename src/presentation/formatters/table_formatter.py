"""Table formatting utilities for displaying trade data."""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

from ...data.models.trades import TradeCandidate


class TableFormatter:
    """Base class for table formatting."""
    
    def __init__(self, max_width: int = 120):
        self.max_width = max_width
    
    def format_table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
        """Format data into a table."""
        if not rows:
            return self._format_empty_table(title)
        
        # Calculate column widths
        col_widths = self._calculate_column_widths(headers, rows)
        
        # Build table
        table_lines = []
        
        # Add title if provided
        if title:
            table_lines.append(self._format_title(title))
            table_lines.append("")
        
        # Add header
        table_lines.append(self._format_header(headers, col_widths))
        table_lines.append(self._format_separator(col_widths))
        
        # Add rows
        for row in rows:
            table_lines.append(self._format_row(row, col_widths))
        
        return "\n".join(table_lines)
    
    def _calculate_column_widths(self, headers: List[str], rows: List[List[str]]) -> List[int]:
        """Calculate optimal column widths."""
        if not headers:
            return []
        
        # Start with header widths
        col_widths = [len(header) for header in headers]
        
        # Check row data
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Ensure minimum widths and respect max_width
        total_width = sum(col_widths) + len(col_widths) * 3  # Account for separators
        
        if total_width > self.max_width:
            # Proportionally reduce widths
            reduction_factor = self.max_width / total_width
            col_widths = [max(8, int(w * reduction_factor)) for w in col_widths]
        
        return col_widths
    
    def _format_title(self, title: str) -> str:
        """Format table title."""
        return f"{'=' * len(title)}\n{title}\n{'=' * len(title)}"
    
    def _format_header(self, headers: List[str], col_widths: List[int]) -> str:
        """Format table header."""
        formatted_headers = []
        for header, width in zip(headers, col_widths):
            formatted_headers.append(header.ljust(width))
        return " | ".join(formatted_headers)
    
    def _format_separator(self, col_widths: List[int]) -> str:
        """Format separator line."""
        separators = ["-" * width for width in col_widths]
        return "-+-".join(separators)
    
    def _format_row(self, row: List[str], col_widths: List[int]) -> str:
        """Format a single row."""
        formatted_cells = []
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            cell_str = str(cell)
            if len(cell_str) > width:
                cell_str = cell_str[:width-3] + "..."
            formatted_cells.append(cell_str.ljust(width))
        return " | ".join(formatted_cells)
    
    def _format_empty_table(self, title: Optional[str] = None) -> str:
        """Format empty table message."""
        message = "No data to display"
        if title:
            return f"{title}\n{'-' * len(title)}\n{message}"
        return message


class TradeTableFormatter(TableFormatter):
    """Specialized formatter for trade data."""
    
    def format_trades(self, trades: List[TradeCandidate], title: str = "Top Trade Recommendations") -> str:
        """Format trade candidates into a table."""
        if not trades:
            return self._format_empty_table(title)
        
        headers = ["Ticker", "Strategy", "Legs", "Thesis", "POP"]
        rows = []
        
        for trade in trades:
            row = [
                trade.underlying,
                self._format_strategy_type(trade.strategy_type),
                self._format_legs(trade.strategy.legs),
                trade.thesis or "N/A",
                self._format_percentage(trade.probability_of_profit)
            ]
            rows.append(row)
        
        return self.format_table(headers, rows, title)
    
    def format_detailed_trades(self, trades: List[TradeCandidate], title: str = "Detailed Trade Analysis") -> str:
        """Format trades with detailed information."""
        if not trades:
            return self._format_empty_table(title)
        
        headers = [
            "Ticker", "Strategy", "POP", "Premium", "Max Loss", 
            "C/ML", "Delta", "Vega", "DTE", "Sector"
        ]
        rows = []
        
        for trade in trades:
            row = [
                trade.underlying,
                self._format_strategy_type(trade.strategy_type),
                self._format_percentage(trade.probability_of_profit),
                self._format_currency(trade.net_premium),
                self._format_currency(trade.max_loss),
                self._format_ratio(trade.strategy.credit_to_max_loss_ratio),
                self._format_decimal(trade.strategy.net_delta, 2),
                self._format_decimal(trade.strategy.net_vega, 2),
                str(trade.days_to_expiration) if trade.days_to_expiration else "N/A",
                trade.sector or "N/A"
            ]
            rows.append(row)
        
        return self.format_table(headers, rows, title)
    
    def _format_strategy_type(self, strategy_type) -> str:
        """Format strategy type for display."""
        return strategy_type.value.replace("_", " ").title()
    
    def _format_legs(self, legs) -> str:
        """Format strategy legs for display."""
        if not legs:
            return "N/A"
        
        leg_descriptions = []
        for leg in legs:
            direction = "L" if leg.direction.value == "long" else "S"
            strike = f"{leg.option.strike:.0f}" if leg.option.strike else "N/A"
            opt_type = "C" if leg.option.option_type.value == "call" else "P"
            leg_descriptions.append(f"{direction}{strike}{opt_type}")
        
        return "/".join(leg_descriptions)
    
    def _format_percentage(self, value: Optional[float]) -> str:
        """Format percentage value."""
        if value is None:
            return "N/A"
        return f"{value:.1%}"
    
    def _format_currency(self, value: Optional[Decimal]) -> str:
        """Format currency value."""
        if value is None:
            return "N/A"
        return f"${value:.0f}"
    
    def _format_ratio(self, value: Optional[float]) -> str:
        """Format ratio value."""
        if value is None:
            return "N/A"
        return f"{value:.2f}"
    
    def _format_decimal(self, value: Optional[float], decimals: int = 2) -> str:
        """Format decimal value."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"


class PortfolioTableFormatter(TableFormatter):
    """Specialized formatter for portfolio data."""
    
    def format_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> str:
        """Format portfolio summary."""
        headers = ["Metric", "Value"]
        rows = [
            ["Total Capital", self._format_currency(portfolio_data.get("total_capital"))],
            ["Capital Utilization", self._format_percentage(portfolio_data.get("capital_utilization"))],
            ["Total Delta", self._format_decimal(portfolio_data.get("total_delta"))],
            ["Total Vega", self._format_decimal(portfolio_data.get("total_vega"))],
            ["Active Trades", str(portfolio_data.get("active_trades", 0))]
        ]
        
        return self.format_table(headers, rows, "Portfolio Summary")
    
    def format_sector_distribution(self, sector_data: Dict[str, int]) -> str:
        """Format sector distribution."""
        if not sector_data:
            return self._format_empty_table("Sector Distribution")
        
        headers = ["Sector", "Count"]
        rows = [[sector, str(count)] for sector, count in sorted(sector_data.items())]
        
        return self.format_table(headers, rows, "Sector Distribution")
    
    def _format_currency(self, value: Optional[Decimal]) -> str:
        """Format currency value."""
        if value is None:
            return "N/A"
        return f"${value:,.0f}"
    
    def _format_percentage(self, value: Optional[float]) -> str:
        """Format percentage value."""
        if value is None:
            return "N/A"
        return f"{value:.1%}"
    
    def _format_decimal(self, value: Optional[float], decimals: int = 2) -> str:
        """Format decimal value."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"


class StatsTableFormatter(TableFormatter):
    """Specialized formatter for statistics and performance data."""
    
    def format_generation_stats(self, stats: Dict[str, Any]) -> str:
        """Format trade generation statistics."""
        headers = ["Metric", "Value"]
        rows = [
            ["Total Candidates", str(stats.get("total_candidates", 0))],
            ["Scored Candidates", str(stats.get("scored_candidates", 0))],
            ["Portfolio Filtered", str(stats.get("portfolio_filtered", 0))],
            ["Selected Trades", str(stats.get("selected_trades", 0))],
            ["Generation Time", f"{stats.get('generation_time', 0):.2f}s"]
        ]
        
        return self.format_table(headers, rows, "Generation Statistics")
    
    def format_scan_stats(self, stats: Dict[str, Any]) -> str:
        """Format market scan statistics."""
        headers = ["Metric", "Value"]
        rows = [
            ["Symbols Scanned", str(stats.get("symbols_scanned", 0))],
            ["Strategies Generated", str(stats.get("strategies_generated", 0))],
            ["Candidates Found", str(stats.get("candidates_found", 0))],
            ["Scan Duration", f"{stats.get('scan_duration', 0):.2f}s"]
        ]
        
        return self.format_table(headers, rows, "Scan Statistics")