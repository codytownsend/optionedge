"""Console-specific formatting utilities."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

from ...data.models.trades import TradeCandidate
from .table_formatter import TradeTableFormatter, PortfolioTableFormatter, StatsTableFormatter


class ConsoleColors:
    """ANSI color codes for console output."""
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'


class ConsoleFormatter:
    """Main console formatter for the options engine."""
    
    def __init__(self, use_colors: bool = True, max_width: int = 120):
        self.use_colors = use_colors and sys.stdout.isatty()
        self.max_width = max_width
        
        # Initialize specialized formatters
        self.trade_formatter = TradeTableFormatter(max_width)
        self.portfolio_formatter = PortfolioTableFormatter(max_width)
        self.stats_formatter = StatsTableFormatter(max_width)
    
    def format_trade_results(
        self, 
        trades: List[TradeCandidate], 
        portfolio_impact: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """Format complete trade generation results."""
        output_sections = []
        
        # Header
        output_sections.append(self._format_header())
        
        # Main trade table
        trade_table = self.trade_formatter.format_trades(trades)
        output_sections.append(self._colorize(trade_table, ConsoleColors.BRIGHT_WHITE))
        
        # Portfolio impact
        if portfolio_impact:
            portfolio_summary = self.portfolio_formatter.format_portfolio_summary(portfolio_impact)
            output_sections.append(self._colorize(portfolio_summary, ConsoleColors.CYAN))
        
        # Sector distribution
        if portfolio_impact and "sector_distribution" in portfolio_impact:
            sector_table = self.portfolio_formatter.format_sector_distribution(
                portfolio_impact["sector_distribution"]
            )
            output_sections.append(self._colorize(sector_table, ConsoleColors.BLUE))
        
        # Generation statistics
        if stats:
            stats_table = self.stats_formatter.format_generation_stats(stats)
            output_sections.append(self._colorize(stats_table, ConsoleColors.DIM))
        
        # Footer
        output_sections.append(self._format_footer())
        
        return "\n\n".join(output_sections)
    
    def format_detailed_results(
        self, 
        trades: List[TradeCandidate], 
        portfolio_impact: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """Format detailed trade analysis results."""
        output_sections = []
        
        # Header
        output_sections.append(self._format_header())
        
        # Detailed trade table
        detailed_table = self.trade_formatter.format_detailed_trades(trades)
        output_sections.append(self._colorize(detailed_table, ConsoleColors.BRIGHT_WHITE))
        
        # Portfolio impact
        if portfolio_impact:
            portfolio_summary = self.portfolio_formatter.format_portfolio_summary(portfolio_impact)
            output_sections.append(self._colorize(portfolio_summary, ConsoleColors.CYAN))
        
        # Statistics
        if stats:
            stats_table = self.stats_formatter.format_generation_stats(stats)
            output_sections.append(self._colorize(stats_table, ConsoleColors.DIM))
        
        return "\n\n".join(output_sections)
    
    def format_error_message(self, error: str, details: Optional[str] = None) -> str:
        """Format error message for console output."""
        error_lines = [
            self._colorize("ERROR", ConsoleColors.BRIGHT_RED + ConsoleColors.BOLD),
            self._colorize(error, ConsoleColors.RED)
        ]
        
        if details:
            error_lines.append(self._colorize(f"Details: {details}", ConsoleColors.DIM))
        
        return "\n".join(error_lines)
    
    def format_warning_message(self, warning: str, details: Optional[str] = None) -> str:
        """Format warning message for console output."""
        warning_lines = [
            self._colorize("WARNING", ConsoleColors.BRIGHT_YELLOW + ConsoleColors.BOLD),
            self._colorize(warning, ConsoleColors.YELLOW)
        ]
        
        if details:
            warning_lines.append(self._colorize(f"Details: {details}", ConsoleColors.DIM))
        
        return "\n".join(warning_lines)
    
    def format_info_message(self, message: str) -> str:
        """Format info message for console output."""
        return self._colorize(message, ConsoleColors.BRIGHT_BLUE)
    
    def format_success_message(self, message: str) -> str:
        """Format success message for console output."""
        return self._colorize(message, ConsoleColors.BRIGHT_GREEN)
    
    def format_progress_message(self, message: str, percentage: Optional[float] = None) -> str:
        """Format progress message for console output."""
        if percentage is not None:
            message = f"{message} ({percentage:.1%})"
        return self._colorize(message, ConsoleColors.CYAN)
    
    def format_loading_message(self, message: str) -> str:
        """Format loading message for console output."""
        return self._colorize(f"⏳ {message}", ConsoleColors.YELLOW)
    
    def format_completion_message(self, message: str) -> str:
        """Format completion message for console output."""
        return self._colorize(f"✅ {message}", ConsoleColors.GREEN)
    
    def _format_header(self) -> str:
        """Format the application header."""
        header_lines = [
            "Options Trading Engine",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * self.max_width
        ]
        
        return self._colorize("\n".join(header_lines), ConsoleColors.BRIGHT_MAGENTA + ConsoleColors.BOLD)
    
    def _format_footer(self) -> str:
        """Format the application footer."""
        footer_lines = [
            "=" * self.max_width,
            "⚠️  This is for educational/simulation purposes only.",
            "⚠️  Always verify trade details before execution."
        ]
        
        return self._colorize("\n".join(footer_lines), ConsoleColors.DIM)
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{color}{text}{ConsoleColors.RESET}"
    
    def create_progress_bar(self, percentage: float, width: int = 40) -> str:
        """Create a progress bar."""
        filled_width = int(width * percentage)
        bar = "█" * filled_width + "░" * (width - filled_width)
        
        progress_text = f"[{bar}] {percentage:.1%}"
        return self._colorize(progress_text, ConsoleColors.CYAN)
    
    def format_spinner(self, message: str, frame: int = 0) -> str:
        """Format a spinner for long-running operations."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_char = spinner_chars[frame % len(spinner_chars)]
        
        return self._colorize(f"{spinner_char} {message}", ConsoleColors.YELLOW)


class ConsoleTableRenderer:
    """Renderer for console-specific table enhancements."""
    
    def __init__(self, formatter: ConsoleFormatter):
        self.formatter = formatter
    
    def render_trades_with_highlights(self, trades: List[TradeCandidate]) -> str:
        """Render trades with performance-based highlighting."""
        if not trades:
            return self.formatter.format_info_message("No trades found.")
        
        # Get base table
        base_table = self.formatter.trade_formatter.format_trades(trades)
        
        # Apply highlighting based on trade quality
        highlighted_lines = []
        
        for line in base_table.split('\n'):
            if any(trade.underlying in line for trade in trades):
                # Find the corresponding trade
                trade = next((t for t in trades if t.underlying in line), None)
                if trade:
                    color = self._get_trade_color(trade)
                    highlighted_lines.append(self.formatter._colorize(line, color))
                else:
                    highlighted_lines.append(line)
            else:
                highlighted_lines.append(line)
        
        return '\n'.join(highlighted_lines)
    
    def _get_trade_color(self, trade: TradeCandidate) -> str:
        """Get color based on trade quality."""
        if trade.model_score is None:
            return ConsoleColors.WHITE
        
        if trade.model_score >= 0.8:
            return ConsoleColors.BRIGHT_GREEN
        elif trade.model_score >= 0.6:
            return ConsoleColors.GREEN
        elif trade.model_score >= 0.4:
            return ConsoleColors.YELLOW
        else:
            return ConsoleColors.RED
    
    def render_portfolio_alerts(self, portfolio_impact: Dict[str, Any]) -> str:
        """Render portfolio alerts and warnings."""
        alerts = []
        
        # Check capital utilization
        capital_util = portfolio_impact.get("capital_utilization", 0)
        if capital_util > 0.8:
            alerts.append(self.formatter.format_warning_message(
                f"High capital utilization: {capital_util:.1%}"
            ))
        
        # Check delta exposure
        total_delta = portfolio_impact.get("total_delta", 0)
        if abs(total_delta) > 0.25:
            alerts.append(self.formatter.format_warning_message(
                f"High delta exposure: {total_delta:.2f}"
            ))
        
        # Check vega exposure
        total_vega = portfolio_impact.get("total_vega", 0)
        if total_vega < -0.04:
            alerts.append(self.formatter.format_warning_message(
                f"High vega exposure: {total_vega:.2f}"
            ))
        
        if alerts:
            return "\n".join(alerts)
        else:
            return self.formatter.format_success_message("Portfolio risk metrics within acceptable ranges.")


class ConsoleReportGenerator:
    """Generator for comprehensive console reports."""
    
    def __init__(self, formatter: ConsoleFormatter):
        self.formatter = formatter
        self.renderer = ConsoleTableRenderer(formatter)
    
    def generate_summary_report(
        self, 
        trades: List[TradeCandidate], 
        portfolio_impact: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """Generate a summary report."""
        return self.formatter.format_trade_results(trades, portfolio_impact, stats)
    
    def generate_detailed_report(
        self, 
        trades: List[TradeCandidate], 
        portfolio_impact: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """Generate a detailed report."""
        return self.formatter.format_detailed_results(trades, portfolio_impact, stats)
    
    def generate_risk_report(self, portfolio_impact: Dict[str, Any]) -> str:
        """Generate a risk-focused report."""
        sections = []
        
        # Portfolio alerts
        alerts = self.renderer.render_portfolio_alerts(portfolio_impact)
        sections.append(alerts)
        
        # Risk metrics table
        risk_table = self.formatter.portfolio_formatter.format_portfolio_summary(portfolio_impact)
        sections.append(self.formatter._colorize(risk_table, ConsoleColors.CYAN))
        
        return "\n\n".join(sections)