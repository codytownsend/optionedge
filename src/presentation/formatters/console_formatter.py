"""
Console formatting utilities for the Options Trading Engine.
"""

import sys
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class Color(Enum):
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ConsoleFormatter:
    """
    Console output formatter with color support and structured output.
    
    Provides methods for formatting various types of output including
    success/error messages, progress indicators, and structured data.
    """
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and self._supports_color()
        self.terminal_width = self._get_terminal_width()
    
    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        return (
            hasattr(sys.stdout, 'isatty') and 
            sys.stdout.isatty() and 
            sys.platform != 'win32'
        )
    
    def _get_terminal_width(self) -> int:
        """Get terminal width for formatting."""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except:
            return 80  # Default width
    
    def _colorize(self, text: str, color: Color) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{color.value}{text}{Color.RESET.value}"
    
    def print_success(self, message: str):
        """Print success message in green."""
        colored_message = self._colorize(f"✓ {message}", Color.GREEN)
        print(colored_message)
    
    def print_error(self, message: str):
        """Print error message in red."""
        colored_message = self._colorize(f"✗ {message}", Color.RED)
        print(colored_message, file=sys.stderr)
    
    def print_warning(self, message: str):
        """Print warning message in yellow."""
        colored_message = self._colorize(f"⚠ {message}", Color.YELLOW)
        print(colored_message)
    
    def print_info(self, message: str):
        """Print info message in blue."""
        colored_message = self._colorize(f"ℹ {message}", Color.BLUE)
        print(colored_message)
    
    def print_debug(self, message: str):
        """Print debug message in dim."""
        colored_message = self._colorize(f"🔍 {message}", Color.DIM)
        print(colored_message)
    
    def print_header(self, title: str, level: int = 1):
        """Print formatted header."""
        if level == 1:
            # Main header
            separator = "=" * len(title)
            colored_title = self._colorize(title, Color.BOLD)
            print(f"\n{colored_title}")
            print(separator)
        elif level == 2:
            # Sub header
            colored_title = self._colorize(title, Color.UNDERLINE)
            print(f"\n{colored_title}")
        else:
            # Simple header
            print(f"\n{title}")
    
    def print_separator(self, char: str = "-", length: Optional[int] = None):
        """Print separator line."""
        if length is None:
            length = min(self.terminal_width, 80)
        separator = char * length
        print(self._colorize(separator, Color.DIM))
    
    def print_progress(self, current: int, total: int, message: str = ""):
        """Print progress indicator."""
        if total == 0:
            percentage = 0
        else:
            percentage = (current / total) * 100
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        
        # Format message
        progress_text = f"[{bar}] {percentage:.1f}% ({current}/{total})"
        if message:
            progress_text += f" {message}"
        
        colored_progress = self._colorize(progress_text, Color.CYAN)
        print(f"\r{colored_progress}", end="", flush=True)
    
    def print_key_value(self, key: str, value: Any, indent: int = 0):
        """Print key-value pair with formatting."""
        indentation = "  " * indent
        colored_key = self._colorize(f"{key}:", Color.BOLD)
        print(f"{indentation}{colored_key} {value}")
    
    def print_list(self, items: List[str], bullet: str = "•", indent: int = 0):
        """Print formatted list."""
        indentation = "  " * indent
        colored_bullet = self._colorize(bullet, Color.CYAN)
        
        for item in items:
            print(f"{indentation}{colored_bullet} {item}")
    
    def print_table_simple(self, headers: List[str], rows: List[List[str]]):
        """Print simple table without borders."""
        if not headers or not rows:
            return
        
        # Calculate column widths
        all_rows = [headers] + rows
        col_widths = []
        
        for col in range(len(headers)):
            max_width = max(len(str(row[col])) for row in all_rows if col < len(row))
            col_widths.append(max_width)
        
        # Print header
        header_row = " | ".join(
            self._colorize(headers[i].ljust(col_widths[i]), Color.BOLD) 
            for i in range(len(headers))
        )
        print(header_row)
        
        # Print separator
        separator = "-+-".join("-" * width for width in col_widths)
        print(self._colorize(separator, Color.DIM))
        
        # Print rows
        for row in rows:
            row_str = " | ".join(
                str(row[i]).ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i])
                for i in range(len(headers))
            )
            print(row_str)
    
    def print_json(self, data: Dict[str, Any], indent: int = 2):
        """Print JSON data with syntax highlighting."""
        import json
        
        try:
            json_str = json.dumps(data, indent=indent, default=str)
            
            if self.use_colors:
                # Simple syntax highlighting
                json_str = json_str.replace('"', self._colorize('"', Color.GREEN))
                json_str = json_str.replace(':', self._colorize(':', Color.CYAN))
                json_str = json_str.replace('{', self._colorize('{', Color.YELLOW))
                json_str = json_str.replace('}', self._colorize('}', Color.YELLOW))
                json_str = json_str.replace('[', self._colorize('[', Color.YELLOW))
                json_str = json_str.replace(']', self._colorize(']', Color.YELLOW))
            
            print(json_str)
        except Exception as e:
            self.print_error(f"Failed to format JSON: {str(e)}")
            print(str(data))
    
    def print_metrics(self, metrics: Dict[str, Any], title: str = "Metrics"):
        """Print metrics in a formatted way."""
        self.print_header(title, level=2)
        
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                if abs(value) < 1:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            
            self.print_key_value(formatted_key, formatted_value)
    
    def print_status_indicator(self, status: str, message: str = ""):
        """Print status with appropriate color."""
        status_lower = status.lower()
        
        if status_lower in ['success', 'ok', 'healthy', 'passed']:
            icon = "✓"
            color = Color.GREEN
        elif status_lower in ['error', 'failed', 'unhealthy']:
            icon = "✗"
            color = Color.RED
        elif status_lower in ['warning', 'caution']:
            icon = "⚠"
            color = Color.YELLOW
        elif status_lower in ['info', 'running', 'processing']:
            icon = "ℹ"
            color = Color.BLUE
        else:
            icon = "•"
            color = Color.WHITE
        
        status_text = f"{icon} {status.upper()}"
        if message:
            status_text += f": {message}"
        
        colored_status = self._colorize(status_text, color)
        print(colored_status)
    
    def print_banner(self, text: str, char: str = "=", width: Optional[int] = None):
        """Print banner with text centered."""
        if width is None:
            width = min(self.terminal_width, 80)
        
        # Create banner
        text_length = len(text)
        if text_length >= width - 4:
            # Text too long, just print with minimal padding
            banner = f"{char} {text} {char}"
        else:
            padding = (width - text_length - 2) // 2
            banner = char * padding + f" {text} " + char * padding
            
            # Adjust if odd width
            if len(banner) < width:
                banner += char
        
        colored_banner = self._colorize(banner, Color.BOLD)
        print(f"\n{colored_banner}")
    
    def print_countdown(self, seconds: int, message: str = ""):
        """Print countdown timer."""
        import time
        
        for i in range(seconds, 0, -1):
            countdown_text = f"⏰ {i}s"
            if message:
                countdown_text += f" {message}"
            
            colored_countdown = self._colorize(countdown_text, Color.YELLOW)
            print(f"\r{colored_countdown}", end="", flush=True)
            time.sleep(1)
        
        print()  # New line after countdown
    
    def print_box(self, content: List[str], title: str = ""):
        """Print content in a box."""
        if not content:
            return
        
        # Calculate box width
        max_content_width = max(len(line) for line in content)
        title_width = len(title) if title else 0
        box_width = max(max_content_width + 4, title_width + 4, 20)
        
        # Top border
        if title:
            title_padding = (box_width - len(title) - 2) // 2
            top_border = "┌" + "─" * title_padding + f" {title} " + "─" * (box_width - title_padding - len(title) - 3) + "┐"
        else:
            top_border = "┌" + "─" * (box_width - 2) + "┐"
        
        print(self._colorize(top_border, Color.CYAN))
        
        # Content
        for line in content:
            padded_line = f"│ {line.ljust(box_width - 4)} │"
            print(self._colorize(padded_line, Color.CYAN))
        
        # Bottom border
        bottom_border = "└" + "─" * (box_width - 2) + "┘"
        print(self._colorize(bottom_border, Color.CYAN))
    
    def clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_timestamp(self, timestamp: datetime = None) -> str:
        """Format timestamp for display."""
        if timestamp is None:
            timestamp = datetime.now()
        
        return self._colorize(
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            Color.DIM
        )
    
    def format_currency(self, amount: float, currency: str = "$") -> str:
        """Format currency amount."""
        if amount >= 0:
            color = Color.GREEN
        else:
            color = Color.RED
        
        formatted = f"{currency}{amount:,.2f}"
        return self._colorize(formatted, color)
    
    def format_percentage(self, percentage: float, show_sign: bool = True) -> str:
        """Format percentage with color."""
        if percentage >= 0:
            color = Color.GREEN
            sign = "+" if show_sign else ""
        else:
            color = Color.RED
            sign = ""
        
        formatted = f"{sign}{percentage:.2f}%"
        return self._colorize(formatted, color)
    
    def input_with_prompt(self, prompt: str, color: Color = Color.CYAN) -> str:
        """Get user input with colored prompt."""
        colored_prompt = self._colorize(f"{prompt}: ", color)
        return input(colored_prompt)
    
    def confirm(self, message: str) -> bool:
        """Ask for yes/no confirmation."""
        prompt = f"{message} (y/n)"
        colored_prompt = self._colorize(prompt, Color.YELLOW)
        
        while True:
            response = input(f"{colored_prompt}: ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                self.print_error("Please enter 'y' or 'n'")
    
    def print_loading(self, message: str = "Loading..."):
        """Print loading indicator."""
        loading_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        import time
        
        for char in loading_chars:
            loading_text = f"{char} {message}"
            colored_loading = self._colorize(loading_text, Color.CYAN)
            print(f"\r{colored_loading}", end="", flush=True)
            time.sleep(0.1)
    
    def print_divider(self, title: str = ""):
        """Print section divider."""
        if title:
            self.print_separator()
            centered_title = title.center(min(self.terminal_width, 80))
            print(self._colorize(centered_title, Color.BOLD))
            self.print_separator()
        else:
            self.print_separator()
    
    def print_summary(self, data: Dict[str, Any]):
        """Print summary information."""
        self.print_header("Summary", level=2)
        
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, bool):
                status = "✓" if value else "✗"
                color = Color.GREEN if value else Color.RED
                formatted_value = self._colorize(status, color)
            elif isinstance(value, (int, float)):
                formatted_value = self.format_currency(value) if 'amount' in key.lower() or 'price' in key.lower() else str(value)
            else:
                formatted_value = str(value)
            
            self.print_key_value(formatted_key, formatted_value)