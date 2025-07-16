"""
Table formatting utilities for the Options Trading Engine.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
import sys


class TableStyle(Enum):
    """Table formatting styles."""
    SIMPLE = "simple"
    GRID = "grid"
    FANCY = "fancy"
    MINIMAL = "minimal"
    MARKDOWN = "markdown"


class TableAlignment(Enum):
    """Column alignment options."""
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class TableFormatter:
    """
    Advanced table formatter with multiple styles and alignment options.
    
    Supports various table styles, column alignment, and formatting options
    for displaying tabular data in the terminal.
    """
    
    def __init__(self, 
                 style: TableStyle = TableStyle.GRID,
                 max_width: Optional[int] = None,
                 truncate_symbol: str = "…"):
        self.style = style
        self.max_width = max_width or self._get_terminal_width()
        self.truncate_symbol = truncate_symbol
        
        # Style definitions
        self.styles = {
            TableStyle.SIMPLE: {
                'horizontal': '-',
                'vertical': '|',
                'corner': '+',
                'header_separator': True,
                'border': False
            },
            TableStyle.GRID: {
                'horizontal': '─',
                'vertical': '│',
                'corner': '┼',
                'top_left': '┌',
                'top_right': '┐',
                'bottom_left': '└',
                'bottom_right': '┘',
                'header_separator': True,
                'border': True
            },
            TableStyle.FANCY: {
                'horizontal': '═',
                'vertical': '║',
                'corner': '╬',
                'top_left': '╔',
                'top_right': '╗',
                'bottom_left': '╚',
                'bottom_right': '╝',
                'header_separator': True,
                'border': True
            },
            TableStyle.MINIMAL: {
                'horizontal': '',
                'vertical': '  ',
                'corner': '',
                'header_separator': False,
                'border': False
            },
            TableStyle.MARKDOWN: {
                'horizontal': '-',
                'vertical': '|',
                'corner': '|',
                'header_separator': True,
                'border': True
            }
        }
    
    def _get_terminal_width(self) -> int:
        """Get terminal width."""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except:
            return 80
    
    def print_table(self, 
                   headers: List[str], 
                   rows: List[List[Union[str, int, float]]], 
                   title: Optional[str] = None,
                   alignments: Optional[List[TableAlignment]] = None,
                   column_widths: Optional[List[int]] = None,
                   show_index: bool = False,
                   index_label: str = "#"):
        """
        Print formatted table to console.
        
        Args:
            headers: Column headers
            rows: Table rows
            title: Optional table title
            alignments: Column alignments
            column_widths: Fixed column widths
            show_index: Show row index column
            index_label: Label for index column
        """
        if not headers or not rows:
            print("No data to display")
            return
        
        # Add index column if requested
        if show_index:
            headers = [index_label] + headers
            indexed_rows = []
            for i, row in enumerate(rows):
                indexed_rows.append([str(i + 1)] + [str(cell) for cell in row])
            rows = indexed_rows
        else:
            rows = [[str(cell) for cell in row] for row in rows]
        
        # Calculate column widths
        if column_widths is None:
            column_widths = self._calculate_column_widths(headers, rows)
        
        # Adjust widths if total exceeds terminal width
        column_widths = self._adjust_column_widths(column_widths)
        
        # Set default alignments
        if alignments is None:
            alignments = [TableAlignment.LEFT] * len(headers)
        
        # Truncate content to fit column widths
        headers = [self._truncate_text(header, column_widths[i]) for i, header in enumerate(headers)]
        rows = [[self._truncate_text(cell, column_widths[j]) for j, cell in enumerate(row)] for row in rows]
        
        # Print title if provided
        if title:
            self._print_title(title, sum(column_widths) + len(headers) - 1)
        
        # Print table based on style
        if self.style == TableStyle.MARKDOWN:
            self._print_markdown_table(headers, rows, alignments, column_widths)
        else:
            self._print_standard_table(headers, rows, alignments, column_widths)
    
    def _calculate_column_widths(self, headers: List[str], rows: List[List[str]]) -> List[int]:
        """Calculate optimal column widths."""
        if not headers:
            return []
        
        # Initialize with header widths
        widths = [len(header) for header in headers]
        
        # Update with row content widths
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        
        return widths
    
    def _adjust_column_widths(self, widths: List[int]) -> List[int]:
        """Adjust column widths to fit terminal width."""
        if not widths:
            return []
        
        # Calculate total width including separators
        total_width = sum(widths) + len(widths) - 1
        style_info = self.styles[self.style]
        
        # Add border width if applicable
        if style_info.get('border', False):
            total_width += 2
        
        # If total width exceeds terminal width, proportionally reduce
        if total_width > self.max_width:
            # Reserve space for separators and borders
            available_width = self.max_width - (len(widths) - 1)
            if style_info.get('border', False):
                available_width -= 2
            
            # Proportionally reduce column widths
            total_content_width = sum(widths)
            if total_content_width > 0:
                ratio = available_width / total_content_width
                widths = [max(3, int(width * ratio)) for width in widths]
        
        return widths
    
    def _truncate_text(self, text: str, width: int) -> str:
        """Truncate text to fit width."""
        if len(text) <= width:
            return text
        
        if width <= len(self.truncate_symbol):
            return text[:width]
        
        return text[:width - len(self.truncate_symbol)] + self.truncate_symbol
    
    def _print_title(self, title: str, width: int):
        """Print table title."""
        if len(title) > width:
            title = self._truncate_text(title, width)
        
        centered_title = title.center(width)
        print(centered_title)
        print('=' * width)
    
    def _print_standard_table(self, 
                            headers: List[str], 
                            rows: List[List[str]], 
                            alignments: List[TableAlignment], 
                            column_widths: List[int]):
        """Print table in standard format."""
        style_info = self.styles[self.style]
        
        # Print top border
        if style_info.get('border', False):
            self._print_border_line(column_widths, 'top')
        
        # Print headers
        self._print_row(headers, alignments, column_widths, is_header=True)
        
        # Print header separator
        if style_info.get('header_separator', False):
            self._print_separator_line(column_widths)
        
        # Print rows
        for row in rows:
            # Pad row to match header length
            padded_row = row + [''] * (len(headers) - len(row))
            self._print_row(padded_row, alignments, column_widths)
        
        # Print bottom border
        if style_info.get('border', False):
            self._print_border_line(column_widths, 'bottom')
    
    def _print_markdown_table(self, 
                            headers: List[str], 
                            rows: List[List[str]], 
                            alignments: List[TableAlignment], 
                            column_widths: List[int]):
        """Print table in markdown format."""
        # Print headers
        header_row = '| ' + ' | '.join(headers) + ' |'
        print(header_row)
        
        # Print separator with alignment indicators
        separator_parts = []
        for i, alignment in enumerate(alignments):
            if alignment == TableAlignment.LEFT:
                separator_parts.append(':' + '-' * (column_widths[i] - 1))
            elif alignment == TableAlignment.RIGHT:
                separator_parts.append('-' * (column_widths[i] - 1) + ':')
            elif alignment == TableAlignment.CENTER:
                separator_parts.append(':' + '-' * (column_widths[i] - 2) + ':')
            else:
                separator_parts.append('-' * column_widths[i])
        
        separator_row = '| ' + ' | '.join(separator_parts) + ' |'
        print(separator_row)
        
        # Print rows
        for row in rows:
            padded_row = row + [''] * (len(headers) - len(row))
            row_str = '| ' + ' | '.join(padded_row) + ' |'
            print(row_str)
    
    def _print_row(self, 
                  row: List[str], 
                  alignments: List[TableAlignment], 
                  column_widths: List[int], 
                  is_header: bool = False):
        """Print a single table row."""
        style_info = self.styles[self.style]
        
        if self.style == TableStyle.MINIMAL:
            # Minimal style - just space-separated values
            aligned_cells = []
            for i, (cell, alignment, width) in enumerate(zip(row, alignments, column_widths)):
                aligned_cells.append(self._align_text(cell, alignment, width))
            print('  '.join(aligned_cells))
        else:
            # Standard style with borders
            vertical = style_info['vertical']
            
            # Align cells
            aligned_cells = []
            for i, (cell, alignment, width) in enumerate(zip(row, alignments, column_widths)):
                aligned_cells.append(self._align_text(cell, alignment, width))
            
            # Build row
            if style_info.get('border', False):
                row_str = vertical + vertical.join(aligned_cells) + vertical
            else:
                row_str = vertical.join(aligned_cells)
            
            print(row_str)
    
    def _print_border_line(self, column_widths: List[int], position: str):
        """Print border line (top or bottom)."""
        style_info = self.styles[self.style]
        horizontal = style_info['horizontal']
        
        if position == 'top':
            left_corner = style_info.get('top_left', style_info.get('corner', '+'))
            right_corner = style_info.get('top_right', style_info.get('corner', '+'))
        else:  # bottom
            left_corner = style_info.get('bottom_left', style_info.get('corner', '+'))
            right_corner = style_info.get('bottom_right', style_info.get('corner', '+'))
        
        corner = style_info.get('corner', '+')
        
        # Build line
        line_parts = [horizontal * width for width in column_widths]
        line = left_corner + corner.join(line_parts) + right_corner
        print(line)
    
    def _print_separator_line(self, column_widths: List[int]):
        """Print separator line between header and body."""
        style_info = self.styles[self.style]
        horizontal = style_info['horizontal']
        corner = style_info.get('corner', '+')
        
        if style_info.get('border', False):
            vertical = style_info['vertical']
            line_parts = [horizontal * width for width in column_widths]
            line = vertical + corner.join(line_parts) + vertical
        else:
            line_parts = [horizontal * width for width in column_widths]
            line = corner.join(line_parts)
        
        print(line)
    
    def _align_text(self, text: str, alignment: TableAlignment, width: int) -> str:
        """Align text within specified width."""
        if alignment == TableAlignment.LEFT:
            return text.ljust(width)
        elif alignment == TableAlignment.RIGHT:
            return text.rjust(width)
        elif alignment == TableAlignment.CENTER:
            return text.center(width)
        else:
            return text.ljust(width)
    
    def format_table_string(self, 
                           headers: List[str], 
                           rows: List[List[Union[str, int, float]]], 
                           **kwargs) -> str:
        """
        Format table as string instead of printing.
        
        Args:
            headers: Column headers
            rows: Table rows
            **kwargs: Additional formatting options
            
        Returns:
            Formatted table as string
        """
        # Capture print output
        import io
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            self.print_table(headers, rows, **kwargs)
        
        return output.getvalue()
    
    def print_summary_table(self, data: Dict[str, Any], title: str = "Summary"):
        """Print summary data as a two-column table."""
        headers = ["Metric", "Value"]
        rows = []
        
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                if abs(value) < 1:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            elif isinstance(value, bool):
                formatted_value = "✓" if value else "✗"
            else:
                formatted_value = str(value)
            
            rows.append([formatted_key, formatted_value])
        
        self.print_table(headers, rows, title=title)
    
    def print_comparison_table(self, 
                             data: List[Dict[str, Any]], 
                             title: str = "Comparison"):
        """Print comparison data as a table."""
        if not data:
            print("No data to compare")
            return
        
        # Extract headers from first item
        headers = list(data[0].keys())
        
        # Create rows
        rows = []
        for item in data:
            row = [str(item.get(header, '')) for header in headers]
            rows.append(row)
        
        # Format headers
        formatted_headers = [header.replace('_', ' ').title() for header in headers]
        
        self.print_table(formatted_headers, rows, title=title)
    
    def print_financial_table(self, 
                            data: List[Dict[str, Any]], 
                            currency_columns: List[str] = None,
                            percentage_columns: List[str] = None,
                            title: str = "Financial Data"):
        """Print financial data with proper formatting."""
        if not data:
            print("No financial data to display")
            return
        
        currency_columns = currency_columns or []
        percentage_columns = percentage_columns or []
        
        headers = list(data[0].keys())
        formatted_headers = [header.replace('_', ' ').title() for header in headers]
        
        rows = []
        for item in data:
            row = []
            for header in headers:
                value = item.get(header, '')
                
                if header in currency_columns and isinstance(value, (int, float)):
                    formatted_value = f"${value:,.2f}"
                elif header in percentage_columns and isinstance(value, (int, float)):
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = str(value)
                
                row.append(formatted_value)
            rows.append(row)
        
        # Set right alignment for numeric columns
        alignments = []
        for header in headers:
            if header in currency_columns or header in percentage_columns:
                alignments.append(TableAlignment.RIGHT)
            else:
                alignments.append(TableAlignment.LEFT)
        
        self.print_table(formatted_headers, rows, title=title, alignments=alignments)
    
    def print_status_table(self, 
                          statuses: Dict[str, Dict[str, Any]], 
                          title: str = "Status"):
        """Print status information as a table."""
        headers = ["Component", "Status", "Details"]
        rows = []
        
        for component, status_info in statuses.items():
            status = status_info.get('status', 'unknown')
            details = status_info.get('details', '')
            
            # Format status with icons
            if status.lower() in ['healthy', 'ok', 'success']:
                status_formatted = f"✓ {status}"
            elif status.lower() in ['error', 'failed', 'unhealthy']:
                status_formatted = f"✗ {status}"
            elif status.lower() in ['warning', 'caution']:
                status_formatted = f"⚠ {status}"
            else:
                status_formatted = f"• {status}"
            
            rows.append([component, status_formatted, details])
        
        self.print_table(headers, rows, title=title)
    
    def print_performance_table(self, 
                              metrics: Dict[str, float], 
                              title: str = "Performance Metrics"):
        """Print performance metrics with proper formatting."""
        headers = ["Metric", "Value", "Grade"]
        rows = []
        
        for metric, value in metrics.items():
            formatted_metric = metric.replace('_', ' ').title()
            
            # Format value
            if 'ratio' in metric.lower() or 'percentage' in metric.lower():
                formatted_value = f"{value:.2%}"
            elif 'time' in metric.lower():
                formatted_value = f"{value:.2f}s"
            else:
                formatted_value = f"{value:.2f}"
            
            # Assign grade (simple logic)
            if value >= 0.8:
                grade = "A"
            elif value >= 0.6:
                grade = "B"
            elif value >= 0.4:
                grade = "C"
            elif value >= 0.2:
                grade = "D"
            else:
                grade = "F"
            
            rows.append([formatted_metric, formatted_value, grade])
        
        alignments = [TableAlignment.LEFT, TableAlignment.RIGHT, TableAlignment.CENTER]
        self.print_table(headers, rows, title=title, alignments=alignments)