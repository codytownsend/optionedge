"""
Structured logging system for the options trading engine.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

import structlog
from structlog.stdlib import LoggerFactory
import colorama
from colorama import Fore, Back, Style


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(str, Enum):
    """Log category enumeration."""
    API = "api"
    CACHE = "cache"
    STRATEGY = "strategy"
    TRADE = "trade"
    RISK = "risk"
    DATA = "data"
    SYSTEM = "system"
    PERFORMANCE = "performance"


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON formatting
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
    """
    # Initialize colorama for cross-platform colored output
    colorama.init()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove default handlers
    
    # Console handler with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter() if not enable_json else JsonFormatter()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "options_engine.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_formatter = JsonFormatter() if enable_json else DetailedFormatter()
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        root_logger.addHandler(file_handler)
        
        # Separate error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Set root logger level
    root_logger.setLevel(logging.DEBUG)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        reset = Style.RESET_ALL
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Format message
        message = super().format(record)
        
        # Add color
        formatted = f"{Fore.WHITE}[{timestamp}]{reset} {log_color}{record.levelname:<8}{reset} {Fore.BLUE}{record.name:<20}{reset} {message}"
        
        return formatted


class DetailedFormatter(logging.Formatter):
    """Detailed file formatter."""
    
    def format(self, record):
        # Add timestamp
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Format base message
        message = super().format(record)
        
        # Add context information
        context_info = []
        
        # Add function and line info
        if hasattr(record, 'funcName') and hasattr(record, 'lineno'):
            context_info.append(f"func={record.funcName}:{record.lineno}")
        
        # Add thread info
        if hasattr(record, 'thread'):
            context_info.append(f"thread={record.thread}")
        
        context_str = " | ".join(context_info)
        context_suffix = f" | {context_str}" if context_str else ""
        
        return f"[{timestamp}] {record.levelname:<8} {record.name:<30} {message}{context_suffix}"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ContextLogger:
    """Logger with context management."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self._context = {}
    
    def add_context(self, **kwargs):
        """Add context to all subsequent log messages."""
        self._context.update(kwargs)
        return self
    
    def clear_context(self):
        """Clear all context."""
        self._context.clear()
        return self
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context."""
        log_data = {**self._context, **kwargs}
        getattr(self.logger, level)(message, **log_data)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context('critical', message, **kwargs)


class APILogger(ContextLogger):
    """Specialized logger for API operations."""
    
    def __init__(self, client_name: str):
        super().__init__(f"api.{client_name}")
        self.add_context(client=client_name, category=LogCategory.API)
    
    def log_request(self, method: str, url: str, **kwargs):
        """Log API request."""
        self.info(
            "API request",
            method=method,
            url=url,
            **kwargs
        )
    
    def log_response(self, method: str, url: str, status_code: int, duration_ms: float, **kwargs):
        """Log API response."""
        level = 'info' if 200 <= status_code < 300 else 'warning' if status_code < 500 else 'error'
        
        getattr(self, level)(
            "API response",
            method=method,
            url=url,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )
    
    def log_error(self, method: str, url: str, error: Exception, **kwargs):
        """Log API error."""
        self.error(
            "API error",
            method=method,
            url=url,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )


class TradeLogger(ContextLogger):
    """Specialized logger for trading operations."""
    
    def __init__(self):
        super().__init__("trade")
        self.add_context(category=LogCategory.TRADE)
    
    def log_strategy_generated(self, symbol: str, strategy_type: str, **kwargs):
        """Log strategy generation."""
        self.info(
            "Strategy generated",
            symbol=symbol,
            strategy_type=strategy_type,
            **kwargs
        )
    
    def log_trade_candidate(self, symbol: str, strategy_type: str, score: float, **kwargs):
        """Log trade candidate creation."""
        self.info(
            "Trade candidate created",
            symbol=symbol,
            strategy_type=strategy_type,
            score=score,
            **kwargs
        )
    
    def log_trade_filtered(self, symbol: str, reason: str, **kwargs):
        """Log trade filtering."""
        self.debug(
            "Trade filtered out",
            symbol=symbol,
            reason=reason,
            **kwargs
        )
    
    def log_trade_selected(self, symbol: str, strategy_type: str, rank: int, **kwargs):
        """Log trade selection."""
        self.info(
            "Trade selected for recommendation",
            symbol=symbol,
            strategy_type=strategy_type,
            rank=rank,
            **kwargs
        )


class PerformanceLogger(ContextLogger):
    """Logger for performance monitoring."""
    
    def __init__(self):
        super().__init__("performance")
        self.add_context(category=LogCategory.PERFORMANCE)
    
    def log_timing(self, operation: str, duration_ms: float, **kwargs):
        """Log operation timing."""
        level = 'debug' if duration_ms < 1000 else 'info' if duration_ms < 5000 else 'warning'
        
        getattr(self, level)(
            "Operation timing",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )
    
    def log_cache_stats(self, cache_name: str, hit_rate: float, **kwargs):
        """Log cache statistics."""
        self.info(
            "Cache statistics",
            cache_name=cache_name,
            hit_rate=round(hit_rate, 3),
            **kwargs
        )


# Convenience functions for getting specialized loggers
def get_api_logger(client_name: str) -> APILogger:
    """Get API logger for specific client."""
    return APILogger(client_name)


def get_trade_logger() -> TradeLogger:
    """Get trade operations logger."""
    return TradeLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get performance monitoring logger."""
    return PerformanceLogger()


def get_logger(name: str) -> ContextLogger:
    """Get context logger with specified name."""
    return ContextLogger(name)