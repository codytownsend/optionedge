"""
Centralized error handling and recovery system.
"""

import logging
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Type, Union
from functools import wraps
from collections import defaultdict, deque
import threading

from .exceptions import (
    OptionsEngineError, 
    ErrorSeverity, 
    ErrorCategory,
    APIError,
    RateLimitExceededError,
    ExternalServiceError
)


class ErrorRecoveryStrategy:
    """Strategy for recovering from specific error types."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        recoverable_errors: Optional[List[Type[Exception]]] = None
    ):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.recoverable_errors = recoverable_errors or [
            APIError, 
            RateLimitExceededError, 
            ExternalServiceError
        ]
    
    def is_recoverable(self, error: Exception) -> bool:
        """Check if error is recoverable."""
        if isinstance(error, OptionsEngineError):
            return error.recoverable
        
        return any(isinstance(error, err_type) for err_type in self.recoverable_errors)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_delay)


class ErrorTracker:
    """Track error patterns and frequencies."""
    
    def __init__(self, window_size: int = 100, time_window_hours: int = 24):
        self.window_size = window_size
        self.time_window = timedelta(hours=time_window_hours)
        self._errors: deque = deque(maxlen=window_size)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record an error occurrence."""
        with self._lock:
            error_record = {
                "timestamp": datetime.utcnow(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            }
            
            if isinstance(error, OptionsEngineError):
                error_record.update({
                    "error_id": error.error_id,
                    "severity": error.severity.value,
                    "category": error.category.value,
                    "recoverable": error.recoverable
                })
            
            self._errors.append(error_record)
            self._error_counts[type(error).__name__] += 1
    
    def get_error_frequency(self, error_type: str) -> float:
        """Get error frequency within time window."""
        with self._lock:
            cutoff_time = datetime.utcnow() - self.time_window
            recent_errors = [
                err for err in self._errors 
                if err["timestamp"] > cutoff_time and err["error_type"] == error_type
            ]
            
            if not recent_errors:
                return 0.0
            
            # Return errors per hour
            time_span_hours = self.time_window.total_seconds() / 3600
            return len(recent_errors) / time_span_hours
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        with self._lock:
            cutoff_time = datetime.utcnow() - self.time_window
            recent_errors = [err for err in self._errors if err["timestamp"] > cutoff_time]
            
            if not recent_errors:
                return {}
            
            # Count by type
            type_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for error in recent_errors:
                type_counts[error["error_type"]] += 1
                if "severity" in error:
                    severity_counts[error["severity"]] += 1
                if "category" in error:
                    category_counts[error["category"]] += 1
            
            return {
                "total_errors": len(recent_errors),
                "error_types": dict(type_counts),
                "severities": dict(severity_counts),
                "categories": dict(category_counts),
                "time_window_hours": self.time_window.total_seconds() / 3600
            }


class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(
        self,
        default_strategy: Optional[ErrorRecoveryStrategy] = None,
        enable_tracking: bool = True
    ):
        self.default_strategy = default_strategy or ErrorRecoveryStrategy()
        self.enable_tracking = enable_tracking
        
        # Strategy registry for specific error types
        self._strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {}
        
        # Error tracking
        self.tracker = ErrorTracker() if enable_tracking else None
        
        # Circuit breaker for repeated failures
        self._circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "failure_count": 0,
            "last_failure": None,
            "is_open": False,
            "failure_threshold": 5,
            "recovery_timeout": 300  # 5 minutes
        })
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def register_strategy(
        self, 
        error_type: Type[Exception], 
        strategy: ErrorRecoveryStrategy
    ):
        """Register recovery strategy for specific error type."""
        self._strategies[error_type] = strategy
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle error and return recovery information.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            operation_name: Name of the operation that failed (for circuit breaker)
            
        Returns:
            Dictionary with error handling results
        """
        # Record error for tracking
        if self.tracker:
            self.tracker.record_error(error, context)
        
        # Log error with appropriate level
        error_info = self._format_error_info(error, context)
        severity = self._get_error_severity(error)
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_info)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(error_info)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_info)
        else:
            self.logger.info(error_info)
        
        # Check circuit breaker
        if operation_name:
            if self._is_circuit_open(operation_name):
                self.logger.warning(f"Circuit breaker open for {operation_name}")
                return {
                    "handled": False,
                    "recoverable": False,
                    "circuit_open": True,
                    "retry_after": self._circuit_breakers[operation_name]["recovery_timeout"]
                }
            
            self._record_failure(operation_name)
        
        # Get recovery strategy
        strategy = self._get_strategy(error)
        
        # Determine if error is recoverable
        recoverable = strategy.is_recoverable(error)
        
        # Get retry information
        retry_after = None
        if recoverable:
            if isinstance(error, OptionsEngineError) and error.retry_after:
                retry_after = error.retry_after
            else:
                retry_after = strategy.get_delay(1)  # First retry delay
        
        return {
            "handled": True,
            "recoverable": recoverable,
            "retry_after": retry_after,
            "max_retries": strategy.max_retries if recoverable else 0,
            "error_id": getattr(error, "error_id", None),
            "severity": severity.value,
            "circuit_open": False
        }
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        operation_name: Optional[str] = None,
        strategy: Optional[ErrorRecoveryStrategy] = None,
        **kwargs
    ):
        """
        Execute function with automatic retry and backoff.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            operation_name: Name for circuit breaker tracking
            strategy: Custom recovery strategy
            **kwargs: Keyword arguments for func
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        effective_strategy = strategy or self.default_strategy
        last_exception = None
        
        # Check circuit breaker
        if operation_name and self._is_circuit_open(operation_name):
            raise ExternalServiceError(
                f"Circuit breaker open for {operation_name}",
                service_name=operation_name,
                retry_after=self._circuit_breakers[operation_name]["recovery_timeout"]
            )
        
        for attempt in range(effective_strategy.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Reset circuit breaker on success
                if operation_name:
                    self._record_success(operation_name)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Handle the error
                error_result = self.handle_error(e, operation_name=operation_name)
                
                # If circuit is open, don't retry
                if error_result.get("circuit_open", False):
                    break
                
                # If not recoverable or last attempt, don't retry
                if not error_result.get("recoverable", False) or attempt == effective_strategy.max_retries:
                    break
                
                # Calculate delay and wait
                delay = effective_strategy.get_delay(attempt + 1)
                
                # Use error-specific retry_after if available
                if error_result.get("retry_after"):
                    delay = max(delay, error_result["retry_after"])
                
                self.logger.info(
                    f"Retrying {func.__name__} in {delay:.1f}s (attempt {attempt + 1}/{effective_strategy.max_retries})"
                )
                
                time.sleep(delay)
        
        # All retries failed
        raise last_exception
    
    def _get_strategy(self, error: Exception) -> ErrorRecoveryStrategy:
        """Get appropriate recovery strategy for error."""
        error_type = type(error)
        
        # Check for exact type match
        if error_type in self._strategies:
            return self._strategies[error_type]
        
        # Check for parent type matches
        for registered_type, strategy in self._strategies.items():
            if isinstance(error, registered_type):
                return strategy
        
        return self.default_strategy
    
    def _get_error_severity(self, error: Exception) -> ErrorSeverity:
        """Get error severity level."""
        if isinstance(error, OptionsEngineError):
            return error.severity
        
        # Default severity based on exception type
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _format_error_info(self, error: Exception, context: Optional[Dict[str, Any]]) -> str:
        """Format error information for logging."""
        if isinstance(error, OptionsEngineError):
            info_parts = [
                f"Error ID: {error.error_id}",
                f"Message: {error.message}",
                f"Category: {error.category.value}",
                f"Severity: {error.severity.value}"
            ]
            
            if error.context:
                info_parts.append(f"Context: {error.context}")
        else:
            info_parts = [
                f"Exception: {type(error).__name__}",
                f"Message: {str(error)}"
            ]
            
            if context:
                info_parts.append(f"Context: {context}")
        
        # Add stack trace for debugging
        info_parts.append(f"Traceback: {traceback.format_exc()}")
        
        return " | ".join(info_parts)
    
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for operation."""
        breaker = self._circuit_breakers[operation_name]
        
        if not breaker["is_open"]:
            return False
        
        # Check if recovery timeout has passed
        if breaker["last_failure"]:
            time_since_failure = (datetime.utcnow() - breaker["last_failure"]).total_seconds()
            if time_since_failure > breaker["recovery_timeout"]:
                # Reset circuit breaker
                breaker["is_open"] = False
                breaker["failure_count"] = 0
                return False
        
        return True
    
    def _record_failure(self, operation_name: str):
        """Record failure for circuit breaker."""
        breaker = self._circuit_breakers[operation_name]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        if breaker["failure_count"] >= breaker["failure_threshold"]:
            breaker["is_open"] = True
            self.logger.warning(f"Circuit breaker opened for {operation_name}")
    
    def _record_success(self, operation_name: str):
        """Record success for circuit breaker."""
        breaker = self._circuit_breakers[operation_name]
        breaker["failure_count"] = 0
        breaker["is_open"] = False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        stats = {
            "circuit_breakers": {}
        }
        
        # Circuit breaker stats
        for name, breaker in self._circuit_breakers.items():
            stats["circuit_breakers"][name] = {
                "failure_count": breaker["failure_count"],
                "is_open": breaker["is_open"],
                "last_failure": breaker["last_failure"].isoformat() if breaker["last_failure"] else None
            }
        
        # Error tracking stats
        if self.tracker:
            stats["error_patterns"] = self.tracker.get_error_patterns()
        
        return stats
    
    def reset_circuit_breaker(self, operation_name: str):
        """Manually reset circuit breaker."""
        if operation_name in self._circuit_breakers:
            self._circuit_breakers[operation_name] = {
                "failure_count": 0,
                "last_failure": None,
                "is_open": False,
                "failure_threshold": 5,
                "recovery_timeout": 300
            }
            self.logger.info(f"Circuit breaker reset for {operation_name}")


def handle_errors(
    operation_name: Optional[str] = None,
    strategy: Optional[ErrorRecoveryStrategy] = None,
    reraise: bool = True
):
    """
    Decorator for automatic error handling.
    
    Args:
        operation_name: Name for circuit breaker tracking
        strategy: Custom recovery strategy
        reraise: Whether to reraise exceptions after handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get global error handler (would be injected in real implementation)
            error_handler = ErrorHandler()
            
            try:
                return error_handler.retry_with_backoff(
                    func, 
                    *args, 
                    operation_name=operation_name,
                    strategy=strategy,
                    **kwargs
                )
            except Exception as e:
                error_handler.handle_error(e, operation_name=operation_name)
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator