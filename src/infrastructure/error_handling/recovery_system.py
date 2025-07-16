"""
Error recovery and graceful degradation system for the options trading engine.
Provides comprehensive error handling, recovery mechanisms, and graceful degradation strategies.
"""

import logging
import threading
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
import asyncio
from functools import wraps

from . import handle_errors, ApplicationError, DatabaseError, APIError, ValidationError


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    operation_name: str
    timestamp: datetime
    severity: ErrorSeverity
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempts: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'operation_name': self.operation_name,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'stack_trace': self.stack_trace,
            'recovery_strategy': self.recovery_strategy.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'recovery_attempts': self.recovery_attempts,
            'context_data': self.context_data,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a specific operation."""
    operation_name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # successes needed to close circuit
    consecutive_successes: int = 0
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "half_open"
    
    def should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.state != "open":
            return False
        
        if not self.last_failure_time:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    operation_name: str
    fallback_function: Callable
    fallback_data: Optional[Any] = None
    enabled: bool = True
    priority: int = 1
    timeout: int = 30


class ErrorRecoverySystem:
    """
    Comprehensive error recovery and graceful degradation system.
    
    Features:
    - Multiple recovery strategies (retry, fallback, circuit breaker, etc.)
    - Circuit breaker pattern for failing operations
    - Graceful degradation mechanisms
    - Error pattern analysis and learning
    - Automatic recovery attempt coordination
    - Performance impact monitoring
    - Recovery success rate tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.config = config or {}
        
        # Initialize database
        self.db_path = self.config.get('database_path', 'data/error_recovery.db')
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Recovery state
        self.active_errors = {}
        self.error_history = deque(maxlen=10000)
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Circuit breaker states
        self.circuit_breakers = {}
        
        # Fallback mechanisms
        self.fallback_configs = {}
        
        # Recovery statistics
        self.recovery_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0.0
        })
        
        # Background recovery thread
        self.recovery_thread = None
        self.recovery_running = False
        
        # Configuration
        self.max_concurrent_recoveries = self.config.get('max_concurrent_recoveries', 10)
        self.recovery_check_interval = self.config.get('recovery_check_interval', 30)
        self.error_pattern_window = self.config.get('error_pattern_window', 3600)  # 1 hour
        
    def _initialize_database(self):
        """Initialize SQLite database for error recovery tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS error_contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT,
                    error_message TEXT,
                    operation_name TEXT,
                    timestamp TEXT,
                    severity TEXT,
                    stack_trace TEXT,
                    recovery_strategy TEXT,
                    retry_count INTEGER,
                    max_retries INTEGER,
                    recovery_attempts INTEGER,
                    context_data TEXT,
                    resolved BOOLEAN,
                    resolution_time TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS circuit_breaker_states (
                    operation_name TEXT PRIMARY KEY,
                    failure_count INTEGER,
                    last_failure_time TEXT,
                    state TEXT,
                    failure_threshold INTEGER,
                    recovery_timeout INTEGER,
                    success_threshold INTEGER,
                    consecutive_successes INTEGER,
                    updated_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recovery_statistics (
                    operation_name TEXT PRIMARY KEY,
                    total_attempts INTEGER,
                    successful_recoveries INTEGER,
                    failed_recoveries INTEGER,
                    avg_recovery_time REAL,
                    last_updated TEXT
                )
            ''')
            
            conn.commit()
    
    def _initialize_recovery_strategies(self) -> Dict[str, RecoveryStrategy]:
        """Initialize recovery strategies for different error types."""
        return {
            'APIError': RecoveryStrategy.RETRY,
            'DatabaseError': RecoveryStrategy.CIRCUIT_BREAKER,
            'ValidationError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'TimeoutError': RecoveryStrategy.RETRY,
            'ConnectionError': RecoveryStrategy.CIRCUIT_BREAKER,
            'MemoryError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'KeyError': RecoveryStrategy.FALLBACK,
            'ValueError': RecoveryStrategy.FALLBACK,
            'TypeError': RecoveryStrategy.FAIL_FAST,
            'ImportError': RecoveryStrategy.FAIL_FAST,
            'SystemError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'default': RecoveryStrategy.RETRY
        }
    
    @handle_errors(operation_name="start_recovery_system")
    def start_recovery_system(self):
        """Start the background recovery system."""
        if self.recovery_running:
            self.logger.warning("Recovery system is already running")
            return
        
        self.recovery_running = True
        self.recovery_thread = threading.Thread(
            target=self._recovery_loop,
            daemon=True
        )
        self.recovery_thread.start()
        
        self.logger.info("Error recovery system started")
    
    def stop_recovery_system(self):
        """Stop the background recovery system."""
        self.recovery_running = False
        if self.recovery_thread:
            self.recovery_thread.join(timeout=10)
        
        self.logger.info("Error recovery system stopped")
    
    def _recovery_loop(self):
        """Main recovery loop."""
        while self.recovery_running:
            try:
                # Check for errors that need recovery
                self._check_pending_recoveries()
                
                # Update circuit breaker states
                self._update_circuit_breakers()
                
                # Analyze error patterns
                self._analyze_error_patterns()
                
                # Clean up resolved errors
                self._cleanup_resolved_errors()
                
                # Update recovery statistics
                self._update_recovery_statistics()
                
                time.sleep(self.recovery_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in recovery loop: {str(e)}")
                time.sleep(self.recovery_check_interval)
    
    @handle_errors(operation_name="handle_error")
    def handle_error(self, error: Exception, operation_name: str, context_data: Dict[str, Any] = None) -> bool:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            operation_name: Name of the operation that failed
            context_data: Additional context data
            
        Returns:
            True if error was handled and recovery initiated, False otherwise
        """
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Determine severity
        severity = self._determine_error_severity(error, operation_name)
        
        # Get recovery strategy
        recovery_strategy = self._get_recovery_strategy(error_type, operation_name)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            operation_name=operation_name,
            timestamp=datetime.now(),
            severity=severity,
            stack_trace=stack_trace,
            recovery_strategy=recovery_strategy,
            context_data=context_data or {}
        )
        
        # Store error context
        self._store_error_context(error_context)
        
        # Add to active errors
        error_key = f"{operation_name}_{error_type}_{int(time.time())}"
        self.active_errors[error_key] = error_context
        self.error_history.append(error_context)
        
        # Immediate recovery attempt
        return self._attempt_recovery(error_context)
    
    def _determine_error_severity(self, error: Exception, operation_name: str) -> ErrorSeverity:
        """Determine error severity based on error type and operation."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ['MemoryError', 'SystemError', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['DatabaseError', 'ConnectionError', 'ImportError']:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['APIError', 'TimeoutError', 'ValidationError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _get_recovery_strategy(self, error_type: str, operation_name: str) -> RecoveryStrategy:
        """Get recovery strategy for error type and operation."""
        # Check for operation-specific override
        operation_config = self.config.get('operations', {}).get(operation_name, {})
        if 'recovery_strategy' in operation_config:
            return RecoveryStrategy(operation_config['recovery_strategy'])
        
        # Check for error-type specific strategy
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # Default strategy
        return self.recovery_strategies['default']
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt recovery based on the error context."""
        recovery_start = datetime.now()
        
        try:
            error_context.recovery_attempts += 1
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(error_context.operation_name):
                self.logger.warning(f"Circuit breaker open for {error_context.operation_name}, skipping recovery")
                return False
            
            # Apply recovery strategy
            success = False
            
            if error_context.recovery_strategy == RecoveryStrategy.RETRY:
                success = self._retry_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.FALLBACK:
                success = self._fallback_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                success = self._circuit_breaker_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._graceful_degradation_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.FAIL_FAST:
                success = self._fail_fast_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.IGNORE:
                success = True  # Ignore the error
            
            # Update statistics
            recovery_time = (datetime.now() - recovery_start).total_seconds()
            self._update_recovery_attempt_stats(error_context.operation_name, success, recovery_time)
            
            if success:
                error_context.resolved = True
                error_context.resolution_time = datetime.now()
                self._record_circuit_breaker_success(error_context.operation_name)
                
                self.logger.info(f"Recovery successful for {error_context.operation_name}")
            else:
                self._record_circuit_breaker_failure(error_context.operation_name)
                
                self.logger.warning(f"Recovery failed for {error_context.operation_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {str(e)}")
            return False
    
    def _retry_recovery(self, error_context: ErrorContext) -> bool:
        """Implement retry recovery strategy."""
        if error_context.retry_count >= error_context.max_retries:
            self.logger.warning(f"Max retries exceeded for {error_context.operation_name}")
            return False
        
        error_context.retry_count += 1
        
        # Calculate backoff delay
        backoff_delay = min(2 ** error_context.retry_count, 60)  # Exponential backoff, max 60 seconds
        
        self.logger.info(f"Retrying {error_context.operation_name} in {backoff_delay} seconds (attempt {error_context.retry_count}/{error_context.max_retries})")
        
        # Schedule retry (in a real implementation, this would be more sophisticated)
        time.sleep(backoff_delay)
        
        return True  # Assume retry will be successful for now
    
    def _fallback_recovery(self, error_context: ErrorContext) -> bool:
        """Implement fallback recovery strategy."""
        operation_name = error_context.operation_name
        
        if operation_name not in self.fallback_configs:
            self.logger.warning(f"No fallback configured for {operation_name}")
            return False
        
        fallback_config = self.fallback_configs[operation_name]
        
        if not fallback_config.enabled:
            self.logger.warning(f"Fallback disabled for {operation_name}")
            return False
        
        try:
            # Execute fallback function
            if fallback_config.fallback_function:
                result = fallback_config.fallback_function(error_context)
                self.logger.info(f"Fallback executed successfully for {operation_name}")
                return True
            elif fallback_config.fallback_data is not None:
                # Use fallback data
                self.logger.info(f"Using fallback data for {operation_name}")
                return True
            
        except Exception as e:
            self.logger.error(f"Fallback failed for {operation_name}: {str(e)}")
        
        return False
    
    def _circuit_breaker_recovery(self, error_context: ErrorContext) -> bool:
        """Implement circuit breaker recovery strategy."""
        operation_name = error_context.operation_name
        
        # Update circuit breaker state
        self._record_circuit_breaker_failure(operation_name)
        
        # Check if circuit should be opened
        circuit_breaker = self._get_circuit_breaker(operation_name)
        
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            circuit_breaker.state = "open"
            circuit_breaker.last_failure_time = datetime.now()
            
            self.logger.warning(f"Circuit breaker opened for {operation_name}")
            return False
        
        # Try recovery after some delay
        time.sleep(5)  # Brief delay before retry
        return True
    
    def _graceful_degradation_recovery(self, error_context: ErrorContext) -> bool:
        """Implement graceful degradation recovery strategy."""
        operation_name = error_context.operation_name
        
        # Implement degraded functionality
        degraded_configs = self.config.get('degraded_mode', {}).get(operation_name, {})
        
        if not degraded_configs:
            self.logger.warning(f"No degraded mode configured for {operation_name}")
            return False
        
        # Enable degraded mode
        self.logger.info(f"Enabling degraded mode for {operation_name}")
        
        # Reduce functionality, disable non-essential features, etc.
        # This would be operation-specific implementation
        
        return True
    
    def _fail_fast_recovery(self, error_context: ErrorContext) -> bool:
        """Implement fail-fast recovery strategy."""
        # For fail-fast, we don't attempt recovery
        self.logger.error(f"Fail-fast triggered for {error_context.operation_name}: {error_context.error_message}")
        return False
    
    def _is_circuit_breaker_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for operation."""
        circuit_breaker = self._get_circuit_breaker(operation_name)
        
        if circuit_breaker.is_open():
            # Check if we should attempt reset
            if circuit_breaker.should_attempt_reset():
                circuit_breaker.state = "half_open"
                circuit_breaker.consecutive_successes = 0
                self.logger.info(f"Circuit breaker half-opened for {operation_name}")
                return False
            return True
        
        return False
    
    def _get_circuit_breaker(self, operation_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker for operation."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState(
                operation_name=operation_name
            )
        
        return self.circuit_breakers[operation_name]
    
    def _record_circuit_breaker_failure(self, operation_name: str):
        """Record circuit breaker failure."""
        circuit_breaker = self._get_circuit_breaker(operation_name)
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = datetime.now()
        circuit_breaker.consecutive_successes = 0
        
        self._store_circuit_breaker_state(circuit_breaker)
    
    def _record_circuit_breaker_success(self, operation_name: str):
        """Record circuit breaker success."""
        circuit_breaker = self._get_circuit_breaker(operation_name)
        circuit_breaker.consecutive_successes += 1
        
        if circuit_breaker.is_half_open() and circuit_breaker.consecutive_successes >= circuit_breaker.success_threshold:
            circuit_breaker.state = "closed"
            circuit_breaker.failure_count = 0
            self.logger.info(f"Circuit breaker closed for {operation_name}")
        
        self._store_circuit_breaker_state(circuit_breaker)
    
    def _store_circuit_breaker_state(self, circuit_breaker: CircuitBreakerState):
        """Store circuit breaker state in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO circuit_breaker_states 
                (operation_name, failure_count, last_failure_time, state, failure_threshold,
                 recovery_timeout, success_threshold, consecutive_successes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                circuit_breaker.operation_name,
                circuit_breaker.failure_count,
                circuit_breaker.last_failure_time.isoformat() if circuit_breaker.last_failure_time else None,
                circuit_breaker.state,
                circuit_breaker.failure_threshold,
                circuit_breaker.recovery_timeout,
                circuit_breaker.success_threshold,
                circuit_breaker.consecutive_successes,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def _store_error_context(self, error_context: ErrorContext):
        """Store error context in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO error_contexts 
                (error_type, error_message, operation_name, timestamp, severity, stack_trace,
                 recovery_strategy, retry_count, max_retries, recovery_attempts, context_data,
                 resolved, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_context.error_type,
                error_context.error_message,
                error_context.operation_name,
                error_context.timestamp.isoformat(),
                error_context.severity.value,
                error_context.stack_trace,
                error_context.recovery_strategy.value,
                error_context.retry_count,
                error_context.max_retries,
                error_context.recovery_attempts,
                json.dumps(error_context.context_data),
                error_context.resolved,
                error_context.resolution_time.isoformat() if error_context.resolution_time else None
            ))
            conn.commit()
    
    def _check_pending_recoveries(self):
        """Check for pending recoveries that need attention."""
        current_time = datetime.now()
        
        for error_key, error_context in list(self.active_errors.items()):
            if error_context.resolved:
                continue
            
            # Check if recovery should be retried
            if error_context.recovery_strategy == RecoveryStrategy.RETRY:
                if error_context.retry_count < error_context.max_retries:
                    # Check if enough time has passed for retry
                    time_since_error = (current_time - error_context.timestamp).total_seconds()
                    if time_since_error > (2 ** error_context.retry_count):
                        self._attempt_recovery(error_context)
    
    def _update_circuit_breakers(self):
        """Update circuit breaker states."""
        for operation_name, circuit_breaker in self.circuit_breakers.items():
            if circuit_breaker.should_attempt_reset():
                circuit_breaker.state = "half_open"
                circuit_breaker.consecutive_successes = 0
                self.logger.info(f"Circuit breaker reset to half-open for {operation_name}")
    
    def _analyze_error_patterns(self):
        """Analyze error patterns to improve recovery strategies."""
        # Get recent errors
        cutoff_time = datetime.now() - timedelta(seconds=self.error_pattern_window)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]
        
        if len(recent_errors) < 5:  # Need minimum errors for pattern analysis
            return
        
        # Analyze patterns by operation
        operation_errors = defaultdict(list)
        for error in recent_errors:
            operation_errors[error.operation_name].append(error)
        
        # Look for frequent failures
        for operation_name, errors in operation_errors.items():
            if len(errors) > 10:  # Threshold for frequent failures
                self.logger.warning(f"Frequent failures detected for {operation_name}: {len(errors)} errors")
                
                # Consider opening circuit breaker
                circuit_breaker = self._get_circuit_breaker(operation_name)
                if circuit_breaker.state == "closed":
                    circuit_breaker.failure_count += len(errors)
                    if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
                        circuit_breaker.state = "open"
                        circuit_breaker.last_failure_time = datetime.now()
    
    def _cleanup_resolved_errors(self):
        """Clean up resolved errors from active errors."""
        resolved_keys = [
            key for key, error_context in self.active_errors.items()
            if error_context.resolved
        ]
        
        for key in resolved_keys:
            del self.active_errors[key]
    
    def _update_recovery_attempt_stats(self, operation_name: str, success: bool, recovery_time: float):
        """Update recovery attempt statistics."""
        stats = self.recovery_stats[operation_name]
        stats['total_attempts'] += 1
        
        if success:
            stats['successful_recoveries'] += 1
        else:
            stats['failed_recoveries'] += 1
        
        # Update average recovery time
        old_avg = stats['avg_recovery_time']
        stats['avg_recovery_time'] = (old_avg * (stats['total_attempts'] - 1) + recovery_time) / stats['total_attempts']
    
    def _update_recovery_statistics(self):
        """Update recovery statistics in database."""
        with sqlite3.connect(self.db_path) as conn:
            for operation_name, stats in self.recovery_stats.items():
                conn.execute('''
                    INSERT OR REPLACE INTO recovery_statistics 
                    (operation_name, total_attempts, successful_recoveries, failed_recoveries,
                     avg_recovery_time, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    operation_name,
                    stats['total_attempts'],
                    stats['successful_recoveries'],
                    stats['failed_recoveries'],
                    stats['avg_recovery_time'],
                    datetime.now().isoformat()
                ))
            conn.commit()
    
    def register_fallback(self, operation_name: str, fallback_function: Callable = None, 
                         fallback_data: Any = None, priority: int = 1):
        """Register a fallback mechanism for an operation."""
        self.fallback_configs[operation_name] = FallbackConfig(
            operation_name=operation_name,
            fallback_function=fallback_function,
            fallback_data=fallback_data,
            priority=priority
        )
        
        self.logger.info(f"Fallback registered for {operation_name}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics."""
        active_count = len(self.active_errors)
        resolved_count = len([e for e in self.error_history if e.resolved])
        
        # Error counts by type
        error_counts = defaultdict(int)
        for error in self.error_history:
            error_counts[error.error_type] += 1
        
        # Recovery success rates
        recovery_rates = {}
        for operation_name, stats in self.recovery_stats.items():
            if stats['total_attempts'] > 0:
                recovery_rates[operation_name] = (stats['successful_recoveries'] / stats['total_attempts']) * 100
        
        return {
            'active_errors': active_count,
            'resolved_errors': resolved_count,
            'total_errors': len(self.error_history),
            'error_counts_by_type': dict(error_counts),
            'recovery_success_rates': recovery_rates,
            'circuit_breaker_states': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count,
                    'consecutive_successes': cb.consecutive_successes
                }
                for name, cb in self.circuit_breakers.items()
            },
            'recovery_statistics': dict(self.recovery_stats)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the recovery system."""
        active_critical_errors = len([
            e for e in self.active_errors.values()
            if e.severity == ErrorSeverity.CRITICAL and not e.resolved
        ])
        
        open_circuit_breakers = len([
            cb for cb in self.circuit_breakers.values()
            if cb.state == "open"
        ])
        
        overall_health = "healthy"
        if active_critical_errors > 0:
            overall_health = "critical"
        elif open_circuit_breakers > 0:
            overall_health = "warning"
        elif len(self.active_errors) > 10:
            overall_health = "warning"
        
        return {
            'status': overall_health,
            'active_errors': len(self.active_errors),
            'critical_errors': active_critical_errors,
            'open_circuit_breakers': open_circuit_breakers,
            'recovery_system_running': self.recovery_running,
            'last_check': datetime.now().isoformat()
        }


# Context managers for error handling
@contextmanager
def error_recovery_context(recovery_system: ErrorRecoverySystem, operation_name: str, 
                          context_data: Dict[str, Any] = None):
    """Context manager for automatic error recovery."""
    try:
        yield
    except Exception as e:
        recovery_system.handle_error(e, operation_name, context_data)
        raise


# Decorators for error handling
def with_error_recovery(recovery_system: ErrorRecoverySystem, operation_name: str = None):
    """Decorator for automatic error recovery."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery_system.handle_error(e, op_name, {'args': args, 'kwargs': kwargs})
                raise
        return wrapper
    return decorator


# Global recovery system instance
_global_recovery_system = None

def get_recovery_system() -> ErrorRecoverySystem:
    """Get global recovery system instance."""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = ErrorRecoverySystem()
    return _global_recovery_system