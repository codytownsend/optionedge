"""
Error handling infrastructure package.
"""

from .exceptions import (
    OptionsEngineError,
    ErrorSeverity,
    ErrorCategory,
    DataQualityError,
    APIError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    BusinessLogicError,
    ExternalServiceError,
    InsufficientDataError,
    StaleDataError,
    RateLimitExceededError,
    InsufficientLiquidityError,
    ConstraintViolationError,
    GreeksCalculationError,
    ProbabilityCalculationError
)

from .error_handler import (
    ErrorRecoveryStrategy,
    ErrorTracker,
    ErrorHandler,
    handle_errors
)

__all__ = [
    # Exceptions
    "OptionsEngineError",
    "ErrorSeverity", 
    "ErrorCategory",
    "DataQualityError",
    "APIError",
    "ValidationError",
    "CalculationError",
    "ConfigurationError",
    "BusinessLogicError",
    "ExternalServiceError",
    "InsufficientDataError",
    "StaleDataError",
    "RateLimitExceededError",
    "InsufficientLiquidityError",
    "ConstraintViolationError",
    "GreeksCalculationError",
    "ProbabilityCalculationError",
    
    # Error handling
    "ErrorRecoveryStrategy",
    "ErrorTracker",
    "ErrorHandler",
    "handle_errors"
]