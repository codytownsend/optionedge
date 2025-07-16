"""
Custom exception classes for the options trading engine.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_QUALITY = "data_quality"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class OptionsEngineError(Exception):
    """Base exception class for the options trading engine."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow()
        self.error_id = f"{self.category.value}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat()
        }


class DataQualityError(OptionsEngineError):
    """Raised when data quality issues are detected."""
    
    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        symbol: Optional[str] = None,
        quality_issues: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "data_source": data_source,
            "symbol": symbol,
            "quality_issues": quality_issues or []
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_QUALITY,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class APIError(OptionsEngineError):
    """Raised when API calls fail."""
    
    def __init__(
        self,
        message: str,
        api_provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "api_provider": api_provider,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_data": response_data
        })
        
        # Determine if error is recoverable based on status code
        recoverable = True
        retry_after = None
        
        if status_code:
            if status_code == 429:  # Rate limited
                retry_after = 60
            elif 500 <= status_code < 600:  # Server errors are usually temporary
                retry_after = 30
            elif 400 <= status_code < 500 and status_code not in [429, 408]:  # Client errors
                recoverable = False
        
        super().__init__(
            message=message,
            category=ErrorCategory.API_ERROR,
            severity=ErrorSeverity.HIGH if not recoverable else ErrorSeverity.MEDIUM,
            context=context,
            recoverable=recoverable,
            retry_after=retry_after,
            **kwargs
        )


class ValidationError(OptionsEngineError):
    """Raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Any = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "field_name": field_name,
            "field_value": str(field_value) if field_value is not None else None,
            "validation_rule": validation_rule
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            context=context,
            recoverable=False,  # Validation errors require code/data fixes
            **kwargs
        )


class CalculationError(OptionsEngineError):
    """Raised when calculations fail or produce invalid results."""
    
    def __init__(
        self,
        message: str,
        calculation_type: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "calculation_type": calculation_type,
            "input_data": input_data
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.CALCULATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class ConfigurationError(OptionsEngineError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            recoverable=False,  # Config errors need manual fixes
            **kwargs
        )


class BusinessLogicError(OptionsEngineError):
    """Raised when business logic constraints are violated."""
    
    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        rule_description: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "rule_name": rule_name,
            "rule_description": rule_description
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class ExternalServiceError(OptionsEngineError):
    """Raised when external services are unavailable or malfunctioning."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        service_endpoint: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "service_name": service_name,
            "service_endpoint": service_endpoint
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            retry_after=60,  # External services often recover
            **kwargs
        )


class InsufficientDataError(DataQualityError):
    """Raised when insufficient data is available for analysis."""
    
    def __init__(
        self,
        message: str,
        required_data_points: Optional[int] = None,
        available_data_points: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "required_data_points": required_data_points,
            "available_data_points": available_data_points
        })
        
        super().__init__(
            message=message,
            quality_issues=["insufficient_data"],
            context=context,
            **kwargs
        )


class StaleDataError(DataQualityError):
    """Raised when data is too old to be reliable."""
    
    def __init__(
        self,
        message: str,
        data_age_seconds: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "data_age_seconds": data_age_seconds,
            "max_age_seconds": max_age_seconds
        })
        
        super().__init__(
            message=message,
            quality_issues=["stale_data"],
            context=context,
            **kwargs
        )


class RateLimitExceededError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        requests_per_minute: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "requests_per_minute": requests_per_minute,
            "reset_time": reset_time.isoformat() if reset_time else None
        })
        
        # Calculate retry_after based on reset_time
        retry_after = 60  # Default
        if reset_time:
            retry_after = max(1, int((reset_time - datetime.utcnow()).total_seconds()))
        
        super().__init__(
            message=message,
            status_code=429,
            context=context,
            retry_after=retry_after,
            **kwargs
        )


class InsufficientLiquidityError(BusinessLogicError):
    """Raised when options don't meet liquidity requirements."""
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        volume: Optional[int] = None,
        open_interest: Optional[int] = None,
        bid_ask_spread: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "symbol": symbol,
            "volume": volume,
            "open_interest": open_interest,
            "bid_ask_spread": bid_ask_spread
        })
        
        super().__init__(
            message=message,
            rule_name="liquidity_requirements",
            context=context,
            **kwargs
        )


class ConstraintViolationError(BusinessLogicError):
    """Raised when trades violate risk constraints."""
    
    def __init__(
        self,
        message: str,
        constraint_name: Optional[str] = None,
        constraint_value: Any = None,
        actual_value: Any = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "constraint_name": constraint_name,
            "constraint_value": str(constraint_value) if constraint_value is not None else None,
            "actual_value": str(actual_value) if actual_value is not None else None
        })
        
        super().__init__(
            message=message,
            rule_name=constraint_name,
            context=context,
            **kwargs
        )


class GreeksCalculationError(CalculationError):
    """Raised when Greeks calculations fail."""
    
    def __init__(
        self,
        message: str,
        greek_type: Optional[str] = None,
        underlying_price: Optional[float] = None,
        strike: Optional[float] = None,
        time_to_expiration: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "greek_type": greek_type,
            "underlying_price": underlying_price,
            "strike": strike,
            "time_to_expiration": time_to_expiration
        })
        
        super().__init__(
            message=message,
            calculation_type="greeks",
            context=context,
            **kwargs
        )


class ProbabilityCalculationError(CalculationError):
    """Raised when probability calculations fail."""
    
    def __init__(
        self,
        message: str,
        probability_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({
            "probability_type": probability_type
        })
        
        super().__init__(
            message=message,
            calculation_type="probability",
            context=context,
            **kwargs
        )