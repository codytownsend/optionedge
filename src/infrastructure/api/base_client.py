"""
Abstract base class for all API clients with standardized functionality.
"""

import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int
    requests_per_hour: Optional[int] = None
    burst_allowance: int = 5


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker implementation for API resilience."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_call_count = 0
        
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit breaker state."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time >= timedelta(seconds=self.config.timeout_seconds)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_call_count = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_call_count < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful request."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_call_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_call_count += 1


class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_allowance
        self.last_refill = time.time()
        self.request_times: List[float] = []
    
    def can_proceed(self) -> bool:
        """Check if request can proceed based on rate limits."""
        now = time.time()
        
        # Refill tokens based on time elapsed
        time_elapsed = now - self.last_refill
        tokens_to_add = (time_elapsed / 60.0) * self.config.requests_per_minute
        self.tokens = min(self.config.burst_allowance, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if we have tokens available
        if self.tokens < 1:
            return False
        
        # Check hourly limit if configured
        if self.config.requests_per_hour:
            hour_ago = now - 3600
            recent_requests = [t for t in self.request_times if t > hour_ago]
            if len(recent_requests) >= self.config.requests_per_hour:
                return False
        
        return True
    
    def consume_token(self):
        """Consume a rate limit token."""
        self.tokens -= 1
        self.request_times.append(time.time())
        
        # Clean old request times (keep only last hour)
        hour_ago = time.time() - 3600
        self.request_times = [t for t in self.request_times if t > hour_ago]


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class RateLimitError(APIClientError):
    """Raised when rate limit is exceeded."""
    pass


class CircuitBreakerError(APIClientError):
    """Raised when circuit breaker is open."""
    pass


class BaseAPIClient(ABC):
    """
    Abstract base class for all API clients with standardized functionality.
    
    Provides:
    - Standardized request/response handling
    - Automatic retry logic with exponential backoff
    - Rate limiting respecting provider constraints
    - Circuit breaker pattern for failing services
    - Request/response logging for debugging
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rate_limit_config or RateLimitConfig(requests_per_minute=60)
        )
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s delays
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    def authenticate(self) -> Dict[str, str]:
        """Return authentication headers for requests."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the data provider."""
        pass
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"
    
    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        if not self.rate_limiter.can_proceed():
            self.logger.warning(f"Rate limit exceeded for {self.get_provider_name()}")
            raise RateLimitError(f"Rate limit exceeded for {self.get_provider_name()}")
    
    def _check_circuit_breaker(self):
        """Check circuit breaker state."""
        if not self.circuit_breaker.can_execute():
            self.logger.warning(f"Circuit breaker open for {self.get_provider_name()}")
            raise CircuitBreakerError(f"Circuit breaker open for {self.get_provider_name()}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> requests.Response:
        """Make HTTP request with all safety mechanisms."""
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Check rate limits
        self._check_rate_limit()
        
        # Build request
        url = self._build_url(endpoint)
        request_headers = self.authenticate()
        if headers:
            request_headers.update(headers)
        
        # Log request
        self.logger.debug(
            f"Making {method} request to {url} with params: {params}, headers: {list(request_headers.keys())}"
        )
        
        try:
            # Consume rate limit token
            self.rate_limiter.consume_token()
            
            # Make request
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                timeout=self.timeout
            )
            
            # Log response
            self.logger.debug(
                f"Response from {url}: status={response.status_code}, "
                f"content_length={len(response.content) if response.content else 0}"
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Record success
            self.circuit_breaker.record_success()
            
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {str(e)}")
            self.circuit_breaker.record_failure()
            raise APIClientError(f"Request failed: {str(e)}") from e
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make GET request and return JSON response."""
        response = self._make_request("GET", endpoint, params=params, headers=headers)
        
        try:
            return response.json()
        except ValueError as e:
            self.logger.error(f"Failed to parse JSON response from {endpoint}")
            raise APIClientError(f"Invalid JSON response: {str(e)}") from e
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make POST request and return JSON response."""
        response = self._make_request("POST", endpoint, params=params, data=data, headers=headers)
        
        try:
            return response.json()
        except ValueError as e:
            self.logger.error(f"Failed to parse JSON response from {endpoint}")
            raise APIClientError(f"Invalid JSON response: {str(e)}") from e
    
    def health_check(self) -> bool:
        """Perform health check on the API."""
        try:
            # This should be overridden by subclasses with provider-specific health checks
            response = self._make_request("GET", "/")
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Health check failed for {self.get_provider_name()}: {str(e)}")
            return False
    
    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.circuit_breaker.state
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "tokens_available": self.rate_limiter.tokens,
            "requests_per_minute": self.rate_limiter.config.requests_per_minute,
            "recent_request_count": len(self.rate_limiter.request_times)
        }
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (for admin use)."""
        self.circuit_breaker.state = CircuitBreakerState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.last_failure_time = None
        self.logger.info(f"Circuit breaker manually reset for {self.get_provider_name()}")