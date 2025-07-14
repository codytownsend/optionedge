"""
Base API client with retry logic, rate limiting, and error handling.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base API error."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class DataQualityError(APIError):
    """Data quality issue error."""
    pass


class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock() if asyncio.iscoroutinefunction(self.__init__) else None
    
    def can_proceed(self) -> bool:
        """Check if request can proceed without hitting rate limit."""
        now = time.time()
        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.window_seconds]
        
        return len(self.requests) < self.max_requests
    
    def add_request(self):
        """Record a new request."""
        now = time.time()
        self.requests.append(now)
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request can be made."""
        if self.can_proceed():
            return 0.0
        
        # Find the oldest request that needs to expire
        now = time.time()
        oldest_in_window = min(self.requests)
        return max(0, self.window_seconds - (now - oldest_in_window))


class BaseAPIClient(ABC):
    """Base class for all API clients with common functionality."""
    
    def __init__(self, 
                 base_url: str,
                 api_key: str,
                 rate_limit_per_minute: int = 60,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize base API client.
        
        Args:
            base_url: API base URL
            api_key: API authentication key
            rate_limit_per_minute: Maximum requests per minute
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Rate limiting
        self.rate_limiter = RateLimiter(rate_limit_per_minute, 60)
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update(self._get_default_headers())
        
        # Circuit breaker state
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
    
    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        pass
    
    @abstractmethod
    def _validate_response(self, response: requests.Response) -> bool:
        """Validate API response format and data quality."""
        pass
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.consecutive_failures < self.circuit_breaker_threshold:
            return False
        
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure < self.circuit_breaker_timeout
    
    def _record_success(self):
        """Record successful API call."""
        self.consecutive_failures = 0
        self.last_failure_time = None
    
    def _record_failure(self):
        """Record failed API call."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
    
    def _wait_for_rate_limit(self):
        """Wait if rate limit would be exceeded."""
        wait_time = self.rate_limiter.get_wait_time()
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, RateLimitError))
    )
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            headers: Additional headers
            
        Returns:
            Response object
            
        Raises:
            APIError: For API-specific errors
            RateLimitError: When rate limited
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise APIError(f"Circuit breaker open for {self.__class__.__name__}")
        
        # Check rate limit
        self._wait_for_rate_limit()
        
        # Prepare request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        try:
            # Make request
            self.rate_limiter.add_request()
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data if method.upper() in ['POST', 'PUT', 'PATCH'] else None,
                headers=request_headers,
                timeout=self.timeout
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self._record_failure()
                raise RateLimitError(f"Rate limit exceeded", retry_after)
            
            # Handle other HTTP errors
            if not response.ok:
                self._record_failure()
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise APIError(error_msg, response.status_code)
            
            # Validate response
            if not self._validate_response(response):
                self._record_failure()
                raise DataQualityError("Response failed validation")
            
            self._record_success()
            return response
            
        except requests.RequestException as e:
            self._record_failure()
            logger.error(f"Request failed: {e}")
            raise APIError(f"Request failed: {e}")
    
    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Make GET request and return JSON response."""
        response = self._make_request('GET', endpoint, params=params, **kwargs)
        return response.json()
    
    def post(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Make POST request and return JSON response."""
        response = self._make_request('POST', endpoint, data=data, **kwargs)
        return response.json()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get API client health status."""
        return {
            'client_name': self.__class__.__name__,
            'base_url': self.base_url,
            'consecutive_failures': self.consecutive_failures,
            'circuit_breaker_open': self._is_circuit_breaker_open(),
            'last_failure_time': self.last_failure_time,
            'rate_limit_requests': len(self.rate_limiter.requests),
            'rate_limit_capacity': self.rate_limiter.max_requests
        }


class DataValidator:
    """Utility class for validating API response data."""
    
    @staticmethod
    def validate_quote_data(data: Dict) -> bool:
        """Validate options or stock quote data."""
        required_fields = ['symbol', 'bid', 'ask']
        
        # Check required fields exist
        if not all(field in data for field in required_fields):
            return False
        
        # Check bid/ask are valid numbers
        try:
            bid = float(data['bid']) if data['bid'] is not None else None
            ask = float(data['ask']) if data['ask'] is not None else None
            
            if bid is not None and ask is not None:
                # Bid should be <= ask
                if bid > ask:
                    logger.warning(f"Invalid bid/ask: bid={bid} > ask={ask}")
                    return False
                
                # Bid and ask should be positive
                if bid < 0 or ask < 0:
                    logger.warning(f"Negative bid/ask: bid={bid}, ask={ask}")
                    return False
                
                # Spread should be reasonable (< 50% of mid)
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid if mid > 0 else float('inf')
                if spread_pct > 0.5:
                    logger.warning(f"Wide spread: {spread_pct:.2%}")
                    return False
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid numeric data: {e}")
            return False
    
    @staticmethod
    def validate_timestamp(timestamp: Union[str, int, float]) -> bool:
        """Validate timestamp is recent enough."""
        try:
            if isinstance(timestamp, str):
                # Try parsing ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Assume Unix timestamp
                dt = datetime.fromtimestamp(float(timestamp))
            
            # Check if timestamp is within last 24 hours
            age = datetime.utcnow() - dt
            return age < timedelta(hours=24)
            
        except (ValueError, OSError):
            return False
    
    @staticmethod
    def validate_greeks(greeks: Dict) -> bool:
        """Validate options Greeks are within reasonable ranges."""
        validations = [
            ('delta', lambda x: -1 <= x <= 1),
            ('gamma', lambda x: x >= 0),
            ('theta', lambda x: x <= 0),  # Theta is typically negative
            ('vega', lambda x: x >= 0),
            ('rho', lambda x: True)  # Rho can be positive or negative
        ]
        
        for greek, validator in validations:
            if greek in greeks and greeks[greek] is not None:
                try:
                    value = float(greeks[greek])
                    if not validator(value):
                        logger.warning(f"Invalid {greek}: {value}")
                        return False
                except (ValueError, TypeError):
                    logger.warning(f"Invalid {greek} format: {greeks[greek]}")
                    return False
        
        return True