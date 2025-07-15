"""
API clients for external data sources.

This module provides API clients for all external data sources used
by the options trading engine:

- TradierClient: Options chain data and real-time quotes
- YahooFinanceClient: Fundamental data and historical prices
- FREDClient: Economic indicators and macro data
- QuiverQuantClient: Alternative data and sentiment analysis

All clients follow the same base architecture with:
- Retry logic and circuit breakers
- Rate limiting and request throttling
- Structured error handling
- Response caching capabilities
- Health monitoring and status checks
"""

from .base_client import BaseAPIClient, APIError, RateLimitError, DataQualityError
from .tradier_client import TradierClient
from .fred_client import FREDClient

# Note: Import will be enabled when file is renamed from yahoo_cleint.py to yahoo_client.py
# from .yahoo_client import YahooFinanceClient

# Note: Import will be uncommented when QuiverQuant client is implemented
# from .quiver_client import QuiverQuantClient

__all__ = [
    # Base classes
    "BaseAPIClient", "APIError", "RateLimitError", "DataQualityError",
    
    # API Clients
    "TradierClient",
    "FREDClient",
    # "YahooFinanceClient",  # Enable when file is renamed
    # "QuiverQuantClient",   # Enable when implemented
]