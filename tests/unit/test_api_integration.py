"""
API integration tests with mock responses.
Tests API failure and retry logic, data parsing, and error handling.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date
from decimal import Decimal
import json

from src.infrastructure.api.tradier_client import TradierClient
from src.infrastructure.api.yahoo_client import YahooClient
from src.infrastructure.api.fred_client import FredClient
from src.infrastructure.api.base_client import BaseAPIClient, APIError, RateLimitError
from src.data.models.options import OptionQuote, OptionType, Greeks
from src.data.models.market_data import StockQuote, TechnicalIndicators


class TestTradierAPIIntegration:
    """Test Tradier API integration with mock responses."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.api_key = "test_api_key"
        self.client = TradierClient(api_key=self.api_key)
    
    def test_tradier_options_chain_success(self):
        """Test Tradier options chain API call with successful response."""
        # Mock response matching Tradier API format
        mock_response = {
            "options": {
                "option": [
                    {
                        "symbol": "AAPL250117P00150000",
                        "description": "AAPL Jan 17 2025 $150.00 Put",
                        "exch": "Z",
                        "type": "put",
                        "last": 2.55,
                        "change": 0.05,
                        "volume": 150,
                        "open": 2.50,
                        "high": 2.60,
                        "low": 2.45,
                        "close": 2.50,
                        "bid": 2.50,
                        "ask": 2.55,
                        "underlying": "AAPL",
                        "strike": 150.0,
                        "greeks": {
                            "delta": -0.35,
                            "gamma": 0.05,
                            "theta": -0.08,
                            "vega": 0.12,
                            "rho": -0.02
                        },
                        "change_percentage": 2.0,
                        "average_volume": 200,
                        "last_volume": 150,
                        "trade_date": 1642723200000,
                        "prevclose": 2.50,
                        "week_52_high": 3.00,
                        "week_52_low": 1.50,
                        "bidsize": 10,
                        "bidexch": "Z",
                        "bid_date": 1642723200000,
                        "asksize": 15,
                        "askexch": "Z",
                        "ask_date": 1642723200000,
                        "open_interest": 500,
                        "contract_size": 100,
                        "expiration_date": "2025-01-17",
                        "expiration_type": "standard",
                        "option_type": "put",
                        "root_symbol": "AAPL"
                    }
                ]
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.status_code = 200
            mock_get.return_value = mock_response_obj
            
            # Test the API call
            options_data = self.client.get_options_chain("AAPL")
            
            # Verify request was made correctly
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert "api.tradier.com" in args[0]
            assert kwargs['headers']['Authorization'] == f"Bearer {self.api_key}"
            
            # Verify response parsing
            assert len(options_data) == 1
            option = options_data[0]
            assert option.symbol == "AAPL250117P00150000"
            assert option.strike == Decimal('150.0')
            assert option.option_type == OptionType.PUT
            assert option.bid == Decimal('2.50')
            assert option.ask == Decimal('2.55')
            assert option.volume == 150
            assert option.open_interest == 500
            assert option.greeks.delta == -0.35
            assert option.greeks.gamma == 0.05
            assert option.greeks.theta == -0.08
            assert option.greeks.vega == 0.12
            assert option.greeks.rho == -0.02
    
    def test_tradier_stock_quotes_success(self):
        """Test Tradier stock quotes API call with successful response."""
        mock_response = {
            "quotes": {
                "quote": [
                    {
                        "symbol": "AAPL",
                        "description": "Apple Inc",
                        "exch": "Q",
                        "type": "stock",
                        "last": 150.25,
                        "change": 1.25,
                        "volume": 45123456,
                        "open": 149.50,
                        "high": 151.00,
                        "low": 149.00,
                        "close": 150.25,
                        "bid": 150.20,
                        "ask": 150.30,
                        "change_percentage": 0.84,
                        "average_volume": 50000000,
                        "last_volume": 100,
                        "trade_date": 1642723200000,
                        "prevclose": 149.00,
                        "week_52_high": 180.00,
                        "week_52_low": 120.00,
                        "bidsize": 100,
                        "bidexch": "Q",
                        "bid_date": 1642723200000,
                        "asksize": 200,
                        "askexch": "Q",
                        "ask_date": 1642723200000
                    }
                ]
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.status_code = 200
            mock_get.return_value = mock_response_obj
            
            # Test the API call
            quotes_data = self.client.get_stock_quotes(["AAPL"])
            
            # Verify response parsing
            assert len(quotes_data) == 1
            quote = quotes_data[0]
            assert quote.symbol == "AAPL"
            assert quote.last == Decimal('150.25')
            assert quote.volume == 45123456
            assert quote.bid == Decimal('150.20')
            assert quote.ask == Decimal('150.30')
    
    def test_tradier_api_error_handling(self):
        """Test Tradier API error handling and retry logic."""
        with patch('requests.get') as mock_get:
            # Simulate 3 failures then success
            mock_get.side_effect = [
                requests.RequestException("Connection failed"),
                requests.RequestException("Timeout"),
                requests.RequestException("Server error"),
                Mock(status_code=200, json=lambda: {"quotes": {"quote": []}})
            ]
            
            # Should succeed after retries
            result = self.client.get_stock_quotes(["AAPL"])
            assert result == []
            assert mock_get.call_count == 4
    
    def test_tradier_rate_limit_handling(self):
        """Test Tradier rate limit handling."""
        with patch('requests.get') as mock_get:
            # Simulate rate limit error
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': '60'}
            mock_get.return_value = mock_response
            
            # Should raise RateLimitError
            with pytest.raises(RateLimitError):
                self.client.get_stock_quotes(["AAPL"])
    
    def test_tradier_invalid_response_handling(self):
        """Test handling of invalid API responses."""
        with patch('requests.get') as mock_get:
            # Simulate invalid JSON response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_get.return_value = mock_response
            
            # Should raise APIError
            with pytest.raises(APIError):
                self.client.get_stock_quotes(["AAPL"])


class TestYahooAPIIntegration:
    """Test Yahoo Finance API integration with mock responses."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.client = YahooClient()
    
    def test_yahoo_fundamentals_success(self):
        """Test Yahoo Finance fundamentals API call."""
        mock_info = {
            'trailingPE': 28.5,
            'pegRatio': 1.2,
            'grossMargins': 0.38,
            'operatingMargins': 0.27,
            'freeCashflow': 92000000000,
            'totalRevenue': 365000000000,
            'netIncomeToCommon': 95000000000,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 2800000000000,
            'enterpriseValue': 2750000000000,
            'bookValue': 3.85,
            'priceToBook': 39.0,
            'returnOnEquity': 1.47,
            'returnOnAssets': 0.27,
            'debtToEquity': 1.96,
            'currentRatio': 1.04,
            'quickRatio': 0.89
        }
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = mock_info
            mock_ticker.return_value = mock_ticker_instance
            
            # Test the API call
            fundamentals = self.client.get_fundamentals("AAPL")
            
            # Verify response parsing
            assert fundamentals.trailing_pe == 28.5
            assert fundamentals.peg_ratio == 1.2
            assert fundamentals.gross_margins == 0.38
            assert fundamentals.operating_margins == 0.27
            assert fundamentals.free_cashflow == 92000000000
            assert fundamentals.sector == 'Technology'
    
    def test_yahoo_price_history_success(self):
        """Test Yahoo Finance price history API call."""
        import pandas as pd
        
        # Create mock price data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        mock_history = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
            'High': [151.0, 152.0, 153.0, 154.0, 155.0],
            'Low': [149.0, 150.0, 151.0, 152.0, 153.0],
            'Close': [150.5, 151.5, 152.5, 153.5, 154.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_ticker_instance
            
            # Test the API call
            price_history = self.client.get_price_history("AAPL", days=5)
            
            # Verify response parsing
            assert len(price_history) == 5
            assert price_history[0].open == Decimal('150.0')
            assert price_history[0].high == Decimal('151.0')
            assert price_history[0].low == Decimal('149.0')
            assert price_history[0].close == Decimal('150.5')
            assert price_history[0].volume == 1000000
    
    def test_yahoo_api_error_handling(self):
        """Test Yahoo Finance API error handling."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {}  # Empty info
            mock_ticker.return_value = mock_ticker_instance
            
            # Should handle missing data gracefully
            fundamentals = self.client.get_fundamentals("INVALID")
            assert fundamentals.trailing_pe is None
            assert fundamentals.sector is None


class TestFredAPIIntegration:
    """Test FRED API integration with mock responses."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.api_key = "test_fred_api_key"
        self.client = FredClient(api_key=self.api_key)
    
    def test_fred_economic_data_success(self):
        """Test FRED economic data API call."""
        mock_response = {
            "observations": [
                {
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-01",
                    "date": "2024-01-01",
                    "value": "3.2"
                },
                {
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-01", 
                    "date": "2024-02-01",
                    "value": "3.3"
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.status_code = 200
            mock_get.return_value = mock_response_obj
            
            # Test the API call
            economic_data = self.client.get_economic_series("UNRATE")
            
            # Verify request was made correctly
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert "api.stlouisfed.org" in args[0]
            assert f"api_key={self.api_key}" in args[0]
            
            # Verify response parsing
            assert len(economic_data) == 2
            assert economic_data[0].value == 3.2
            assert economic_data[1].value == 3.3
    
    def test_fred_multiple_series_success(self):
        """Test FRED multiple series API call."""
        series_ids = ['CPIAUCSL', 'GDP', 'UNRATE', 'DGS10', 'HOUST']
        
        with patch.object(self.client, 'get_economic_series') as mock_get_series:
            mock_get_series.return_value = [
                Mock(date=date(2024, 1, 1), value=3.2),
                Mock(date=date(2024, 2, 1), value=3.3)
            ]
            
            # Test multiple series call
            all_data = self.client.get_multiple_series(series_ids)
            
            # Verify all series were requested
            assert mock_get_series.call_count == len(series_ids)
            assert len(all_data) == len(series_ids)
            
            # Verify each series has data
            for series_id in series_ids:
                assert series_id in all_data
                assert len(all_data[series_id]) == 2


class TestBaseAPIClient:
    """Test base API client functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.client = BaseAPIClient()
    
    def test_exponential_backoff_retry(self):
        """Test exponential backoff retry logic."""
        with patch('requests.get') as mock_get:
            # Simulate failures with exponential backoff
            mock_get.side_effect = [
                requests.RequestException("Failure 1"),
                requests.RequestException("Failure 2"),
                Mock(status_code=200, json=lambda: {"success": True})
            ]
            
            with patch('time.sleep') as mock_sleep:
                # Test retry with backoff
                result = self.client.make_request_with_retry("http://test.com")
                
                # Verify retry attempts
                assert mock_get.call_count == 3
                
                # Verify exponential backoff delays
                assert mock_sleep.call_count == 2
                sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert sleep_calls[0] == 1  # First retry: 1 second
                assert sleep_calls[1] == 2  # Second retry: 2 seconds
    
    def test_circuit_breaker_activation(self):
        """Test circuit breaker pattern activation."""
        with patch('requests.get') as mock_get:
            # Simulate continuous failures
            mock_get.side_effect = requests.RequestException("Continuous failure")
            
            # Make multiple failed requests to trigger circuit breaker
            for _ in range(6):  # 5 failures should trigger circuit breaker
                try:
                    self.client.make_request_with_retry("http://test.com")
                except:
                    pass
            
            # Circuit breaker should be open
            assert self.client.circuit_breaker.is_open()
            
            # Next request should fail immediately without HTTP call
            initial_call_count = mock_get.call_count
            
            with pytest.raises(APIError):
                self.client.make_request_with_retry("http://test.com")
            
            # No additional HTTP calls should have been made
            assert mock_get.call_count == initial_call_count
    
    def test_request_response_logging(self):
        """Test request/response logging functionality."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_get.return_value = mock_response
            
            with patch('logging.Logger.info') as mock_log:
                # Make request
                result = self.client.make_request_with_retry("http://test.com")
                
                # Verify logging occurred
                assert mock_log.call_count >= 2  # At least request and response logs
                
                # Verify log messages contain relevant info
                log_messages = [call[0][0] for call in mock_log.call_args_list]
                assert any("Making request" in msg for msg in log_messages)
                assert any("Request completed" in msg for msg in log_messages)
    
    def test_api_health_monitoring(self):
        """Test API health monitoring functionality."""
        with patch('requests.get') as mock_get:
            # Simulate mixed success/failure pattern
            mock_get.side_effect = [
                Mock(status_code=200, json=lambda: {"success": True}),
                requests.RequestException("Failure"),
                Mock(status_code=200, json=lambda: {"success": True}),
                Mock(status_code=200, json=lambda: {"success": True})
            ]
            
            # Make multiple requests
            for _ in range(4):
                try:
                    self.client.make_request_with_retry("http://test.com")
                except:
                    pass
            
            # Check health metrics
            health_metrics = self.client.get_health_metrics()
            
            assert health_metrics['total_requests'] == 4
            assert health_metrics['successful_requests'] == 3
            assert health_metrics['failed_requests'] == 1
            assert health_metrics['success_rate'] == 0.75
            assert health_metrics['average_response_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])