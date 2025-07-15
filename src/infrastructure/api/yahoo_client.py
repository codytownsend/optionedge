"""
Yahoo Finance API client for fundamental data and market information.
"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, APIError, DataValidator
from ...data.models.market_data import FundamentalData, StockQuote, TechnicalIndicators, OHLCVData

logger = logging.getLogger(__name__)


class YahooFinanceClient(BaseAPIClient):
    """Yahoo Finance API client via RapidAPI."""
    
    def __init__(self, 
                 rapidapi_key: str,
                 **kwargs):
        """
        Initialize Yahoo Finance client.
        
        Args:
            rapidapi_key: RapidAPI key for Yahoo Finance API
            **kwargs: Additional arguments for base client
        """
        super().__init__(
            base_url="https://yahoo-finance15.p.rapidapi.com",
            api_key=rapidapi_key,
            rate_limit_per_minute=500,  # RapidAPI rate limit
            **kwargs
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Yahoo Finance API."""
        return {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'yahoo-finance15.p.rapidapi.com',
            'Accept': 'application/json'
        }
    
    def _validate_response(self, response) -> bool:
        """Validate Yahoo Finance API response."""
        try:
            data = response.json()
            
            # Check for API errors
            if isinstance(data, dict):
                if 'error' in data:
                    logger.error(f"Yahoo Finance API error: {data['error']}")
                    return False
                
                if 'message' in data and 'error' in data['message'].lower():
                    logger.error(f"Yahoo Finance API message: {data['message']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock information dictionary
        """
        params = {'symbol': symbol.upper()}
        
        try:
            response_data = self.get('/api/yahoo/qu/quote', params=params)
            return self._parse_stock_info(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get stock info for {symbol}: {e}")
            return None
    
    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """
        Get fundamental financial data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            FundamentalData object or None
        """
        params = {'symbol': symbol.upper()}
        
        try:
            # Get financial data
            financial_data = self.get('/api/yahoo/fi/financials', params=params)
            
            # Get key statistics
            stats_data = self.get('/api/yahoo/qu/quote', params=params)
            
            return self._parse_fundamental_data(symbol, financial_data, stats_data)
            
        except Exception as e:
            logger.error(f"Failed to get fundamental data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, 
                           symbol: str, 
                           period: str = "1y",
                           interval: str = "1d") -> List[OHLCVData]:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            List of OHLCV data
        """
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'interval': interval
        }
        
        try:
            response_data = self.get('/api/yahoo/hi/history', params=params)
            return self._parse_historical_data(symbol, response_data, interval)
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    def get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get options data (alternative to Tradier).
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            Options data dictionary
        """
        params = {'symbol': symbol.upper()}
        
        try:
            response_data = self.get('/api/yahoo/op/option', params=params)
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to get options data for {symbol}: {e}")
            return None
    
    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for stock symbols.
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols
        """
        params = {'q': query}
        
        try:
            response_data = self.get('/api/yahoo/se/search', params=params)
            return self._parse_search_results(response_data)
            
        except Exception as e:
            logger.error(f"Failed to search symbols for '{query}': {e}")
            return []
    
    def _parse_stock_info(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Parse stock information response."""
        try:
            if not data or 'body' not in data:
                return None
            
            body = data['body']
            if isinstance(body, list) and len(body) > 0:
                return body[0]
            elif isinstance(body, dict):
                return body
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse stock info: {e}")
            return None
    
    def _parse_fundamental_data(self, 
                               symbol: str, 
                               financial_data: Dict, 
                               stats_data: Dict) -> Optional[FundamentalData]:
        """Parse fundamental data from multiple API responses."""
        try:
            # Extract current date as report date
            report_date = date.today()
            
            # Parse financial metrics
            financials = financial_data.get('body', {}) if financial_data else {}
            stats = self._parse_stock_info(stats_data) if stats_data else {}
            
            # Extract income statement data
            revenue = self._safe_int(financials.get('totalRevenue'))
            net_income = self._safe_int(financials.get('netIncome'))
            gross_profit = self._safe_int(financials.get('grossProfit'))
            operating_income = self._safe_int(financials.get('operatingIncome'))
            
            # Extract per-share data
            eps = self._safe_decimal(stats.get('epsTrailingTwelveMonths'))
            eps_diluted = self._safe_decimal(stats.get('epsForward'))
            
            # Extract balance sheet data
            total_assets = self._safe_int(financials.get('totalAssets'))
            total_debt = self._safe_int(financials.get('totalDebt'))
            shareholders_equity = self._safe_int(financials.get('totalStockholderEquity'))
            cash = self._safe_int(financials.get('totalCash'))
            
            # Extract cash flow data
            operating_cash_flow = self._safe_int(financials.get('operatingCashflow'))
            free_cash_flow = self._safe_int(financials.get('freeCashflow'))
            
            # Extract ratios
            pe_ratio = self._safe_float(stats.get('trailingPE'))
            peg_ratio = self._safe_float(stats.get('pegRatio'))
            price_to_book = self._safe_float(stats.get('priceToBook'))
            
            # Calculate margins if possible
            gross_margin = None
            operating_margin = None
            net_margin = None
            
            if revenue and revenue > 0:
                if gross_profit:
                    gross_margin = float(gross_profit / revenue)
                if operating_income:
                    operating_margin = float(operating_income / revenue)
                if net_income:
                    net_margin = float(net_income / revenue)
            
            return FundamentalData(
                symbol=symbol.upper(),
                report_date=report_date,
                period_type="trailing_twelve_months",
                revenue=revenue,
                gross_profit=gross_profit,
                operating_income=operating_income,
                net_income=net_income,
                eps=eps,
                eps_diluted=eps_diluted,
                total_assets=total_assets,
                total_debt=total_debt,
                shareholders_equity=shareholders_equity,
                cash_and_equivalents=cash,
                operating_cash_flow=operating_cash_flow,
                free_cash_flow=free_cash_flow,
                pe_ratio=pe_ratio,
                peg_ratio=peg_ratio,
                price_to_book=price_to_book,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse fundamental data: {e}")
            return None
    
    def _parse_historical_data(self, 
                              symbol: str, 
                              data: Dict, 
                              interval: str) -> List[OHLCVData]:
        """Parse historical OHLCV data."""
        ohlcv_list = []
        
        try:
            if not data or 'body' not in data:
                return ohlcv_list
            
            body = data['body']
            if not isinstance(body, list):
                return ohlcv_list
            
            for bar_data in body:
                try:
                    # Parse timestamp
                    timestamp_str = bar_data.get('date', '')
                    if not timestamp_str:
                        continue
                    
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # Parse OHLCV values
                    open_price = self._safe_decimal(bar_data.get('open'))
                    high_price = self._safe_decimal(bar_data.get('high'))
                    low_price = self._safe_decimal(bar_data.get('low'))
                    close_price = self._safe_decimal(bar_data.get('close'))
                    volume = self._safe_int(bar_data.get('volume'))
                    
                    if all(v is not None for v in [open_price, high_price, low_price, close_price, volume]):
                        ohlcv = OHLCVData(
                            symbol=symbol.upper(),
                            timestamp=timestamp,
                            timeframe=interval,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume
                        )
                        ohlcv_list.append(ohlcv)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse OHLCV bar: {e}")
                    continue
            
            return sorted(ohlcv_list, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.warning(f"Failed to parse historical data: {e}")
            return ohlcv_list
    
    def _parse_search_results(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse symbol search results."""
        results = []
        
        try:
            if not data or 'body' not in data:
                return results
            
            body = data['body']
            if isinstance(body, list):
                for item in body:
                    if isinstance(item, dict) and 'symbol' in item:
                        results.append({
                            'symbol': item.get('symbol', ''),
                            'name': item.get('shortname', item.get('longname', '')),
                            'exchange': item.get('exchange', ''),
                            'type': item.get('quoteType', '')
                        })
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to parse search results: {e}")
            return results
    
    @staticmethod
    def _safe_decimal(value) -> Optional[Decimal]:
        """Safely convert value to Decimal."""
        if value is None or value == '' or value == 'N/A':
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, ArithmeticError):
            return None
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '' or value == 'N/A':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int."""
        if value is None or value == '' or value == 'N/A':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple symbol search
            results = self.search_symbols("AAPL")
            return len(results) > 0
        except Exception as e:
            logger.error(f"Yahoo Finance connection test failed: {e}")
            return False