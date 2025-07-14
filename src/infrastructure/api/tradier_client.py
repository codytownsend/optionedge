"""
Tradier API client for options chain data and market quotes.
"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, APIError, DataValidator
from ...data.models.options import OptionsChain, OptionQuote, OptionType, Greeks
from ...data.models.market_data import StockQuote

logger = logging.getLogger(__name__)


class TradierClient(BaseAPIClient):
    """Tradier API client for options and stock data."""
    
    def __init__(self, 
                 api_key: str,
                 sandbox: bool = False,
                 **kwargs):
        """
        Initialize Tradier client.
        
        Args:
            api_key: Tradier API key
            sandbox: Use sandbox environment
            **kwargs: Additional arguments for base client
        """
        base_url = "https://sandbox.tradier.com" if sandbox else "https://api.tradier.com"
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            rate_limit_per_minute=120,  # Tradier's rate limit
            **kwargs
        )
        self.sandbox = sandbox
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Tradier API."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def _validate_response(self, response) -> bool:
        """Validate Tradier API response."""
        try:
            data = response.json()
            
            # Check for API errors
            if 'fault' in data:
                logger.error(f"Tradier API fault: {data['fault']}")
                return False
            
            # Check for error messages
            if 'error' in data:
                logger.error(f"Tradier API error: {data['error']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False
    
    def get_options_chain(self, 
                         symbol: str, 
                         expiration: Optional[str] = None,
                         greeks: bool = True) -> OptionsChain:
        """
        Get complete options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration: Specific expiration date (YYYY-MM-DD) or None for all
            greeks: Include Greeks in response
            
        Returns:
            OptionsChain object
            
        Raises:
            APIError: If request fails
        """
        params = {
            'symbol': symbol.upper(),
            'greeks': 'true' if greeks else 'false'
        }
        
        if expiration:
            params['expiration'] = expiration
        
        try:
            response_data = self.get('/v1/markets/options/chains', params=params)
            return self._parse_options_chain(symbol, response_data)
            
        except Exception as e:
            logger.error(f"Failed to get options chain for {symbol}: {e}")
            raise APIError(f"Options chain request failed: {e}")
    
    def get_options_expirations(self, symbol: str) -> List[date]:
        """
        Get available expiration dates for a symbol.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            List of expiration dates
        """
        params = {'symbol': symbol.upper()}
        
        try:
            response_data = self.get('/v1/markets/options/expirations', params=params)
            
            if 'expirations' not in response_data:
                return []
            
            expirations = response_data['expirations']
            if isinstance(expirations, dict) and 'date' in expirations:
                dates = expirations['date']
                if isinstance(dates, str):
                    dates = [dates]
                return [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get expirations for {symbol}: {e}")
            return []
    
    def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time stock quote.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            StockQuote object or None if failed
        """
        params = {'symbols': symbol.upper()}
        
        try:
            response_data = self.get('/v1/markets/quotes', params=params)
            return self._parse_stock_quote(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get stock quote for {symbol}: {e}")
            return None
    
    def get_market_calendar(self) -> Dict[str, Any]:
        """
        Get market calendar information.
        
        Returns:
            Market calendar data
        """
        try:
            return self.get('/v1/markets/calendar')
        except Exception as e:
            logger.error(f"Failed to get market calendar: {e}")
            return {}
    
    def _parse_options_chain(self, symbol: str, data: Dict) -> OptionsChain:
        """Parse Tradier options chain response."""
        chain = OptionsChain(underlying=symbol)
        
        if 'options' not in data or data['options'] is None:
            logger.warning(f"No options data found for {symbol}")
            return chain
        
        options_data = data['options']
        if 'option' not in options_data:
            return chain
        
        # Handle single option vs list of options
        options_list = options_data['option']
        if isinstance(options_list, dict):
            options_list = [options_list]
        
        for option_data in options_list:
            try:
                option_quote = self._parse_option_quote(option_data)
                if option_quote:
                    chain.add_option(option_quote)
            except Exception as e:
                logger.warning(f"Failed to parse option: {e}")
                continue
        
        # Set underlying price if available
        if 'underlying' in data and data['underlying']:
            try:
                chain.underlying_price = Decimal(str(data['underlying']))
            except (ValueError, TypeError):
                pass
        
        return chain
    
    def _parse_option_quote(self, data: Dict) -> Optional[OptionQuote]:
        """Parse individual option quote from Tradier response."""
        try:
            # Extract basic information
            symbol = data.get('symbol', '')
            underlying = data.get('underlying', '')
            
            # Parse strike and expiration
            strike = Decimal(str(data.get('strike', 0)))
            exp_date = datetime.strptime(data.get('expiration_date', ''), '%Y-%m-%d').date()
            
            # Determine option type
            option_type = OptionType.CALL if data.get('option_type', '') == 'call' else OptionType.PUT
            
            # Extract price data
            bid = self._safe_decimal(data.get('bid'))
            ask = self._safe_decimal(data.get('ask'))
            last = self._safe_decimal(data.get('last'))
            
            # Calculate mark price
            mark = None
            if bid is not None and ask is not None:
                mark = (bid + ask) / 2
            elif last is not None:
                mark = last
            
            # Extract volume and open interest
            volume = self._safe_int(data.get('volume'))
            open_interest = self._safe_int(data.get('open_interest'))
            
            # Extract implied volatility
            iv = self._safe_float(data.get('iv'))
            
            # Parse Greeks
            greeks = None
            if any(greek in data for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']):
                greeks = Greeks(
                    delta=self._safe_float(data.get('delta')),
                    gamma=self._safe_float(data.get('gamma')),
                    theta=self._safe_float(data.get('theta')),
                    vega=self._safe_float(data.get('vega')),
                    rho=self._safe_float(data.get('rho'))
                )
            
            # Validate quote data
            quote_data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask
            }
            if not DataValidator.validate_quote_data(quote_data):
                logger.warning(f"Invalid quote data for {symbol}")
                return None
            
            return OptionQuote(
                symbol=symbol,
                underlying=underlying,
                strike=strike,
                expiration=exp_date,
                option_type=option_type,
                bid=bid,
                ask=ask,
                last=last,
                mark=mark,
                volume=volume,
                open_interest=open_interest,
                implied_volatility=iv,
                greeks=greeks,
                quote_time=datetime.utcnow(),
                exchange='TRADIER'
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse option quote: {e}")
            return None
    
    def _parse_stock_quote(self, data: Dict) -> Optional[StockQuote]:
        """Parse stock quote from Tradier response."""
        try:
            if 'quotes' not in data or not data['quotes']:
                return None
            
            quotes = data['quotes']
            if 'quote' not in quotes:
                return None
            
            quote_data = quotes['quote']
            if isinstance(quote_data, list):
                quote_data = quote_data[0]  # Take first quote if multiple
            
            symbol = quote_data.get('symbol', '')
            price = self._safe_decimal(quote_data.get('last'))
            
            if not price:
                return None
            
            return StockQuote(
                symbol=symbol,
                price=price,
                bid=self._safe_decimal(quote_data.get('bid')),
                ask=self._safe_decimal(quote_data.get('ask')),
                open=self._safe_decimal(quote_data.get('open')),
                high=self._safe_decimal(quote_data.get('high')),
                low=self._safe_decimal(quote_data.get('low')),
                previous_close=self._safe_decimal(quote_data.get('prevclose')),
                volume=self._safe_int(quote_data.get('volume')),
                avg_volume=self._safe_int(quote_data.get('average_volume')),
                change=self._safe_decimal(quote_data.get('change')),
                change_percent=self._safe_float(quote_data.get('change_percentage')),
                quote_time=datetime.utcnow(),
                exchange=quote_data.get('exchange', 'TRADIER')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse stock quote: {e}")
            return None
    
    @staticmethod
    def _safe_decimal(value) -> Optional[Decimal]:
        """Safely convert value to Decimal."""
        if value is None or value == '':
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int."""
        if value is None or value == '':
            return None
        try:
            return int(float(value))  # Handle cases where int comes as float
        except (ValueError, TypeError):
            return None
    
    def test_connection(self) -> bool:
        """Test API connection and authentication."""
        try:
            # Try to get market calendar as a simple test
            calendar = self.get_market_calendar()
            return 'calendar' in calendar or 'days' in calendar
        except Exception as e:
            logger.error(f"Tradier connection test failed: {e}")
            return False