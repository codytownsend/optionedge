"""
FRED (Federal Reserve Economic Data) API client for macroeconomic indicators.
"""

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, APIError
from ...data.models.market_data import EconomicIndicator

logger = logging.getLogger(__name__)


class FREDClient(BaseAPIClient):
    """FRED API client for economic data."""
    
    # Common economic indicators
    ECONOMIC_SERIES = {
        'GDP': 'GDP',
        'CPI': 'CPIAUCSL',
        'UNEMPLOYMENT': 'UNRATE',
        'FEDERAL_FUNDS_RATE': 'FEDFUNDS',
        'TEN_YEAR_TREASURY': 'GS10',
        'RETAIL_SALES': 'RSXFS',
        'INDUSTRIAL_PRODUCTION': 'INDPRO',
        'HOUSING_STARTS': 'HOUST',
        'CONSUMER_SENTIMENT': 'UMCSENT',
        'INFLATION_EXPECTATIONS': 'T5YIE'
    }
    
    def __init__(self, 
                 api_key: str,
                 **kwargs):
        """
        Initialize FRED client.
        
        Args:
            api_key: FRED API key
            **kwargs: Additional arguments for base client
        """
        super().__init__(
            base_url="https://api.stlouisfed.org/fred",
            api_key=api_key,
            rate_limit_per_minute=120,  # FRED rate limit
            **kwargs
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for FRED API."""
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def _validate_response(self, response) -> bool:
        """Validate FRED API response."""
        try:
            data = response.json()
            
            # Check for FRED API errors
            if 'error_code' in data:
                logger.error(f"FRED API error {data['error_code']}: {data.get('error_message', 'Unknown error')}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False
    
    def get_series_data(self, 
                       series_id: str,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None,
                       limit: int = 1000) -> List[EconomicIndicator]:
        """
        Get time series data for an economic indicator.
        
        Args:
            series_id: FRED series ID
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of observations
            
        Returns:
            List of EconomicIndicator objects
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit
        }
        
        if start_date:
            params['observation_start'] = start_date.strftime('%Y-%m-%d')
        
        if end_date:
            params['observation_end'] = end_date.strftime('%Y-%m-%d')
        
        try:
            response_data = self.get('/series/observations', params=params)
            return self._parse_series_data(series_id, response_data)
            
        except Exception as e:
            logger.error(f"Failed to get series data for {series_id}: {e}")
            return []
    
    def get_latest_value(self, series_id: str) -> Optional[EconomicIndicator]:
        """
        Get the latest value for an economic indicator.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Latest EconomicIndicator or None
        """
        # Get last 1 observation
        data = self.get_series_data(series_id, limit=1)
        return data[0] if data else None
    
    def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Series metadata dictionary
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response_data = self.get('/series', params=params)
            return self._parse_series_info(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get series info for {series_id}: {e}")
            return None
    
    def get_economic_snapshot(self) -> Dict[str, Optional[EconomicIndicator]]:
        """
        Get latest values for key economic indicators.
        
        Returns:
            Dictionary of indicator names to latest values
        """
        snapshot = {}
        
        for name, series_id in self.ECONOMIC_SERIES.items():
            try:
                latest = self.get_latest_value(series_id)
                snapshot[name] = latest
            except Exception as e:
                logger.warning(f"Failed to get {name} ({series_id}): {e}")
                snapshot[name] = None
        
        return snapshot
    
    def search_series(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for economic data series.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of series information
        """
        params = {
            'search_text': search_text,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit
        }
        
        try:
            response_data = self.get('/series/search', params=params)
            return self._parse_search_results(response_data)
            
        except Exception as e:
            logger.error(f"Failed to search series for '{search_text}': {e}")
            return []
    
    def _parse_series_data(self, series_id: str, data: Dict) -> List[EconomicIndicator]:
        """Parse FRED series observations."""
        indicators = []
        
        try:
            if 'observations' not in data:
                return indicators
            
            observations = data['observations']
            if not isinstance(observations, list):
                return indicators
            
            # Get series info for metadata
            series_info = self.get_series_info(series_id) or {}
            series_title = series_info.get('title', series_id)
            series_units = series_info.get('units', '')
            frequency = series_info.get('frequency', 'Unknown')
            
            for obs in observations:
                try:
                    # Parse date
                    obs_date = datetime.strptime(obs['date'], '%Y-%m-%d').date()
                    
                    # Parse value (skip if missing or '.')
                    value_str = obs.get('value', '.')
                    if value_str == '.' or value_str is None:
                        continue
                    
                    value = Decimal(str(value_str))
                    
                    indicator = EconomicIndicator(
                        indicator_id=series_id,
                        name=series_title,
                        value=value,
                        date=obs_date,
                        frequency=frequency.lower(),
                        units=series_units,
                        source='FRED'
                    )
                    
                    indicators.append(indicator)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse observation: {e}")
                    continue
            
            # Sort by date and calculate period-over-period changes
            indicators.sort(key=lambda x: x.date)
            
            for i in range(1, len(indicators)):
                current = indicators[i]
                previous = indicators[i-1]
                
                # Calculate change
                change = current.value - previous.value
                change_percent = float(change / previous.value * 100) if previous.value != 0 else None
                
                # Update current indicator with change data
                indicators[i] = current.copy(update={
                    'previous_value': previous.value,
                    'change': change,
                    'change_percent': change_percent
                })
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Failed to parse series data: {e}")
            return indicators
    
    def _parse_series_info(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Parse FRED series metadata."""
        try:
            if 'seriess' not in data or not data['seriess']:
                return None
            
            series = data['seriess'][0]  # Take first series
            
            return {
                'id': series.get('id', ''),
                'title': series.get('title', ''),
                'observation_start': series.get('observation_start', ''),
                'observation_end': series.get('observation_end', ''),
                'frequency': series.get('frequency', ''),
                'frequency_short': series.get('frequency_short', ''),
                'units': series.get('units', ''),
                'units_short': series.get('units_short', ''),
                'seasonal_adjustment': series.get('seasonal_adjustment', ''),
                'seasonal_adjustment_short': series.get('seasonal_adjustment_short', ''),
                'last_updated': series.get('last_updated', ''),
                'popularity': series.get('popularity', 0),
                'notes': series.get('notes', '')
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse series info: {e}")
            return None
    
    def _parse_search_results(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse FRED series search results."""
        results = []
        
        try:
            if 'seriess' not in data:
                return results
            
            for series in data['seriess']:
                result = {
                    'id': series.get('id', ''),
                    'title': series.get('title', ''),
                    'units': series.get('units', ''),
                    'frequency': series.get('frequency', ''),
                    'observation_start': series.get('observation_start', ''),
                    'observation_end': series.get('observation_end', ''),
                    'popularity': series.get('popularity', 0),
                    'last_updated': series.get('last_updated', '')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to parse search results: {e}")
            return results
    
    def test_connection(self) -> bool:
        """Test FRED API connection."""
        try:
            # Test with GDP series
            gdp_info = self.get_series_info('GDP')
            return gdp_info is not None
        except Exception as e:
            logger.error(f"FRED connection test failed: {e}")
            return False