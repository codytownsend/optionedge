"""
FRED (Federal Reserve Economic Data) API client for economic indicators.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ...data.models.market_data import EconomicIndicator


class FREDClient(BaseAPIClient):
    """
    FRED API client implementation.
    
    API Documentation: https://fred.stlouisfed.org/docs/api/fred/
    Base URL: https://api.stlouisfed.org/fred/
    Authentication: API key parameter
    Rate Limit: 120 requests/minute
    """
    
    def __init__(self, api_key: str):
        rate_limit_config = RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=None,
            burst_allowance=10
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            half_open_max_calls=3
        )
        
        super().__init__(
            base_url="https://api.stlouisfed.org/fred",
            api_key=api_key,
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=30,
            max_retries=3
        )
    
    def authenticate(self) -> Dict[str, str]:
        """Return headers for FRED API (API key in URL params)."""
        return {
            "Accept": "application/json",
            "User-Agent": "OptionsEngine/1.0"
        }
    
    def get_provider_name(self) -> str:
        """Return the name of the data provider."""
        return "FRED"
    
    def health_check(self) -> bool:
        """Perform health check using series info endpoint."""
        try:
            # Try to get info for GDP series
            response = self.get("series", params={
                "series_id": "GDP",
                "api_key": self.api_key,
                "file_type": "json"
            })
            return "seriess" in response
        except Exception:
            return False
    
    def get_series_observations(
        self, 
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[EconomicIndicator]:
        """
        Get observations for a FRED data series.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE', 'FEDFUNDS')
            start_date: Start date for observations
            end_date: End date for observations
            limit: Maximum number of observations to return
            
        Returns:
            List of EconomicIndicator objects
        """
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": str(limit),
                "sort_order": "desc"  # Get most recent first
            }
            
            if start_date:
                params["observation_start"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                params["observation_end"] = end_date.strftime("%Y-%m-%d")
            
            response = self.get("series/observations", params=params)
            
            if "observations" not in response:
                self.logger.warning(f"No observations found for series {series_id}")
                return []
            
            # Get series metadata for name and units
            series_info = self.get_series_info(series_id)
            series_name = series_info.get("title", series_id) if series_info else series_id
            series_units = series_info.get("units", "") if series_info else ""
            frequency = series_info.get("frequency", "") if series_info else ""
            
            indicators = []
            observations = response["observations"]
            
            # Sort by date to calculate changes
            observations.sort(key=lambda x: x["date"])
            
            for i, obs in enumerate(observations):
                if obs["value"] == ".":  # FRED uses "." for missing values
                    continue
                
                try:
                    indicator = EconomicIndicator(
                        indicator_id=series_id,
                        name=series_name,
                        value=Decimal(obs["value"]),
                        date=datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                        frequency=frequency,
                        units=series_units,
                        source="FRED"
                    )
                    
                    # Calculate change metrics if we have previous value
                    if i > 0 and observations[i-1]["value"] != ".":
                        prev_value = Decimal(observations[i-1]["value"])
                        indicator.calculate_change_metrics(prev_value)
                    
                    indicators.append(indicator)
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid observation value for {series_id}: {obs}")
                    continue
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Failed to get observations for series {series_id}: {str(e)}")
            return []
    
    def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata information for a FRED data series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series metadata or None
        """
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
            
            response = self.get("series", params=params)
            
            if "seriess" not in response or not response["seriess"]:
                return None
            
            series_data = response["seriess"][0]
            return {
                "id": series_data.get("id"),
                "title": series_data.get("title"),
                "units": series_data.get("units"),
                "frequency": series_data.get("frequency"),
                "observation_start": series_data.get("observation_start"),
                "observation_end": series_data.get("observation_end"),
                "last_updated": series_data.get("last_updated"),
                "notes": series_data.get("notes")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get series info for {series_id}: {str(e)}")
            return None
    
    def get_latest_observation(self, series_id: str) -> Optional[EconomicIndicator]:
        """
        Get the most recent observation for a FRED data series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            EconomicIndicator object or None
        """
        observations = self.get_series_observations(series_id, limit=2)  # Get 2 to calculate change
        return observations[0] if observations else None
    
    def get_multiple_series_latest(self, series_ids: List[str]) -> Dict[str, EconomicIndicator]:
        """
        Get latest observations for multiple FRED data series.
        
        Args:
            series_ids: List of FRED series IDs
            
        Returns:
            Dictionary mapping series IDs to EconomicIndicator objects
        """
        results = {}
        
        for series_id in series_ids:
            try:
                latest = self.get_latest_observation(series_id)
                if latest:
                    results[series_id] = latest
            except Exception as e:
                self.logger.error(f"Failed to get latest observation for {series_id}: {str(e)}")
                continue
        
        return results
    
    def search_series(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for FRED data series by text.
        
        Args:
            search_text: Search terms
            limit: Maximum number of results
            
        Returns:
            List of series metadata dictionaries
        """
        try:
            params = {
                "search_text": search_text,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": str(limit)
            }
            
            response = self.get("series/search", params=params)
            
            if "seriess" not in response:
                return []
            
            series_list = []
            for series in response["seriess"]:
                series_list.append({
                    "id": series.get("id"),
                    "title": series.get("title"),
                    "units": series.get("units"),
                    "frequency": series.get("frequency"),
                    "popularity": series.get("popularity"),
                    "observation_start": series.get("observation_start"),
                    "observation_end": series.get("observation_end")
                })
            
            return series_list
            
        except Exception as e:
            self.logger.error(f"Failed to search series with text '{search_text}': {str(e)}")
            return []
    
    def get_economic_indicators_summary(self) -> Dict[str, EconomicIndicator]:
        """
        Get a summary of key economic indicators.
        
        Returns:
            Dictionary mapping indicator names to EconomicIndicator objects
        """
        key_indicators = {
            "GDP": "GDP",
            "UNEMPLOYMENT": "UNRATE", 
            "INFLATION": "CPIAUCSL",
            "FED_FUNDS": "FEDFUNDS",
            "10Y_TREASURY": "GS10",
            "VIX": "VIXCLS",
            "SP500": "SP500",
            "DOLLAR_INDEX": "DEXUSEU"
        }
        
        results = {}
        series_data = self.get_multiple_series_latest(list(key_indicators.values()))
        
        for name, series_id in key_indicators.items():
            if series_id in series_data:
                results[name] = series_data[series_id]
        
        return results