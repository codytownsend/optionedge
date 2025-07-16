"""
Market data repository for the Options Trading Engine.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .base import BaseRepository
from ..models.market_data import MarketData, TechnicalIndicators
from ...infrastructure.api import TradierClient, YahooClient, FredClient, QuiverClient
from ...infrastructure.error_handling import RepositoryError, DataError
from ...infrastructure.cache import CacheManager

logger = logging.getLogger(__name__)


class MarketDataRepository(BaseRepository[MarketData]):
    """
    Repository for market data operations.
    
    Handles market data retrieval, caching, and validation from multiple sources.
    """
    
    def __init__(self, 
                 tradier_client: TradierClient,
                 yahoo_client: YahooClient,
                 fred_client: FredClient,
                 quiver_client: QuiverClient,
                 cache_manager: Optional[CacheManager] = None):
        super().__init__(cache_manager)
        self.tradier_client = tradier_client
        self.yahoo_client = yahoo_client
        self.fred_client = fred_client
        self.quiver_client = quiver_client
    
    def get_by_id(self, id: str) -> Optional[MarketData]:
        """Get market data by symbol."""
        return self.get_market_data(id)
    
    def create(self, entity: MarketData) -> MarketData:
        """Create is not supported for market data."""
        raise NotImplementedError("Market data creation not supported")
    
    def update(self, entity: MarketData) -> MarketData:
        """Update is not supported for market data."""
        raise NotImplementedError("Market data update not supported")
    
    def delete(self, id: str) -> bool:
        """Delete market data from cache."""
        cache_key = self._get_cache_key("market_data", id)
        self._invalidate_cache(cache_key)
        return True
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[MarketData]:
        """Find all market data matching filters."""
        symbols = filters.get('symbols', []) if filters else []
        if not symbols:
            return []
        
        results = []
        for symbol in symbols:
            data = self.get_market_data(symbol)
            if data:
                results.append(data)
        
        return self._apply_filters(results, filters or {})
    
    def get_market_data(self, symbol: str, force_refresh: bool = False) -> Optional[MarketData]:
        """
        Get comprehensive market data for a symbol.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force refresh from API
            
        Returns:
            MarketData object or None if not found
        """
        cache_key = self._get_cache_key("market_data", symbol)
        
        # Check cache first
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data and self._is_data_fresh(cached_data.timestamp, 15):
                return cached_data
        
        try:
            # Get real-time quote
            quote_data = self.tradier_client.get_quote(symbol)
            if not quote_data:
                self.logger.warning(f"No quote data found for {symbol}")
                return None
            
            # Get historical data
            historical_data = self.yahoo_client.get_historical_data(
                symbol, 
                period="1y"
            )
            
            # Get technical indicators
            technical_indicators = self._calculate_technical_indicators(
                symbol, 
                historical_data
            )
            
            # Get fundamental data
            fundamental_data = self._get_fundamental_data(symbol)
            
            # Get sentiment data
            sentiment_data = self._get_sentiment_data(symbol)
            
            # Create market data object
            market_data = MarketData(
                symbol=symbol,
                timestamp=self._get_timestamp(),
                price=quote_data.get('last', 0.0),
                volume=quote_data.get('volume', 0),
                open_price=quote_data.get('open', 0.0),
                high_price=quote_data.get('high', 0.0),
                low_price=quote_data.get('low', 0.0),
                close_price=quote_data.get('close', 0.0),
                previous_close=quote_data.get('prevclose', 0.0),
                change=quote_data.get('change', 0.0),
                change_percentage=quote_data.get('change_percentage', 0.0),
                bid=quote_data.get('bid', 0.0),
                ask=quote_data.get('ask', 0.0),
                bid_size=quote_data.get('bidsize', 0),
                ask_size=quote_data.get('asksize', 0),
                market_cap=fundamental_data.get('market_cap', 0),
                pe_ratio=fundamental_data.get('pe_ratio', 0.0),
                dividend_yield=fundamental_data.get('dividend_yield', 0.0),
                beta=fundamental_data.get('beta', 0.0),
                avg_volume=historical_data.get('avg_volume', 0) if historical_data else 0,
                week_52_high=fundamental_data.get('week_52_high', 0.0),
                week_52_low=fundamental_data.get('week_52_low', 0.0),
                technical_indicators=technical_indicators,
                sentiment_score=sentiment_data.get('sentiment_score', 0.0),
                analyst_rating=sentiment_data.get('analyst_rating', 'HOLD'),
                price_target=sentiment_data.get('price_target', 0.0)
            )
            
            # Cache the result
            self._set_to_cache(cache_key, market_data, ttl=900)  # 15 minutes
            
            return market_data
            
        except Exception as e:
            self._handle_repository_error("get_market_data", e)
            return None
    
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1y",
                          interval: str = "1d") -> Optional[List[Dict[str, Any]]]:
        """
        Get historical market data.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            List of historical data points
        """
        cache_key = self._get_cache_key("historical", symbol, period, interval)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data and self._is_data_fresh(cached_data.get('timestamp'), 60):
            return cached_data.get('data', [])
        
        try:
            historical_data = self.yahoo_client.get_historical_data(
                symbol, 
                period=period,
                interval=interval
            )
            
            if historical_data:
                cache_data = {
                    'data': historical_data,
                    'timestamp': self._get_timestamp()
                }
                self._set_to_cache(cache_key, cache_data, ttl=3600)  # 1 hour
                return historical_data
            
            return []
            
        except Exception as e:
            self._handle_repository_error("get_historical_data", e)
            return []
    
    def get_economic_indicators(self, indicators: List[str]) -> Dict[str, Any]:
        """
        Get economic indicators from FRED.
        
        Args:
            indicators: List of FRED series IDs
            
        Returns:
            Dictionary of indicator values
        """
        cache_key = self._get_cache_key("economic_indicators", *indicators)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data and self._is_data_fresh(cached_data.get('timestamp'), 1440):  # 24 hours
            return cached_data.get('data', {})
        
        try:
            economic_data = {}
            for indicator in indicators:
                data = self.fred_client.get_series(indicator)
                if data:
                    economic_data[indicator] = data
            
            if economic_data:
                cache_data = {
                    'data': economic_data,
                    'timestamp': self._get_timestamp()
                }
                self._set_to_cache(cache_key, cache_data, ttl=86400)  # 24 hours
                return economic_data
            
            return {}
            
        except Exception as e:
            self._handle_repository_error("get_economic_indicators", e)
            return {}
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get market sentiment data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of sentiment metrics
        """
        cache_key = self._get_cache_key("sentiment", symbol)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data and self._is_data_fresh(cached_data.get('timestamp'), 60):
            return cached_data.get('data', {})
        
        try:
            sentiment_data = self.quiver_client.get_sentiment_data(symbol)
            
            if sentiment_data:
                cache_data = {
                    'data': sentiment_data,
                    'timestamp': self._get_timestamp()
                }
                self._set_to_cache(cache_key, cache_data, ttl=3600)  # 1 hour
                return sentiment_data
            
            return {}
            
        except Exception as e:
            self._handle_repository_error("get_market_sentiment", e)
            return {}
    
    def _calculate_technical_indicators(self, 
                                      symbol: str, 
                                      historical_data: Optional[List[Dict[str, Any]]]) -> Optional[TechnicalIndicators]:
        """Calculate technical indicators from historical data."""
        if not historical_data:
            return None
        
        try:
            # Extract price data
            closes = [float(data['close']) for data in historical_data]
            volumes = [int(data['volume']) for data in historical_data]
            
            if len(closes) < 50:  # Need sufficient data for indicators
                return None
            
            # Calculate moving averages
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes)
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes)
            
            # Calculate Average True Range
            atr = self._calculate_atr(historical_data)
            
            # Calculate Volume indicators
            volume_sma = sum(volumes[-20:]) / 20
            
            return TechnicalIndicators(
                symbol=symbol,
                timestamp=self._get_timestamp(),
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=self._calculate_ema(closes, 12),
                ema_26=self._calculate_ema(closes, 26),
                rsi=rsi,
                macd_line=macd_line,
                macd_signal=signal_line,
                macd_histogram=histogram,
                bollinger_upper=bb_upper,
                bollinger_middle=bb_middle,
                bollinger_lower=bb_lower,
                atr=atr,
                volume_sma=volume_sma,
                stochastic_k=self._calculate_stochastic_k(historical_data),
                stochastic_d=self._calculate_stochastic_d(historical_data),
                williams_r=self._calculate_williams_r(historical_data)
            )
            
        except Exception as e:
            self.logger.warning(f"Technical indicator calculation failed for {symbol}: {e}")
            return None
    
    def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for symbol."""
        try:
            return self.yahoo_client.get_fundamental_data(symbol) or {}
        except Exception as e:
            self.logger.warning(f"Fundamental data retrieval failed for {symbol}: {e}")
            return {}
    
    def _get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment data for symbol."""
        try:
            return self.quiver_client.get_sentiment_data(symbol) or {}
        except Exception as e:
            self.logger.warning(f"Sentiment data retrieval failed for {symbol}: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-day EMA of MACD)
        signal_line = macd_line  # Simplified for now
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> tuple:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            avg = sum(prices) / len(prices)
            return avg, avg, avg
        
        sma = sum(prices[-period:]) / period
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
        std_dev = variance ** 0.5
        
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, historical_data: List[Dict[str, Any]], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(historical_data) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(historical_data)):
            high = float(historical_data[i]['high'])
            low = float(historical_data[i]['low'])
            prev_close = float(historical_data[i-1]['close'])
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges)
        
        return sum(true_ranges[-period:]) / period
    
    def _calculate_stochastic_k(self, historical_data: List[Dict[str, Any]], period: int = 14) -> float:
        """Calculate Stochastic %K."""
        if len(historical_data) < period:
            return 50.0
        
        recent_data = historical_data[-period:]
        current_close = float(historical_data[-1]['close'])
        lowest_low = min(float(data['low']) for data in recent_data)
        highest_high = max(float(data['high']) for data in recent_data)
        
        if highest_high == lowest_low:
            return 50.0
        
        stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        return stoch_k
    
    def _calculate_stochastic_d(self, historical_data: List[Dict[str, Any]], period: int = 14) -> float:
        """Calculate Stochastic %D (3-day SMA of %K)."""
        if len(historical_data) < period + 2:
            return 50.0
        
        k_values = []
        for i in range(3):
            data_subset = historical_data[-(period + i):-(i if i > 0 else None)]
            k_value = self._calculate_stochastic_k(data_subset, period)
            k_values.append(k_value)
        
        return sum(k_values) / len(k_values)
    
    def _calculate_williams_r(self, historical_data: List[Dict[str, Any]], period: int = 14) -> float:
        """Calculate Williams %R."""
        if len(historical_data) < period:
            return -50.0
        
        recent_data = historical_data[-period:]
        current_close = float(historical_data[-1]['close'])
        lowest_low = min(float(data['low']) for data in recent_data)
        highest_high = max(float(data['high']) for data in recent_data)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r