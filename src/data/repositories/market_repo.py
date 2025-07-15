"""
Market data repository for managing stock quotes, fundamentals, and technical indicators.
"""

from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from decimal import Decimal

from .base import BaseRepository, RepositoryError, DataNotFoundError
from ..models.market_data import (
    StockQuote, FundamentalData, TechnicalIndicators, 
    EconomicIndicator, SentimentData, ETFFlowData, OHLCVData
)
from ...infrastructure.api.yahoo_client import YahooFinanceClient
from ...infrastructure.api.fred_client import FREDClient
from ...infrastructure.cache.base_cache import CacheInterface


class MarketDataRepository(BaseRepository[StockQuote]):
    """Repository for market data with multiple data source integration."""
    
    def __init__(self,
                 yahoo_client: YahooFinanceClient,
                 fred_client: Optional[FREDClient] = None,
                 cache: Optional[CacheInterface] = None,
                 cache_ttl_seconds: int = 300):
        """
        Initialize market data repository.
        
        Args:
            yahoo_client: Yahoo Finance API client
            fred_client: FRED API client for economic data
            cache: Cache implementation
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(cache_ttl_seconds)
        self.yahoo_client = yahoo_client
        self.fred_client = fred_client
        self.cache = cache
    
    def get_by_id(self, entity_id: str) -> Optional[StockQuote]:
        """Get stock quote by symbol."""
        return self.get_stock_quote(entity_id)
    
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[StockQuote]:
        """Get stock quotes for multiple symbols."""
        if not filters or 'symbols' not in filters:
            raise ValueError("Must provide 'symbols' filter for get_all")
        
        symbols = filters['symbols']
        quotes = []
        
        for symbol in symbols:
            try:
                quote = self.get_stock_quote(symbol)
                if quote:
                    quotes.append(quote)
            except Exception as e:
                self.logger.warning(f"Failed to get quote for {symbol}: {e}")
                continue
        
        return quotes
    
    def save(self, entity: StockQuote) -> StockQuote:
        """Save stock quote to cache."""
        if not self.validate_entity(entity):
            raise RepositoryError("Invalid stock quote entity")
        
        cache_key = f"stock_quote:{entity.symbol}"
        
        if self.cache:
            self.cache.set(cache_key, entity, ttl=self.cache_ttl_seconds)
        else:
            self._set_cache(cache_key, entity)
        
        return entity
    
    def delete(self, entity_id: str) -> bool:
        """Delete stock quote from cache."""
        cache_key = f"stock_quote:{entity_id}"
        
        if self.cache:
            return self.cache.delete(cache_key)
        else:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False
    
    def get_stock_quote(self, 
                       symbol: str,
                       force_refresh: bool = False) -> Optional[StockQuote]:
        """
        Get real-time stock quote.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force API call even if cached
            
        Returns:
            StockQuote object or None if not found
        """
        cache_key = f"stock_quote:{symbol}"
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_quote = self._get_from_cache(cache_key)
            if cached_quote:
                return cached_quote
            
            if self.cache:
                cached_quote = self.cache.get(cache_key)
                if cached_quote:
                    return cached_quote
        
        # Fetch from API
        try:
            # Try Yahoo Finance first
            stock_info = self.yahoo_client.get_stock_info(symbol)
            if stock_info:
                quote = self._parse_yahoo_quote(symbol, stock_info)
                if quote:
                    # Cache the result
                    self._set_cache(cache_key, quote)
                    if self.cache:
                        self.cache.set(cache_key, quote, ttl=self.cache_ttl_seconds)
                    
                    self.logger.debug(f"Retrieved stock quote for {symbol}")
                    return quote
            
            self.logger.warning(f"No stock quote found for {symbol}")
            return None
            
        except Exception as e:
            self._handle_error(f"get_stock_quote({symbol})", e)
            return None
    
    def get_fundamental_data(self, 
                           symbol: str,
                           force_refresh: bool = False) -> Optional[FundamentalData]:
        """
        Get fundamental financial data.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force API call even if cached
            
        Returns:
            FundamentalData object or None if not found
        """
        cache_key = f"fundamentals:{symbol}"
        
        # Check cache first
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data
            
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    return cached_data
        
        # Fetch from API
        try:
            fundamental_data = self.yahoo_client.get_fundamental_data(symbol)
            
            if fundamental_data:
                # Cache with longer TTL since fundamentals change less frequently
                longer_ttl = self.cache_ttl_seconds * 4  # 20 minutes default
                self._set_cache(cache_key, fundamental_data)
                if self.cache:
                    self.cache.set(cache_key, fundamental_data, ttl=longer_ttl)
                
                self.logger.debug(f"Retrieved fundamental data for {symbol}")
                return fundamental_data
            
            return None
            
        except Exception as e:
            self._handle_error(f"get_fundamental_data({symbol})", e)
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
            List of OHLCV data points
        """
        cache_key = f"historical:{symbol}:{period}:{interval}"
        
        # Check cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        # Fetch from API
        try:
            historical_data = self.yahoo_client.get_historical_data(symbol, period, interval)
            
            if historical_data:
                # Cache with longer TTL for historical data
                longer_ttl = self.cache_ttl_seconds * 12  # 1 hour default
                self._set_cache(cache_key, historical_data)
                if self.cache:
                    self.cache.set(cache_key, historical_data, ttl=longer_ttl)
                
                self.logger.debug(f"Retrieved {len(historical_data)} historical data points for {symbol}")
                return historical_data
            
            return []
            
        except Exception as e:
            self._handle_error(f"get_historical_data({symbol})", e)
            return []
    
    def get_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """Calculate technical indicators from historical data."""
        cache_key = f"technicals:{symbol}"
        
        # Check cache
        cached_indicators = self._get_from_cache(cache_key)
        if cached_indicators:
            return cached_indicators
        
        # Calculate from historical data
        try:
            # Get historical data for calculations
            historical_data = self.get_historical_data(symbol, period="1y", interval="1d")
            
            if not historical_data:
                return None
            
            # Calculate indicators (simplified implementation)
            indicators = self._calculate_technical_indicators(symbol, historical_data)
            
            if indicators:
                # Cache with medium TTL
                medium_ttl = self.cache_ttl_seconds * 2  # 10 minutes default
                self._set_cache(cache_key, indicators)
                if self.cache:
                    self.cache.set(cache_key, indicators, ttl=medium_ttl)
                
                return indicators
            
            return None
            
        except Exception as e:
            self._handle_error(f"get_technical_indicators({symbol})", e)
            return None
    
    def get_economic_indicator(self, 
                             indicator_id: str,
                             force_refresh: bool = False) -> Optional[EconomicIndicator]:
        """
        Get economic indicator data.
        
        Args:
            indicator_id: FRED series ID
            force_refresh: Force API call even if cached
            
        Returns:
            Latest EconomicIndicator or None
        """
        if not self.fred_client:
            self.logger.warning("FRED client not available for economic indicators")
            return None
        
        cache_key = f"economic:{indicator_id}"
        
        # Check cache
        if not force_refresh:
            cached_indicator = self._get_from_cache(cache_key)
            if cached_indicator:
                return cached_indicator
        
        # Fetch from FRED API
        try:
            indicator = self.fred_client.get_latest_value(indicator_id)
            
            if indicator:
                # Cache with longer TTL since economic data updates less frequently
                longer_ttl = self.cache_ttl_seconds * 24  # 2 hours default
                self._set_cache(cache_key, indicator)
                if self.cache:
                    self.cache.set(cache_key, indicator, ttl=longer_ttl)
                
                self.logger.debug(f"Retrieved economic indicator {indicator_id}")
                return indicator
            
            return None
            
        except Exception as e:
            self._handle_error(f"get_economic_indicator({indicator_id})", e)
            return None
    
    def get_economic_snapshot(self) -> Dict[str, Optional[EconomicIndicator]]:
        """Get snapshot of key economic indicators."""
        if not self.fred_client:
            return {}
        
        cache_key = "economic_snapshot"
        
        # Check cache
        cached_snapshot = self._get_from_cache(cache_key)
        if cached_snapshot:
            return cached_snapshot
        
        # Fetch from FRED
        try:
            snapshot = self.fred_client.get_economic_snapshot()
            
            if snapshot:
                # Cache for longer since economic data doesn't change frequently
                longer_ttl = self.cache_ttl_seconds * 24  # 2 hours default
                self._set_cache(cache_key, snapshot)
                if self.cache:
                    self.cache.set(cache_key, snapshot, ttl=longer_ttl)
                
                self.logger.debug("Retrieved economic snapshot")
                return snapshot
            
            return {}
            
        except Exception as e:
            self._handle_error("get_economic_snapshot", e)
            return {}
    
    def _parse_yahoo_quote(self, symbol: str, stock_info: Dict[str, Any]) -> Optional[StockQuote]:
        """Parse Yahoo Finance stock info into StockQuote."""
        try:
            if not stock_info:
                return None
            
            # Extract price data (Yahoo Finance API structure varies)
            price = stock_info.get('regularMarketPrice') or stock_info.get('price')
            
            if not price:
                return None
            
            quote = StockQuote(
                symbol=symbol.upper(),
                price=Decimal(str(price)),
                bid=self._safe_decimal(stock_info.get('bid')),
                ask=self._safe_decimal(stock_info.get('ask')),
                open=self._safe_decimal(stock_info.get('regularMarketOpen')),
                high=self._safe_decimal(stock_info.get('regularMarketDayHigh')),
                low=self._safe_decimal(stock_info.get('regularMarketDayLow')),
                previous_close=self._safe_decimal(stock_info.get('regularMarketPreviousClose')),
                volume=self._safe_int(stock_info.get('regularMarketVolume')),
                avg_volume=self._safe_int(stock_info.get('averageDailyVolume3Month')),
                change=self._safe_decimal(stock_info.get('regularMarketChange')),
                change_percent=self._safe_float(stock_info.get('regularMarketChangePercent')),
                quote_time=datetime.utcnow(),
                exchange=stock_info.get('exchange', 'YAHOO')
            )
            
            return quote
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Yahoo quote for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, 
                                      symbol: str, 
                                      historical_data: List[OHLCVData]) -> Optional[TechnicalIndicators]:
        """Calculate technical indicators from historical data."""
        try:
            if len(historical_data) < 200:  # Need enough data for 200-day MA
                self.logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Sort by timestamp
            sorted_data = sorted(historical_data, key=lambda x: x.timestamp)
            
            # Extract closing prices
            closes = [float(bar.close) for bar in sorted_data]
            
            # Calculate simple moving averages
            sma_20 = self._calculate_sma(closes, 20) if len(closes) >= 20 else None
            sma_50 = self._calculate_sma(closes, 50) if len(closes) >= 50 else None
            sma_100 = self._calculate_sma(closes, 100) if len(closes) >= 100 else None
            sma_200 = self._calculate_sma(closes, 200) if len(closes) >= 200 else None
            
            # Calculate RSI
            rsi_14 = self._calculate_rsi(closes, 14) if len(closes) >= 14 else None
            
            # Calculate momentum
            momentum_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else None
            momentum_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else None
            momentum_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else None
            
            # Calculate historical volatility
            hv_30 = self._calculate_historical_volatility(closes, 30) if len(closes) >= 30 else None
            
            indicators = TechnicalIndicators(
                symbol=symbol.upper(),
                timestamp=datetime.utcnow(),
                sma_20=Decimal(str(sma_20)) if sma_20 else None,
                sma_50=Decimal(str(sma_50)) if sma_50 else None,
                sma_100=Decimal(str(sma_100)) if sma_100 else None,
                sma_200=Decimal(str(sma_200)) if sma_200 else None,
                rsi_14=rsi_14,
                hv_30=hv_30,
                momentum_1d=momentum_1d,
                momentum_5d=momentum_5d,
                momentum_20d=momentum_20d
            )
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate technical indicators for {symbol}: {e}")
            return None
    
    def _calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(prices) <= period:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        if len(changes) < period:
            return None
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes[-period:]]
        losses = [-change if change < 0 else 0 for change in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_historical_volatility(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate historical volatility (annualized)."""
        if len(prices) <= period:
            return None
        
        import math
        
        # Calculate daily returns
        returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        recent_returns = returns[-period:]
        
        if not recent_returns:
            return None
        
        # Calculate standard deviation
        mean_return = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
        daily_vol = math.sqrt(variance)
        
        # Annualize (252 trading days)
        annual_vol = daily_vol * math.sqrt(252) * 100
        
        return annual_vol
    
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
    
    def validate_entity(self, entity: StockQuote) -> bool:
        """Validate stock quote entity."""
        if not super().validate_entity(entity):
            return False
        
        if not entity.symbol:
            self.logger.error("Stock quote missing symbol")
            return False
        
        if entity.price <= 0:
            self.logger.error("Stock quote has invalid price")
            return False
        
        return True