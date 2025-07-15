"""
Options data repository for managing options chain data and quotes.
"""

from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from decimal import Decimal

from .base import BaseRepository, RepositoryError, DataNotFoundError
from ..models.options import OptionsChain, OptionQuote, OptionType
from ...infrastructure.api.tradier_client import TradierClient
from ...infrastructure.cache.base_cache import CacheInterface


class OptionsRepository(BaseRepository[OptionsChain]):
    """Repository for options chain data with caching and API integration."""
    
    def __init__(self, 
                 tradier_client: TradierClient,
                 cache: Optional[CacheInterface] = None,
                 cache_ttl_seconds: int = 300):
        """
        Initialize options repository.
        
        Args:
            tradier_client: Tradier API client
            cache: Cache implementation
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(cache_ttl_seconds)
        self.tradier_client = tradier_client
        self.cache = cache
    
    def get_by_id(self, entity_id: str) -> Optional[OptionsChain]:
        """Get options chain by underlying symbol."""
        return self.get_options_chain(entity_id)
    
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[OptionsChain]:
        """Get options chains for multiple symbols."""
        if not filters or 'symbols' not in filters:
            raise ValueError("Must provide 'symbols' filter for get_all")
        
        symbols = filters['symbols']
        chains = []
        
        for symbol in symbols:
            try:
                chain = self.get_options_chain(symbol)
                if chain:
                    chains.append(chain)
            except Exception as e:
                self.logger.warning(f"Failed to get options chain for {symbol}: {e}")
                continue
        
        return chains
    
    def save(self, entity: OptionsChain) -> OptionsChain:
        """Save options chain to cache."""
        if not self.validate_entity(entity):
            raise RepositoryError("Invalid options chain entity")
        
        cache_key = f"options_chain:{entity.underlying}"
        
        if self.cache:
            self.cache.set(cache_key, entity, ttl=self.cache_ttl_seconds)
        else:
            self._set_cache(cache_key, entity)
        
        return entity
    
    def delete(self, entity_id: str) -> bool:
        """Delete options chain from cache."""
        cache_key = f"options_chain:{entity_id}"
        
        if self.cache:
            return self.cache.delete(cache_key)
        else:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False
    
    def get_options_chain(self, 
                         symbol: str, 
                         expiration: Optional[str] = None,
                         force_refresh: bool = False) -> Optional[OptionsChain]:
        """
        Get complete options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration: Specific expiration date (YYYY-MM-DD)
            force_refresh: Force API call even if cached
            
        Returns:
            OptionsChain object or None if not found
        """
        cache_key = f"options_chain:{symbol}"
        if expiration:
            cache_key += f":{expiration}"
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_chain = self._get_from_cache(cache_key)
            if cached_chain:
                return cached_chain
            
            # Also check external cache
            if self.cache:
                cached_chain = self.cache.get(cache_key)
                if cached_chain:
                    return cached_chain
        
        # Fetch from API
        try:
            chain = self.tradier_client.get_options_chain(
                symbol=symbol,
                expiration=expiration,
                greeks=True
            )
            
            if chain:
                # Cache the result
                self._set_cache(cache_key, chain)
                if self.cache:
                    self.cache.set(cache_key, chain, ttl=self.cache_ttl_seconds)
                
                self.logger.info(f"Retrieved options chain for {symbol} with {len(chain.options)} options")
                return chain
            else:
                self.logger.warning(f"No options chain found for {symbol}")
                return None
                
        except Exception as e:
            self._handle_error(f"get_options_chain({symbol})", e)
            return None
    
    def get_option_quote(self, 
                        symbol: str,
                        strike: Decimal,
                        expiration: date,
                        option_type: OptionType,
                        force_refresh: bool = False) -> Optional[OptionQuote]:
        """
        Get specific option quote.
        
        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiration: Expiration date
            option_type: Call or Put
            force_refresh: Force refresh from API
            
        Returns:
            OptionQuote or None if not found
        """
        # First try to get from cached chain
        chain = self.get_options_chain(symbol, force_refresh=force_refresh)
        
        if chain:
            option = chain.get_option_by_strike(strike, option_type, expiration)
            if option:
                return option
        
        self.logger.warning(f"Option not found: {symbol} {strike} {option_type.value} {expiration}")
        return None
    
    def get_liquid_options(self, 
                          symbol: str,
                          min_volume: int = 10,
                          min_open_interest: int = 100,
                          max_spread_pct: float = 0.5) -> List[OptionQuote]:
        """
        Get liquid options for a symbol.
        
        Args:
            symbol: Underlying symbol
            min_volume: Minimum daily volume
            min_open_interest: Minimum open interest
            max_spread_pct: Maximum bid-ask spread percentage
            
        Returns:
            List of liquid option quotes
        """
        chain = self.get_options_chain(symbol)
        
        if not chain:
            return []
        
        return chain.filter_by_liquidity(
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct
        )
    
    def get_options_by_expiration(self, 
                                 symbol: str,
                                 expiration: date) -> List[OptionQuote]:
        """Get all options for a specific expiration."""
        chain = self.get_options_chain(symbol)
        
        if not chain:
            return []
        
        return [opt for opt in chain.options if opt.expiration == expiration]
    
    def get_options_in_range(self,
                           symbol: str,
                           min_strike: Decimal,
                           max_strike: Decimal,
                           option_type: Optional[OptionType] = None) -> List[OptionQuote]:
        """
        Get options within a strike range.
        
        Args:
            symbol: Underlying symbol
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            option_type: Filter by option type (optional)
            
        Returns:
            List of options in range
        """
        chain = self.get_options_chain(symbol)
        
        if not chain:
            return []
        
        filtered_options = []
        
        for option in chain.options:
            # Check strike range
            if not (min_strike <= option.strike <= max_strike):
                continue
            
            # Check option type if specified
            if option_type and option.option_type != option_type:
                continue
            
            filtered_options.append(option)
        
        return sorted(filtered_options, key=lambda x: (x.expiration, x.strike))
    
    def get_atm_options(self, 
                       symbol: str,
                       tolerance: Decimal = Decimal('5.0')) -> List[OptionQuote]:
        """Get at-the-money options within tolerance."""
        chain = self.get_options_chain(symbol)
        
        if not chain:
            return []
        
        return chain.get_atm_options(tolerance)
    
    def get_expiration_dates(self, symbol: str) -> List[date]:
        """Get available expiration dates for a symbol."""
        # Try cache first
        cache_key = f"expirations:{symbol}"
        cached_expirations = self._get_from_cache(cache_key)
        
        if cached_expirations:
            return cached_expirations
        
        # Get from API
        try:
            expirations = self.tradier_client.get_options_expirations(symbol)
            
            # Cache for shorter time since expirations don't change often
            self._set_cache(cache_key, expirations)
            
            return expirations
            
        except Exception as e:
            self._handle_error(f"get_expiration_dates({symbol})", e)
            return []
    
    def get_near_term_options(self, 
                             symbol: str,
                             min_days: int = 7,
                             max_days: int = 45) -> List[OptionQuote]:
        """Get options expiring within specified days range."""
        chain = self.get_options_chain(symbol)
        
        if not chain:
            return []
        
        today = date.today()
        near_term_options = []
        
        for option in chain.options:
            days_to_expiry = (option.expiration - today).days
            
            if min_days <= days_to_expiry <= max_days:
                near_term_options.append(option)
        
        return sorted(near_term_options, key=lambda x: (x.expiration, x.strike))
    
    def validate_entity(self, entity: OptionsChain) -> bool:
        """Validate options chain entity."""
        if not super().validate_entity(entity):
            return False
        
        # Check required fields
        if not entity.underlying:
            self.logger.error("Options chain missing underlying symbol")
            return False
        
        # Validate options in chain
        for option in entity.options:
            if option.underlying != entity.underlying:
                self.logger.error(f"Option underlying {option.underlying} doesn't match chain {entity.underlying}")
                return False
        
        return True
    
    def get_chain_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for an options chain."""
        chain = self.get_options_chain(symbol)
        
        if not chain:
            return {}
        
        return chain.get_statistics()
    
    def refresh_stale_data(self, max_age_minutes: int = 10) -> int:
        """Refresh options chains older than max_age_minutes."""
        refreshed_count = 0
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        # Check cached chains for staleness
        stale_symbols = []
        
        for cache_key, cache_entry in self._cache.items():
            if cache_key.startswith("options_chain:"):
                if cache_entry['timestamp'] < cutoff_time:
                    symbol = cache_key.split(":")[1]
                    stale_symbols.append(symbol)
        
        # Refresh stale chains
        for symbol in stale_symbols:
            try:
                self.get_options_chain(symbol, force_refresh=True)
                refreshed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to refresh stale data for {symbol}: {e}")
        
        self.logger.info(f"Refreshed {refreshed_count} stale options chains")
        return refreshed_count