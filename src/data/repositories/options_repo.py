"""
Options data repository for the Options Trading Engine.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .base import BaseRepository
from ..models.options import OptionContract, OptionChain, OptionData
from ...domain.entities.option_contract import OptionContract as DomainOptionContract
from ...infrastructure.api import TradierClient, YahooClient
from ...infrastructure.error_handling import RepositoryError, DataError
from ...infrastructure.cache import CacheManager

logger = logging.getLogger(__name__)


class OptionsRepository(BaseRepository[OptionContract]):
    """
    Repository for options data operations.
    
    Handles options data retrieval, caching, and validation from multiple sources.
    """
    
    def __init__(self, 
                 tradier_client: TradierClient,
                 yahoo_client: YahooClient,
                 cache_manager: Optional[CacheManager] = None):
        super().__init__(cache_manager)
        self.tradier_client = tradier_client
        self.yahoo_client = yahoo_client
    
    def get_by_id(self, id: str) -> Optional[OptionContract]:
        """Get option contract by ID."""
        return self.get_option_contract(id)
    
    def create(self, entity: OptionContract) -> OptionContract:
        """Create is not supported for options data."""
        raise NotImplementedError("Options data creation not supported")
    
    def update(self, entity: OptionContract) -> OptionContract:
        """Update is not supported for options data."""
        raise NotImplementedError("Options data update not supported")
    
    def delete(self, id: str) -> bool:
        """Delete option contract from cache."""
        cache_key = self._get_cache_key("option_contract", id)
        self._invalidate_cache(cache_key)
        return True
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[OptionContract]:
        """Find all option contracts matching filters."""
        if not filters:
            return []
        
        symbol = filters.get('symbol')
        expiration_date = filters.get('expiration_date')
        
        if not symbol:
            return []
        
        if expiration_date:
            chain = self.get_option_chain(symbol, expiration_date)
            if chain:
                contracts = chain.calls + chain.puts
                return self._apply_filters(contracts, filters)
        
        return []
    
    def get_option_contract(self, symbol: str, force_refresh: bool = False) -> Optional[OptionContract]:
        """
        Get specific option contract.
        
        Args:
            symbol: Option symbol (e.g., 'AAPL210716C00150000')
            force_refresh: Force refresh from API
            
        Returns:
            OptionContract object or None if not found
        """
        cache_key = self._get_cache_key("option_contract", symbol)
        
        # Check cache first
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data and self._is_data_fresh(cached_data.timestamp, 5):
                return cached_data
        
        try:
            # Get option data from Tradier
            option_data = self.tradier_client.get_option_quotes([symbol])
            if not option_data or symbol not in option_data:
                return None
            
            contract_data = option_data[symbol]
            
            # Create option contract
            option_contract = self._create_option_contract(symbol, contract_data)
            
            # Cache the result
            self._set_to_cache(cache_key, option_contract, ttl=300)  # 5 minutes
            
            return option_contract
            
        except Exception as e:
            self._handle_repository_error("get_option_contract", e)
            return None
    
    def get_option_chain(self, 
                        symbol: str, 
                        expiration_date: str,
                        force_refresh: bool = False) -> Optional[OptionChain]:
        """
        Get option chain for symbol and expiration.
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date (YYYY-MM-DD format)
            force_refresh: Force refresh from API
            
        Returns:
            OptionChain object or None if not found
        """
        cache_key = self._get_cache_key("option_chain", symbol, expiration_date)
        
        # Check cache first
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data and self._is_data_fresh(cached_data.timestamp, 5):
                return cached_data
        
        try:
            # Get option chain from Tradier
            chain_data = self.tradier_client.get_option_chain(symbol, expiration_date)
            if not chain_data:
                return None
            
            # Process calls and puts
            calls = []
            puts = []
            
            for option_data in chain_data.get('options', []):
                if option_data.get('option_type') == 'call':
                    calls.append(self._create_option_contract(
                        option_data['symbol'], 
                        option_data
                    ))
                else:
                    puts.append(self._create_option_contract(
                        option_data['symbol'], 
                        option_data
                    ))
            
            # Create option chain
            option_chain = OptionChain(
                symbol=symbol,
                expiration_date=datetime.strptime(expiration_date, '%Y-%m-%d').date(),
                calls=calls,
                puts=puts,
                timestamp=self._get_timestamp()
            )
            
            # Cache the result
            self._set_to_cache(cache_key, option_chain, ttl=300)  # 5 minutes
            
            return option_chain
            
        except Exception as e:
            self._handle_repository_error("get_option_chain", e)
            return None
    
    def get_option_chains_multi_expiry(self, 
                                     symbol: str, 
                                     expiration_dates: List[str],
                                     force_refresh: bool = False) -> List[OptionChain]:
        """
        Get option chains for multiple expiration dates.
        
        Args:
            symbol: Stock symbol
            expiration_dates: List of expiration dates
            force_refresh: Force refresh from API
            
        Returns:
            List of OptionChain objects
        """
        chains = []
        
        for expiration_date in expiration_dates:
            chain = self.get_option_chain(symbol, expiration_date, force_refresh)
            if chain:
                chains.append(chain)
        
        return chains
    
    def get_available_expirations(self, symbol: str) -> List[str]:
        """
        Get available expiration dates for symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        cache_key = self._get_cache_key("expirations", symbol)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data and self._is_data_fresh(cached_data.get('timestamp'), 60):
            return cached_data.get('expirations', [])
        
        try:
            expirations = self.tradier_client.get_option_expirations(symbol)
            
            if expirations:
                cache_data = {
                    'expirations': expirations,
                    'timestamp': self._get_timestamp()
                }
                self._set_to_cache(cache_key, cache_data, ttl=3600)  # 1 hour
                return expirations
            
            return []
            
        except Exception as e:
            self._handle_repository_error("get_available_expirations", e)
            return []
    
    def get_option_quotes(self, symbols: List[str]) -> Dict[str, OptionContract]:
        """
        Get quotes for multiple option symbols.
        
        Args:
            symbols: List of option symbols
            
        Returns:
            Dictionary mapping symbol to OptionContract
        """
        results = {}
        
        # Split into cached and non-cached
        cached_symbols = []
        non_cached_symbols = []
        
        for symbol in symbols:
            cache_key = self._get_cache_key("option_contract", symbol)
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data and self._is_data_fresh(cached_data.timestamp, 5):
                results[symbol] = cached_data
                cached_symbols.append(symbol)
            else:
                non_cached_symbols.append(symbol)
        
        # Get non-cached data from API
        if non_cached_symbols:
            try:
                option_data = self.tradier_client.get_option_quotes(non_cached_symbols)
                
                for symbol, data in option_data.items():
                    contract = self._create_option_contract(symbol, data)
                    results[symbol] = contract
                    
                    # Cache the result
                    cache_key = self._get_cache_key("option_contract", symbol)
                    self._set_to_cache(cache_key, contract, ttl=300)  # 5 minutes
                    
            except Exception as e:
                self._handle_repository_error("get_option_quotes", e)
        
        return results
    
    def search_options(self, 
                      symbol: str,
                      min_days_to_expiry: int = 1,
                      max_days_to_expiry: int = 365,
                      min_delta: float = 0.0,
                      max_delta: float = 1.0,
                      option_type: Optional[str] = None,
                      min_volume: int = 0,
                      min_open_interest: int = 0) -> List[OptionContract]:
        """
        Search for options matching criteria.
        
        Args:
            symbol: Stock symbol
            min_days_to_expiry: Minimum days to expiration
            max_days_to_expiry: Maximum days to expiration
            min_delta: Minimum delta
            max_delta: Maximum delta
            option_type: Option type ('call' or 'put')
            min_volume: Minimum volume
            min_open_interest: Minimum open interest
            
        Returns:
            List of matching OptionContract objects
        """
        # Get available expirations
        expirations = self.get_available_expirations(symbol)
        
        # Filter by days to expiry
        filtered_expirations = []
        current_date = datetime.now().date()
        
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            days_to_expiry = (exp_date - current_date).days
            
            if min_days_to_expiry <= days_to_expiry <= max_days_to_expiry:
                filtered_expirations.append(exp_str)
        
        # Get option chains for filtered expirations
        matching_contracts = []
        
        for expiration in filtered_expirations:
            chain = self.get_option_chain(symbol, expiration)
            if not chain:
                continue
            
            # Filter contracts based on criteria
            contracts = chain.calls + chain.puts
            
            for contract in contracts:
                # Filter by option type
                if option_type and contract.option_type != option_type:
                    continue
                
                # Filter by delta
                if contract.greeks and contract.greeks.delta:
                    if not (min_delta <= abs(contract.greeks.delta) <= max_delta):
                        continue
                
                # Filter by volume
                if contract.volume < min_volume:
                    continue
                
                # Filter by open interest
                if contract.open_interest < min_open_interest:
                    continue
                
                matching_contracts.append(contract)
        
        return matching_contracts
    
    def get_options_by_delta_range(self, 
                                  symbol: str,
                                  target_delta: float,
                                  delta_tolerance: float = 0.05,
                                  option_type: str = 'call') -> List[OptionContract]:
        """
        Get options within a specific delta range.
        
        Args:
            symbol: Stock symbol
            target_delta: Target delta value
            delta_tolerance: Tolerance around target delta
            option_type: Option type ('call' or 'put')
            
        Returns:
            List of OptionContract objects within delta range
        """
        min_delta = target_delta - delta_tolerance
        max_delta = target_delta + delta_tolerance
        
        return self.search_options(
            symbol=symbol,
            min_delta=min_delta,
            max_delta=max_delta,
            option_type=option_type
        )
    
    def get_liquid_options(self, 
                          symbol: str,
                          min_volume: int = 100,
                          min_open_interest: int = 500,
                          max_bid_ask_spread: float = 0.05) -> List[OptionContract]:
        """
        Get liquid options for a symbol.
        
        Args:
            symbol: Stock symbol
            min_volume: Minimum daily volume
            min_open_interest: Minimum open interest
            max_bid_ask_spread: Maximum bid-ask spread as percentage
            
        Returns:
            List of liquid OptionContract objects
        """
        contracts = self.search_options(
            symbol=symbol,
            min_volume=min_volume,
            min_open_interest=min_open_interest
        )
        
        # Filter by bid-ask spread
        liquid_contracts = []
        for contract in contracts:
            if contract.bid > 0 and contract.ask > 0:
                spread_percentage = (contract.ask - contract.bid) / contract.ask
                if spread_percentage <= max_bid_ask_spread:
                    liquid_contracts.append(contract)
        
        return liquid_contracts
    
    def _create_option_contract(self, symbol: str, data: Dict[str, Any]) -> OptionContract:
        """Create OptionContract from API data."""
        # Parse option symbol to extract components
        parsed_symbol = self._parse_option_symbol(symbol)
        
        # Create Greeks object if available
        greeks = None
        if 'greeks' in data:
            greeks_data = data['greeks']
            from ...domain.value_objects.greeks import Greeks
            greeks = Greeks(
                delta=greeks_data.get('delta', 0.0),
                gamma=greeks_data.get('gamma', 0.0),
                theta=greeks_data.get('theta', 0.0),
                vega=greeks_data.get('vega', 0.0),
                rho=greeks_data.get('rho', 0.0)
            )
        
        return OptionContract(
            symbol=symbol,
            underlying_symbol=parsed_symbol['underlying'],
            option_type=parsed_symbol['option_type'],
            strike_price=parsed_symbol['strike'],
            expiration_date=parsed_symbol['expiration'],
            bid=data.get('bid', 0.0),
            ask=data.get('ask', 0.0),
            last_price=data.get('last', 0.0),
            volume=data.get('volume', 0),
            open_interest=data.get('open_interest', 0),
            implied_volatility=data.get('implied_volatility', 0.0),
            greeks=greeks,
            timestamp=self._get_timestamp()
        )
    
    def _parse_option_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Parse option symbol to extract components.
        
        Format: AAPL210716C00150000
        - AAPL: underlying symbol
        - 210716: expiration date (YYMMDD)
        - C: option type (C=call, P=put)
        - 00150000: strike price ($150.00)
        """
        try:
            # Extract underlying symbol (everything before the date)
            underlying_end = len(symbol) - 15  # 15 chars for date + type + strike
            underlying = symbol[:underlying_end]
            
            # Extract expiration date
            exp_str = symbol[underlying_end:underlying_end+6]
            exp_date = datetime.strptime(f"20{exp_str}", '%Y%m%d').date()
            
            # Extract option type
            option_type = symbol[underlying_end+6:underlying_end+7]
            option_type = 'call' if option_type.upper() == 'C' else 'put'
            
            # Extract strike price
            strike_str = symbol[underlying_end+7:]
            strike = float(strike_str) / 1000.0
            
            return {
                'underlying': underlying,
                'expiration': exp_date,
                'option_type': option_type,
                'strike': strike
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse option symbol {symbol}: {e}")
            return {
                'underlying': symbol,
                'expiration': datetime.now().date(),
                'option_type': 'call',
                'strike': 0.0
            }
    
    def get_iv_surface(self, symbol: str, expiration_dates: List[str]) -> Dict[str, Any]:
        """
        Get implied volatility surface for symbol.
        
        Args:
            symbol: Stock symbol
            expiration_dates: List of expiration dates
            
        Returns:
            Dictionary containing IV surface data
        """
        cache_key = self._get_cache_key("iv_surface", symbol, *expiration_dates)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data and self._is_data_fresh(cached_data.get('timestamp'), 15):
            return cached_data.get('data', {})
        
        try:
            iv_surface = {
                'symbol': symbol,
                'expirations': {},
                'timestamp': self._get_timestamp()
            }
            
            for expiration in expiration_dates:
                chain = self.get_option_chain(symbol, expiration)
                if not chain:
                    continue
                
                # Collect IV data by strike
                calls_iv = {}
                puts_iv = {}
                
                for call in chain.calls:
                    if call.implied_volatility > 0:
                        calls_iv[call.strike_price] = call.implied_volatility
                
                for put in chain.puts:
                    if put.implied_volatility > 0:
                        puts_iv[put.strike_price] = put.implied_volatility
                
                iv_surface['expirations'][expiration] = {
                    'calls': calls_iv,
                    'puts': puts_iv
                }
            
            # Cache the result
            cache_data = {
                'data': iv_surface,
                'timestamp': self._get_timestamp()
            }
            self._set_to_cache(cache_key, cache_data, ttl=900)  # 15 minutes
            
            return iv_surface
            
        except Exception as e:
            self._handle_repository_error("get_iv_surface", e)
            return {}
    
    def get_option_volume_analysis(self, symbol: str, days_back: int = 5) -> Dict[str, Any]:
        """
        Get volume analysis for options.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            Dictionary containing volume analysis
        """
        cache_key = self._get_cache_key("volume_analysis", symbol, days_back)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data and self._is_data_fresh(cached_data.get('timestamp'), 60):
            return cached_data.get('data', {})
        
        try:
            # Get current option chains
            expirations = self.get_available_expirations(symbol)[:5]  # Top 5 expirations
            
            volume_analysis = {
                'symbol': symbol,
                'total_call_volume': 0,
                'total_put_volume': 0,
                'put_call_ratio': 0.0,
                'high_volume_strikes': [],
                'unusual_activity': [],
                'timestamp': self._get_timestamp()
            }
            
            total_call_volume = 0
            total_put_volume = 0
            
            for expiration in expirations:
                chain = self.get_option_chain(symbol, expiration)
                if not chain:
                    continue
                
                # Analyze call volume
                for call in chain.calls:
                    total_call_volume += call.volume
                    
                    # Check for unusual activity
                    if call.volume > call.open_interest * 2:  # Volume > 2x open interest
                        volume_analysis['unusual_activity'].append({
                            'symbol': call.symbol,
                            'type': 'call',
                            'strike': call.strike_price,
                            'volume': call.volume,
                            'open_interest': call.open_interest,
                            'ratio': call.volume / max(call.open_interest, 1)
                        })
                
                # Analyze put volume
                for put in chain.puts:
                    total_put_volume += put.volume
                    
                    # Check for unusual activity
                    if put.volume > put.open_interest * 2:  # Volume > 2x open interest
                        volume_analysis['unusual_activity'].append({
                            'symbol': put.symbol,
                            'type': 'put',
                            'strike': put.strike_price,
                            'volume': put.volume,
                            'open_interest': put.open_interest,
                            'ratio': put.volume / max(put.open_interest, 1)
                        })
            
            volume_analysis['total_call_volume'] = total_call_volume
            volume_analysis['total_put_volume'] = total_put_volume
            
            # Calculate put/call ratio
            if total_call_volume > 0:
                volume_analysis['put_call_ratio'] = total_put_volume / total_call_volume
            
            # Cache the result
            cache_data = {
                'data': volume_analysis,
                'timestamp': self._get_timestamp()
            }
            self._set_to_cache(cache_key, cache_data, ttl=3600)  # 1 hour
            
            return volume_analysis
            
        except Exception as e:
            self._handle_repository_error("get_option_volume_analysis", e)
            return {}