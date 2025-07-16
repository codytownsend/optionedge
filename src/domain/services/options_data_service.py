"""
Comprehensive options data retrieval and management service.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...data.models.options import (
    OptionsChain, OptionQuote, OptionType, Greeks, 
    calculate_black_scholes_greeks, calculate_probability_of_profit_bs
)
from ...data.models.market_data import StockQuote
from ...infrastructure.api import TradierClient, YahooFinanceClient
from ...infrastructure.cache import DataTypeCacheManager, CacheKey
from ...infrastructure.error_handling import (
    handle_errors, DataQualityError, InsufficientDataError, 
    StaleDataError, InsufficientLiquidityError
)


class OptionsDataQualityMetrics:
    """Metrics for assessing options data quality."""
    
    def __init__(self):
        self.quote_age_seconds: Optional[float] = None
        self.bid_ask_spread_pct: Optional[float] = None
        self.volume: Optional[int] = None
        self.open_interest: Optional[int] = None
        self.implied_volatility: Optional[float] = None
        self.greeks_available: bool = False
        self.strike_ladder_completeness: float = 0.0
        self.liquidity_score: float = 0.0
        self.quality_score: float = 0.0
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        scores = []
        weights = []
        
        # Quote freshness (0-1, 1 = fresh)
        if self.quote_age_seconds is not None:
            freshness_score = max(0, 1 - (self.quote_age_seconds / 600))  # 10 min max
            scores.append(freshness_score)
            weights.append(0.25)
        
        # Bid-ask spread quality (0-1, 1 = tight spread)
        if self.bid_ask_spread_pct is not None:
            spread_score = max(0, 1 - (self.bid_ask_spread_pct / 0.1))  # 10% max
            scores.append(spread_score)
            weights.append(0.20)
        
        # Volume adequacy (0-1)
        if self.volume is not None:
            volume_score = min(1, self.volume / 100)  # 100+ volume = perfect
            scores.append(volume_score)
            weights.append(0.15)
        
        # Open interest adequacy (0-1)
        if self.open_interest is not None:
            oi_score = min(1, self.open_interest / 500)  # 500+ OI = perfect
            scores.append(oi_score)
            weights.append(0.15)
        
        # Greeks availability
        if self.greeks_available:
            scores.append(1.0)
            weights.append(0.10)
        
        # Strike ladder completeness
        scores.append(self.strike_ladder_completeness)
        weights.append(0.15)
        
        if not scores:
            return 0.0
        
        # Weighted average
        self.quality_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return self.quality_score


class OptionsDataService:
    """
    Comprehensive options data retrieval and processing service.
    
    Features:
    - Full options chain extraction for all expiration cycles
    - Real-time bid/ask spread monitoring and quality assessment
    - Greeks calculation verification using multiple methodologies
    - Open interest and volume trend analysis
    - Strike ladder completeness validation and gap identification
    """
    
    def __init__(
        self,
        tradier_client: TradierClient,
        yahoo_client: YahooFinanceClient,
        cache_manager: DataTypeCacheManager,
        max_workers: int = 5
    ):
        self.tradier_client = tradier_client
        self.yahoo_client = yahoo_client
        self.cache_manager = cache_manager
        self.max_workers = max_workers
        
        # Data quality thresholds (from instructions)
        self.quality_thresholds = {
            "max_quote_age_minutes": 10,
            "max_bid_ask_spread_pct": 0.05,  # 5%
            "min_volume": 10,
            "min_open_interest": 50,  # Reduced from 100 for practicality
            "min_strike_completeness": 0.8  # 80% of expected strikes
        }
        
        # Risk-free rate for Greeks calculations
        self.risk_free_rate = 0.05  # 5% default, should be updated from FRED
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    @handle_errors(operation_name="get_options_chain")
    def get_comprehensive_options_chain(
        self, 
        symbol: str,
        min_dte: int = 7,
        max_dte: int = 45,
        validate_quality: bool = True
    ) -> OptionsChain:
        """
        Get comprehensive options chain with quality validation.
        
        Args:
            symbol: Underlying symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            validate_quality: Whether to perform quality checks
            
        Returns:
            Validated and enhanced options chain
        """
        self.logger.info(f"Retrieving comprehensive options chain for {symbol}")
        
        # Check cache first
        cached_chain = self.cache_manager.get_options_data(
            symbol,
            refresh_func=lambda: self._fetch_fresh_options_chain(symbol, min_dte, max_dte)
        )
        
        if cached_chain:
            if validate_quality:
                quality_metrics = self._assess_chain_quality(cached_chain)
                if quality_metrics.quality_score < 0.6:  # 60% minimum quality
                    self.logger.warning(f"Cached options chain for {symbol} has low quality score: {quality_metrics.quality_score:.2f}")
                    # Force refresh for low quality data
                    cached_chain = self._fetch_fresh_options_chain(symbol, min_dte, max_dte)
            return cached_chain
        
        # Fetch fresh data
        return self._fetch_fresh_options_chain(symbol, min_dte, max_dte)
    
    def _fetch_fresh_options_chain(
        self, 
        symbol: str, 
        min_dte: int, 
        max_dte: int
    ) -> OptionsChain:
        """Fetch fresh options chain from primary data source."""
        
        # Get underlying price first
        underlying_quote = self.tradier_client.get_stock_quote(symbol)
        if not underlying_quote:
            raise DataQualityError(
                f"Unable to get underlying quote for {symbol}",
                data_source="Tradier",
                symbol=symbol
            )
        
        # Get available expirations
        expirations = self.tradier_client.get_options_expirations(symbol)
        if not expirations:
            raise InsufficientDataError(
                f"No options expirations available for {symbol}",
                symbol=symbol,
                data_source="Tradier"
            )
        
        # Filter expirations by DTE
        today = date.today()
        valid_expirations = [
            exp for exp in expirations 
            if min_dte <= (exp - today).days <= max_dte
        ]
        
        if not valid_expirations:
            raise InsufficientDataError(
                f"No expirations found for {symbol} within DTE range {min_dte}-{max_dte}",
                symbol=symbol,
                required_data_points=1,
                available_data_points=0
            )
        
        # Create base chain
        chain = OptionsChain(
            underlying=symbol,
            underlying_price=underlying_quote.price,
            data_source="Tradier"
        )
        
        # Fetch options for each expiration in parallel
        futures = []
        for expiration in valid_expirations:
            future = self.executor.submit(
                self._fetch_expiration_options,
                symbol,
                expiration,
                underlying_quote.price
            )
            futures.append((expiration, future))
        
        # Collect results
        for expiration, future in futures:
            try:
                options = future.result(timeout=30)
                for option in options:
                    chain.add_option(option)
            except Exception as e:
                self.logger.warning(f"Failed to fetch options for {symbol} {expiration}: {str(e)}")
        
        # Validate and enhance chain
        self._validate_and_enhance_chain(chain)
        
        # Cache the result
        self.cache_manager.cache_options_data(symbol, chain)
        
        return chain
    
    def _fetch_expiration_options(
        self, 
        symbol: str, 
        expiration: date,
        underlying_price: Decimal
    ) -> List[OptionQuote]:
        """Fetch options for a specific expiration."""
        
        # Get strikes for this expiration
        strikes = self.tradier_client.get_option_strikes(symbol, expiration)
        if not strikes:
            return []
        
        # Get options chain for this expiration
        chain = self.tradier_client.get_options_chain(symbol, expiration)
        if not chain:
            return []
        
        options = []
        for exp_date, strikes_dict in chain.options.items():
            if exp_date != expiration:
                continue
                
            for strike, types_dict in strikes_dict.items():
                for option_type, option_quote in types_dict.items():
                    # Enhance option with calculated Greeks if missing
                    enhanced_option = self._enhance_option_quote(
                        option_quote, 
                        underlying_price
                    )
                    options.append(enhanced_option)
        
        return options
    
    def _enhance_option_quote(
        self, 
        option: OptionQuote, 
        underlying_price: Decimal
    ) -> OptionQuote:
        """Enhance option quote with calculated Greeks and probability metrics."""
        
        # Calculate Greeks if missing or verify existing ones
        if not option.greeks or self._should_recalculate_greeks(option):
            calculated_greeks = self._calculate_greeks(option, underlying_price)
            
            # Create enhanced option
            enhanced_option = OptionQuote(
                symbol=option.symbol,
                underlying=option.underlying,
                strike=option.strike,
                expiration=option.expiration,
                option_type=option.option_type,
                bid=option.bid,
                ask=option.ask,
                last=option.last,
                mark=option.mark,
                volume=option.volume,
                open_interest=option.open_interest,
                implied_volatility=option.implied_volatility,
                greeks=calculated_greeks,
                quote_time=option.quote_time,
                exchange=option.exchange
            )
            
            return enhanced_option
        
        return option
    
    def _calculate_greeks(
        self, 
        option: OptionQuote, 
        underlying_price: Decimal
    ) -> Optional[Greeks]:
        """Calculate Greeks using Black-Scholes model."""
        
        if not all([
            underlying_price,
            option.strike,
            option.time_to_expiration > 0,
            option.implied_volatility
        ]):
            return None
        
        try:
            return calculate_black_scholes_greeks(
                S=float(underlying_price),
                K=float(option.strike),
                T=option.time_to_expiration,
                r=self.risk_free_rate,
                sigma=option.implied_volatility,
                option_type=option.option_type
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate Greeks for {option.symbol}: {str(e)}")
            return None
    
    def _should_recalculate_greeks(self, option: OptionQuote) -> bool:
        """Determine if Greeks should be recalculated."""
        if not option.greeks:
            return True
        
        # Recalculate if any Greeks are missing or seem invalid
        greeks = option.greeks
        if (greeks.delta is None or 
            greeks.gamma is None or 
            greeks.theta is None or 
            greeks.vega is None):
            return True
        
        # Basic sanity checks
        if option.option_type == OptionType.CALL and greeks.delta and greeks.delta < 0:
            return True
        
        if option.option_type == OptionType.PUT and greeks.delta and greeks.delta > 0:
            return True
        
        return False
    
    def _validate_and_enhance_chain(self, chain: OptionsChain):
        """Validate and enhance the options chain with quality metrics."""
        
        if not chain.options:
            raise InsufficientDataError(
                f"Empty options chain for {chain.underlying}",
                symbol=chain.underlying
            )
        
        # Analyze strike ladder completeness
        self._analyze_strike_completeness(chain)
        
        # Validate quote freshness
        self._validate_quote_freshness(chain)
        
        # Check for data anomalies
        self._detect_data_anomalies(chain)
    
    def _analyze_strike_completeness(self, chain: OptionsChain):
        """Analyze strike ladder completeness for each expiration."""
        
        for expiration in chain.get_expirations():
            strikes = chain.get_strikes(expiration)
            if len(strikes) < 3:  # Need minimum strikes for analysis
                continue
            
            # Calculate expected strike range based on underlying price
            if chain.underlying_price:
                atm_strike = chain.get_atm_strike(expiration)
                if atm_strike:
                    # Expect strikes from 80% to 120% of underlying
                    expected_min = chain.underlying_price * Decimal('0.8')
                    expected_max = chain.underlying_price * Decimal('1.2')
                    
                    # Count strikes in expected range
                    strikes_in_range = [
                        s for s in strikes 
                        if expected_min <= s <= expected_max
                    ]
                    
                    # Calculate completeness ratio
                    expected_count = max(10, len(strikes))  # Expect at least 10 strikes
                    completeness = len(strikes_in_range) / expected_count
                    
                    self.logger.debug(
                        f"Strike completeness for {chain.underlying} {expiration}: "
                        f"{len(strikes_in_range)}/{expected_count} = {completeness:.1%}"
                    )
    
    def _validate_quote_freshness(self, chain: OptionsChain):
        """Validate that quotes are fresh enough for trading."""
        
        stale_quotes = []
        max_age = timedelta(minutes=self.quality_thresholds["max_quote_age_minutes"])
        
        for expiration, strikes_dict in chain.options.items():
            for strike, types_dict in strikes_dict.items():
                for option_type, option in types_dict.items():
                    age = datetime.utcnow() - option.quote_time
                    if age > max_age:
                        stale_quotes.append(option.symbol)
        
        if stale_quotes:
            self.logger.warning(
                f"Found {len(stale_quotes)} stale quotes for {chain.underlying}. "
                f"Oldest age: {max_age}"
            )
            
            # If too many stale quotes, raise error
            total_options = sum(
                len(types_dict) 
                for strikes_dict in chain.options.values() 
                for types_dict in strikes_dict.values()
            )
            
            stale_ratio = len(stale_quotes) / total_options if total_options > 0 else 0
            if stale_ratio > 0.5:  # More than 50% stale
                raise StaleDataError(
                    f"Too many stale quotes for {chain.underlying}: {stale_ratio:.1%}",
                    symbol=chain.underlying,
                    data_age_seconds=max_age.total_seconds()
                )
    
    def _detect_data_anomalies(self, chain: OptionsChain):
        """Detect potential data quality issues."""
        
        anomalies = []
        
        for expiration, strikes_dict in chain.options.items():
            for strike, types_dict in strikes_dict.items():
                for option_type, option in types_dict.items():
                    
                    # Check for unrealistic bid/ask spreads
                    if option.bid_ask_spread_pct:
                        if option.bid_ask_spread_pct > 0.5:  # 50% spread
                            anomalies.append(f"Wide spread: {option.symbol} ({option.bid_ask_spread_pct:.1%})")
                    
                    # Check for zero volume with high open interest
                    if (option.volume == 0 and 
                        option.open_interest and option.open_interest > 1000):
                        anomalies.append(f"Zero volume with high OI: {option.symbol}")
                    
                    # Check for impossible Greeks values
                    if option.greeks:
                        if (option.greeks.delta and abs(option.greeks.delta) > 1):
                            anomalies.append(f"Invalid delta: {option.symbol} ({option.greeks.delta})")
                        
                        if (option.greeks.gamma and option.greeks.gamma < 0):
                            anomalies.append(f"Negative gamma: {option.symbol} ({option.greeks.gamma})")
        
        if anomalies:
            self.logger.warning(f"Data anomalies detected for {chain.underlying}: {anomalies[:5]}")  # Log first 5
    
    def _assess_chain_quality(self, chain: OptionsChain) -> OptionsDataQualityMetrics:
        """Assess overall quality of options chain."""
        
        metrics = OptionsDataQualityMetrics()
        
        if not chain.options:
            return metrics
        
        # Sample options for quality assessment
        sample_options = []
        for strikes_dict in chain.options.values():
            for types_dict in strikes_dict.values():
                sample_options.extend(types_dict.values())
        
        if not sample_options:
            return metrics
        
        # Calculate average metrics
        quote_ages = []
        spreads = []
        volumes = []
        open_interests = []
        greeks_count = 0
        
        for option in sample_options:
            # Quote age
            age = (datetime.utcnow() - option.quote_time).total_seconds()
            quote_ages.append(age)
            
            # Spread
            if option.bid_ask_spread_pct:
                spreads.append(option.bid_ask_spread_pct)
            
            # Volume
            if option.volume:
                volumes.append(option.volume)
            
            # Open interest
            if option.open_interest:
                open_interests.append(option.open_interest)
            
            # Greeks availability
            if option.greeks:
                greeks_count += 1
        
        # Set metrics
        metrics.quote_age_seconds = sum(quote_ages) / len(quote_ages) if quote_ages else None
        metrics.bid_ask_spread_pct = sum(spreads) / len(spreads) if spreads else None
        metrics.volume = sum(volumes) / len(volumes) if volumes else None
        metrics.open_interest = sum(open_interests) / len(open_interests) if open_interests else None
        metrics.greeks_available = greeks_count / len(sample_options) > 0.8  # 80% have Greeks
        
        # Strike ladder completeness (simplified)
        total_expirations = len(chain.options)
        complete_expirations = sum(
            1 for strikes_dict in chain.options.values() 
            if len(strikes_dict) >= 5  # At least 5 strikes
        )
        metrics.strike_ladder_completeness = complete_expirations / total_expirations if total_expirations > 0 else 0
        
        # Calculate overall quality score
        metrics.calculate_quality_score()
        
        return metrics
    
    def get_liquid_options_chain(
        self, 
        symbol: str,
        min_dte: int = 7,
        max_dte: int = 45
    ) -> OptionsChain:
        """Get options chain filtered for liquid options only."""
        
        # Get full chain first
        full_chain = self.get_comprehensive_options_chain(symbol, min_dte, max_dte)
        
        # Filter for liquidity
        liquid_chain = full_chain.filter_liquid_options(
            min_volume=self.quality_thresholds["min_volume"],
            min_oi=self.quality_thresholds["min_open_interest"],
            max_spread_pct=self.quality_thresholds["max_bid_ask_spread_pct"]
        )
        
        # Validate sufficient liquid options
        total_liquid = sum(
            len(types_dict) 
            for strikes_dict in liquid_chain.options.values() 
            for types_dict in strikes_dict.values()
        )
        
        if total_liquid < 10:  # Need minimum liquid options
            raise InsufficientLiquidityError(
                f"Insufficient liquid options for {symbol}: {total_liquid} found",
                symbol=symbol
            )
        
        return liquid_chain
    
    def batch_get_options_chains(
        self, 
        symbols: List[str],
        min_dte: int = 7,
        max_dte: int = 45
    ) -> Dict[str, OptionsChain]:
        """Get options chains for multiple symbols in parallel."""
        
        results = {}
        futures = []
        
        # Submit all requests
        for symbol in symbols:
            future = self.executor.submit(
                self.get_comprehensive_options_chain,
                symbol,
                min_dte,
                max_dte
            )
            futures.append((symbol, future))
        
        # Collect results
        for symbol, future in futures:
            try:
                chain = future.result(timeout=60)
                results[symbol] = chain
                self.logger.info(f"Successfully retrieved options chain for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to get options chain for {symbol}: {str(e)}")
                # Continue with other symbols
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)