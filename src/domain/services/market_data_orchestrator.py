"""
Market data orchestration layer for coordinating all data services.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import logging

from ...data.models.options import OptionsChain
from ...data.models.market_data import (
    MarketData, StockQuote, TechnicalIndicators, 
    FundamentalData, SentimentData, EconomicIndicator
)
from ...data.models.trades import TradeCandidate
from ...infrastructure.api import (
    TradierClient, YahooFinanceClient, FREDClient, QuiverQuantClient
)
from ...infrastructure.cache import DataTypeCacheManager
from ...infrastructure.error_handling import (
    handle_errors, DataQualityError, InsufficientDataError,
    ExternalServiceError
)

from .options_data_service import OptionsDataService
from .data_quality_service import DataQualityService, QualityReport
from .fundamental_data_service import FundamentalDataService, CompanyAnalysis
from .technical_analysis_service import TechnicalAnalysisService, TechnicalSummary


@dataclass
class DataRequest:
    """Request for market data collection."""
    symbol: str
    include_options: bool = True
    include_fundamentals: bool = True
    include_technicals: bool = True
    include_sentiment: bool = True
    min_dte: int = 7
    max_dte: int = 45
    quality_threshold: float = 0.6  # Minimum quality score
    priority: int = 1  # Higher numbers = higher priority


@dataclass
class DataCollectionResult:
    """Result of market data collection for a symbol."""
    symbol: str
    timestamp: datetime
    success: bool
    market_data: Optional[MarketData] = None
    quality_reports: List[QualityReport] = field(default_factory=list)
    technical_summary: Optional[TechnicalSummary] = None
    fundamental_analysis: Optional[CompanyAnalysis] = None
    errors: List[str] = field(default_factory=list)
    collection_time_ms: Optional[float] = None


@dataclass
class BatchCollectionSummary:
    """Summary of batch data collection operation."""
    total_symbols: int
    successful_collections: int
    failed_collections: int
    avg_collection_time_ms: float
    quality_score_avg: float
    start_time: datetime
    end_time: datetime
    error_summary: Dict[str, int] = field(default_factory=dict)


class MarketDataOrchestrator:
    """
    Centralized orchestration layer for all market data collection.
    
    Features:
    - Coordinated data collection from all sources
    - Quality validation and data consistency checks
    - Intelligent caching and refresh strategies
    - Parallel processing for multiple symbols
    - Error handling and fallback mechanisms
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        tradier_client: TradierClient,
        yahoo_client: YahooFinanceClient,
        fred_client: FREDClient,
        quiver_client: Optional[QuiverQuantClient],
        cache_manager: DataTypeCacheManager,
        max_workers: int = 10
    ):
        # API clients
        self.tradier_client = tradier_client
        self.yahoo_client = yahoo_client
        self.fred_client = fred_client
        self.quiver_client = quiver_client
        self.cache_manager = cache_manager
        
        # Initialize service layers
        self.options_service = OptionsDataService(
            tradier_client, yahoo_client, cache_manager, max_workers // 2
        )
        
        self.quality_service = DataQualityService()
        
        self.fundamental_service = FundamentalDataService(
            yahoo_client, fred_client, quiver_client, cache_manager
        )
        
        self.technical_service = TechnicalAnalysisService(
            yahoo_client, quiver_client, cache_manager
        )
        
        # Processing configuration
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Economic context cache
        self._economic_context: Optional[Dict[str, EconomicIndicator]] = None
        self._economic_context_timestamp: Optional[datetime] = None
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @handle_errors(operation_name="collect_market_data")
    def collect_comprehensive_data(
        self, 
        symbol: str,
        request: Optional[DataRequest] = None
    ) -> DataCollectionResult:
        """
        Collect comprehensive market data for a single symbol.
        
        Args:
            symbol: Stock symbol to collect data for
            request: Optional data request specification
            
        Returns:
            Complete data collection result with quality validation
        """
        start_time = datetime.utcnow()
        
        if request is None:
            request = DataRequest(symbol=symbol)
        
        self.logger.info(f"Starting comprehensive data collection for {symbol}")
        
        result = DataCollectionResult(
            symbol=symbol,
            timestamp=start_time,
            success=False
        )
        
        try:
            # Collect all data components in parallel
            data_futures = {}
            
            # Stock quote (always needed)
            data_futures['stock_quote'] = self.executor.submit(
                self._collect_stock_quote, symbol
            )
            
            # Options data
            if request.include_options:
                data_futures['options'] = self.executor.submit(
                    self._collect_options_data, symbol, request.min_dte, request.max_dte
                )
            
            # Fundamental data
            if request.include_fundamentals:
                data_futures['fundamentals'] = self.executor.submit(
                    self._collect_fundamental_data, symbol
                )
            
            # Technical analysis
            if request.include_technicals:
                data_futures['technicals'] = self.executor.submit(
                    self._collect_technical_data, symbol
                )
            
            # Sentiment data
            if request.include_sentiment and self.quiver_client:
                data_futures['sentiment'] = self.executor.submit(
                    self._collect_sentiment_data, symbol
                )
            
            # Collect results
            collected_data = {}
            for data_type, future in data_futures.items():
                try:
                    collected_data[data_type] = future.result(timeout=60)
                except Exception as e:
                    self.logger.warning(f"Failed to collect {data_type} for {symbol}: {str(e)}")
                    result.errors.append(f"{data_type}: {str(e)}")
            
            # Validate minimum requirements
            if 'stock_quote' not in collected_data:
                raise DataQualityError(
                    f"Unable to get stock quote for {symbol}",
                    symbol=symbol
                )
            
            # Create market data object
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                stock_quote=collected_data.get('stock_quote'),
                options_chain=collected_data.get('options'),
                technical_indicators=collected_data.get('technicals', {}).get('indicators'),
                fundamental_data=collected_data.get('fundamentals', {}).get('data'),
                sentiment_data=collected_data.get('sentiment')
            )
            
            # Quality validation
            quality_reports = self._validate_data_quality(market_data, collected_data)
            result.quality_reports = quality_reports
            
            # Check quality threshold
            overall_quality = self._calculate_overall_quality(quality_reports)
            if overall_quality < request.quality_threshold:
                self.logger.warning(
                    f"Data quality for {symbol} below threshold: {overall_quality:.2f} < {request.quality_threshold}"
                )
                # Still return data but mark as lower quality
            
            # Store results
            result.market_data = market_data
            result.technical_summary = collected_data.get('technicals', {}).get('summary')
            result.fundamental_analysis = collected_data.get('fundamentals', {}).get('analysis')
            result.success = True
            
            # Calculate collection time
            end_time = datetime.utcnow()
            result.collection_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"Successfully collected data for {symbol} in {result.collection_time_ms:.1f}ms "
                f"(quality: {overall_quality:.2f})"
            )
            
        except Exception as e:
            result.errors.append(f"Collection failed: {str(e)}")
            self.logger.error(f"Data collection failed for {symbol}: {str(e)}")
        
        return result
    
    def collect_batch_data(
        self, 
        requests: List[DataRequest],
        max_concurrent: Optional[int] = None
    ) -> Tuple[List[DataCollectionResult], BatchCollectionSummary]:
        """
        Collect market data for multiple symbols in parallel.
        
        Args:
            requests: List of data collection requests
            max_concurrent: Maximum concurrent collections (defaults to max_workers)
            
        Returns:
            Tuple of (results list, batch summary)
        """
        start_time = datetime.utcnow()
        max_concurrent = max_concurrent or self.max_workers
        
        self.logger.info(f"Starting batch collection for {len(requests)} symbols")
        
        # Sort by priority (higher priority first)
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        
        results = []
        futures = []
        
        # Submit all requests
        for request in sorted_requests:
            future = self.executor.submit(
                self.collect_comprehensive_data,
                request.symbol,
                request
            )
            futures.append((request, future))
        
        # Collect results
        for request, future in futures:
            try:
                result = future.result(timeout=120)  # 2 minute timeout per symbol
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = DataCollectionResult(
                    symbol=request.symbol,
                    timestamp=datetime.utcnow(),
                    success=False,
                    errors=[f"Batch collection failed: {str(e)}"]
                )
                results.append(error_result)
                self.logger.error(f"Batch collection failed for {request.symbol}: {str(e)}")
        
        # Generate summary
        end_time = datetime.utcnow()
        summary = self._generate_batch_summary(results, start_time, end_time)
        
        self.logger.info(
            f"Batch collection completed: {summary.successful_collections}/{summary.total_symbols} "
            f"successful in {(end_time - start_time).total_seconds():.1f}s"
        )
        
        return results, summary
    
    def get_symbol_universe(
        self, 
        sectors: Optional[List[str]] = None,
        min_market_cap: Optional[int] = None,
        min_options_volume: Optional[int] = None
    ) -> List[str]:
        """
        Get filtered universe of symbols for scanning.
        
        Args:
            sectors: Optional list of GICS sectors to include
            min_market_cap: Minimum market capitalization
            min_options_volume: Minimum options volume
            
        Returns:
            List of symbols meeting criteria
        """
        # This would typically come from a symbol database or screener
        # For now, return a representative universe
        
        base_universe = [
            # Large cap tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            # Large cap finance
            "JPM", "BAC", "WFC", "GS", "MS", "C",
            # Large cap healthcare
            "JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY",
            # Large cap industrials
            "GE", "CAT", "BA", "MMM", "HON", "UPS",
            # Large cap consumer
            "WMT", "HD", "DIS", "MCD", "NKE", "SBUX",
            # Large cap energy
            "XOM", "CVX", "COP", "EOG", "SLB",
            # ETFs for flow analysis
            "SPY", "QQQ", "IWM", "VTI", "XLF", "XLK", "XLE", "XLV"
        ]
        
        # Apply filters (simplified implementation)
        filtered_universe = base_universe.copy()
        
        # Sector filtering would require fundamental data lookup
        if sectors:
            # This would filter by sector in a real implementation
            pass
        
        return filtered_universe
    
    def get_economic_context(self) -> Dict[str, EconomicIndicator]:
        """Get current economic context for market analysis."""
        
        # Check cache (update every 2 hours)
        if (self._economic_context and 
            self._economic_context_timestamp and
            datetime.utcnow() - self._economic_context_timestamp < timedelta(hours=2)):
            return self._economic_context
        
        try:
            # Get economic indicators
            economic_data = self.fred_client.get_economic_indicators_summary()
            self._economic_context = economic_data
            self._economic_context_timestamp = datetime.utcnow()
            
            self.logger.info(f"Updated economic context with {len(economic_data)} indicators")
            return economic_data
            
        except Exception as e:
            self.logger.warning(f"Failed to update economic context: {str(e)}")
            return self._economic_context or {}
    
    def _collect_stock_quote(self, symbol: str) -> StockQuote:
        """Collect stock quote data."""
        quote = self.tradier_client.get_stock_quote(symbol)
        if not quote:
            # Fallback to Yahoo Finance
            quote = self.yahoo_client.get_stock_quote(symbol)
        
        if not quote:
            raise DataQualityError(
                f"Unable to get stock quote for {symbol}",
                symbol=symbol
            )
        
        return quote
    
    def _collect_options_data(self, symbol: str, min_dte: int, max_dte: int) -> OptionsChain:
        """Collect options chain data."""
        return self.options_service.get_liquid_options_chain(
            symbol, min_dte, max_dte
        )
    
    def _collect_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Collect fundamental analysis data."""
        analysis = self.fundamental_service.get_comprehensive_analysis(symbol)
        
        return {
            'data': analysis.fundamental_data,
            'analysis': analysis
        }
    
    def _collect_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Collect technical analysis data."""
        summary = self.technical_service.get_comprehensive_analysis(symbol)
        
        return {
            'indicators': summary.trend_analysis,  # Simplified
            'summary': summary
        }
    
    def _collect_sentiment_data(self, symbol: str) -> Optional[SentimentData]:
        """Collect sentiment data."""
        if not self.quiver_client:
            return None
        
        return self.quiver_client.get_sentiment_summary(symbol)
    
    def _validate_data_quality(
        self, 
        market_data: MarketData, 
        collected_data: Dict[str, Any]
    ) -> List[QualityReport]:
        """Validate quality of collected data."""
        
        reports = []
        
        # Validate stock quote
        if market_data.stock_quote:
            stock_report = self.quality_service.validate_stock_quote(market_data.stock_quote)
            reports.append(stock_report)
        
        # Validate options chain
        if market_data.options_chain:
            options_report = self.quality_service.validate_options_chain(market_data.options_chain)
            reports.append(options_report)
        
        # Validate fundamental data
        if market_data.fundamental_data:
            fundamental_report = self.quality_service.validate_fundamental_data(market_data.fundamental_data)
            reports.append(fundamental_report)
        
        return reports
    
    def _calculate_overall_quality(self, reports: List[QualityReport]) -> float:
        """Calculate overall quality score from individual reports."""
        
        if not reports:
            return 0.0
        
        # Weight different data types
        weights = {
            'StockQuote': 0.3,
            'OptionsChain': 0.4,
            'FundamentalData': 0.2,
            'TechnicalIndicators': 0.1
        }
        
        weighted_scores = []
        total_weight = 0
        
        for report in reports:
            weight = weights.get(report.data_source, 0.1)
            weighted_scores.append(report.overall_score * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return sum(weighted_scores) / total_weight
    
    def _generate_batch_summary(
        self, 
        results: List[DataCollectionResult],
        start_time: datetime,
        end_time: datetime
    ) -> BatchCollectionSummary:
        """Generate summary of batch collection operation."""
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Calculate average collection time
        collection_times = [r.collection_time_ms for r in successful if r.collection_time_ms]
        avg_collection_time = statistics.mean(collection_times) if collection_times else 0.0
        
        # Calculate average quality score
        quality_scores = []
        for result in successful:
            if result.quality_reports:
                overall_quality = self._calculate_overall_quality(result.quality_reports)
                quality_scores.append(overall_quality)
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Error summary
        error_summary = {}
        for result in failed:
            for error in result.errors:
                error_type = error.split(':')[0] if ':' in error else 'Unknown'
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        return BatchCollectionSummary(
            total_symbols=len(results),
            successful_collections=len(successful),
            failed_collections=len(failed),
            avg_collection_time_ms=avg_collection_time,
            quality_score_avg=avg_quality,
            start_time=start_time,
            end_time=end_time,
            error_summary=error_summary
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all data services."""
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }
        
        # Check API clients
        services = {
            "tradier": self.tradier_client,
            "yahoo": self.yahoo_client,
            "fred": self.fred_client,
            "quiver": self.quiver_client
        }
        
        for service_name, client in services.items():
            if client:
                try:
                    is_healthy = client.health_check()
                    health_status["services"][service_name] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "circuit_breaker": client.get_circuit_breaker_state().value,
                        "rate_limit": client.get_rate_limit_status()
                    }
                    
                    if not is_healthy:
                        health_status["overall_status"] = "degraded"
                        
                except Exception as e:
                    health_status["services"][service_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["overall_status"] = "degraded"
            else:
                health_status["services"][service_name] = {
                    "status": "not_configured"
                }
        
        # Check cache status
        try:
            cache_stats = self.cache_manager.cache_manager.get_stats()
            health_status["cache"] = {
                "status": "healthy",
                "stats": cache_stats
            }
        except Exception as e:
            health_status["cache"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        return health_status
    
    def cleanup(self):
        """Cleanup orchestrator resources."""
        self.executor.shutdown(wait=True)
        self.options_service.cleanup()
        self.logger.info("Market data orchestrator cleanup completed")