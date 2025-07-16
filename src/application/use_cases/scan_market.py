"""
Market scanning use case for the Options Trading Engine.
Scans market for opportunities and updates watchlist dynamically.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..config.settings import get_config, get_trading_config
from ...domain.services.market_data_orchestrator import MarketDataOrchestrator
from ...domain.services.technical_analysis_service import TechnicalAnalysisService
from ...domain.services.fundamental_data_service import FundamentalDataService
from ...domain.services.data_quality_service import DataQualityService
from ...infrastructure.error_handling import handle_errors, ApplicationError
from ...infrastructure.monitoring.monitoring_system import get_monitoring_system
from ...infrastructure.performance.performance_optimizer import get_performance_optimizer


class ScanType(Enum):
    """Types of market scans."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    EARNINGS = "earnings"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SECTOR_ROTATION = "sector_rotation"
    FLOW = "flow"
    SENTIMENT = "sentiment"
    OPTIONS_ACTIVITY = "options_activity"


@dataclass
class ScanCriteria:
    """Criteria for market scanning."""
    scan_types: List[ScanType]
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_volume: Optional[int] = None
    min_options_volume: Optional[int] = None
    sectors: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None
    min_iv_rank: Optional[float] = None
    max_iv_rank: Optional[float] = None
    days_to_earnings_range: Optional[Tuple[int, int]] = None
    custom_filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default values."""
        if self.min_market_cap is None:
            self.min_market_cap = 1_000_000_000  # $1B minimum
        if self.min_volume is None:
            self.min_volume = 1_000_000  # 1M shares
        if self.min_options_volume is None:
            self.min_options_volume = 1000  # 1K contracts
        if self.custom_filters is None:
            self.custom_filters = {}


@dataclass
class ScanResult:
    """Result from market scan."""
    symbol: str
    company_name: str
    sector: str
    market_cap: float
    current_price: float
    volume: int
    scan_score: float
    scan_reasons: List[str]
    technical_indicators: Dict[str, float]
    fundamental_metrics: Dict[str, float]
    options_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'current_price': self.current_price,
            'volume': self.volume,
            'scan_score': self.scan_score,
            'scan_reasons': self.scan_reasons,
            'technical_indicators': self.technical_indicators,
            'fundamental_metrics': self.fundamental_metrics,
            'options_metrics': self.options_metrics,
            'risk_metrics': self.risk_metrics,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class MarketScanResponse:
    """Response from market scanning."""
    results: List[ScanResult]
    summary: Dict[str, Any]
    scan_criteria: ScanCriteria
    execution_time: float
    data_quality_score: float
    warnings: List[str]
    errors: List[str]
    
    @property
    def success(self) -> bool:
        """Check if scan was successful."""
        return len(self.errors) == 0


class ScanMarketUseCase:
    """
    Use case for scanning the market for trading opportunities.
    
    This implements various market scanning strategies:
    - Momentum scanning for trending stocks
    - Mean reversion scanning for oversold/overbought conditions
    - Volatility scanning for high/low IV situations
    - Earnings scanning for upcoming earnings plays
    - Technical scanning for chart patterns
    - Fundamental scanning for value/growth opportunities
    - Sector rotation scanning
    - Flow scanning for unusual activity
    - Sentiment scanning for market sentiment shifts
    - Options activity scanning for unusual options flow
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize services
        self.market_data_orchestrator = MarketDataOrchestrator()
        self.technical_analysis_service = TechnicalAnalysisService()
        self.fundamental_data_service = FundamentalDataService()
        self.data_quality_service = DataQualityService()
        
        # Infrastructure services
        self.monitoring_system = get_monitoring_system()
        self.performance_optimizer = get_performance_optimizer()
        
        # Configuration
        self.config = get_config()
        self.trading_config = get_trading_config()
        
        # Universe of stocks to scan (in production, this would be broader)
        self.scan_universe = self._get_scan_universe()
    
    def _get_scan_universe(self) -> List[str]:
        """Get universe of symbols to scan."""
        # In production, this would fetch from a broader universe
        # For now, use extended watchlist
        base_symbols = [
            # Major indices
            'SPY', 'QQQ', 'IWM', 'DIA',
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB',
            # Industrials
            'BA', 'CAT', 'GE', 'MMM', 'HON',
            # Communication
            'VZ', 'T', 'NFLX', 'DIS', 'CMCSA',
            # Utilities
            'NEE', 'D', 'SO', 'AEP', 'EXC',
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG',
            # Materials
            'LIN', 'APD', 'FCX', 'NEM', 'DOW'
        ]
        
        return base_symbols
    
    @handle_errors(operation_name="scan_market")
    def execute(self, scan_criteria: ScanCriteria) -> MarketScanResponse:
        """
        Execute market scan with specified criteria.
        
        Args:
            scan_criteria: Criteria for market scanning
            
        Returns:
            MarketScanResponse with scan results
        """
        start_time = datetime.now()
        warnings = []
        errors = []
        
        try:
            with self.performance_optimizer.performance_tracking("market_scan_workflow"):
                # Step 1: Collect market data for scan universe
                self.logger.info("Step 1: Collecting market data for scan universe...")
                market_data = self._collect_scan_data()
                
                # Validate data quality
                data_quality_score = self.data_quality_service.validate_data_quality(market_data)
                if data_quality_score < 0.7:
                    warnings.append(f"Low data quality score: {data_quality_score:.2f}")
                
                # Step 2: Apply basic filters
                self.logger.info("Step 2: Applying basic filters...")
                filtered_symbols = self._apply_basic_filters(market_data, scan_criteria)
                
                if not filtered_symbols:
                    errors.append("No symbols passed basic filters")
                    return self._create_error_response(scan_criteria, errors, warnings, start_time)
                
                # Step 3: Run scan strategies
                self.logger.info("Step 3: Running scan strategies...")
                scan_results = self._run_scan_strategies(filtered_symbols, market_data, scan_criteria)
                
                if not scan_results:
                    errors.append("No results from scan strategies")
                    return self._create_error_response(scan_criteria, errors, warnings, start_time)
                
                # Step 4: Calculate composite scores
                self.logger.info("Step 4: Calculating composite scores...")
                scored_results = self._calculate_composite_scores(scan_results, scan_criteria)
                
                # Step 5: Rank and filter results
                self.logger.info("Step 5: Ranking and filtering results...")
                final_results = self._rank_and_filter_results(scored_results, scan_criteria)
                
                # Step 6: Generate summary
                summary = self._generate_scan_summary(final_results, market_data, scan_criteria)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                self.logger.info(f"Market scan completed successfully. Found {len(final_results)} opportunities in {execution_time:.2f}s")
                
                return MarketScanResponse(
                    results=final_results,
                    summary=summary,
                    scan_criteria=scan_criteria,
                    execution_time=execution_time,
                    data_quality_score=data_quality_score,
                    warnings=warnings,
                    errors=errors
                )
        
        except Exception as e:
            errors.append(f"Market scan failed: {str(e)}")
            return self._create_error_response(scan_criteria, errors, warnings, start_time)
    
    def _collect_scan_data(self) -> Dict[str, Any]:
        """Collect market data for scan universe."""
        try:
            # Collect comprehensive market data
            market_data = self.market_data_orchestrator.collect_comprehensive_data(
                symbols=self.scan_universe,
                include_options=True,
                include_fundamentals=True,
                include_technical=True
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Scan data collection failed: {str(e)}")
            raise ApplicationError(f"Failed to collect scan data: {str(e)}")
    
    def _apply_basic_filters(self, market_data: Dict[str, Any], criteria: ScanCriteria) -> List[str]:
        """Apply basic filters to scan universe."""
        try:
            filtered_symbols = []
            
            for symbol, data in market_data.get('symbols', {}).items():
                try:
                    # Market cap filter
                    market_cap = data.get('market_cap', 0)
                    if criteria.min_market_cap and market_cap < criteria.min_market_cap:
                        continue
                    if criteria.max_market_cap and market_cap > criteria.max_market_cap:
                        continue
                    
                    # Volume filter
                    volume = data.get('volume', 0)
                    if criteria.min_volume and volume < criteria.min_volume:
                        continue
                    
                    # Options volume filter
                    options_volume = data.get('options_volume', 0)
                    if criteria.min_options_volume and options_volume < criteria.min_options_volume:
                        continue
                    
                    # Sector filter
                    sector = data.get('sector', '')
                    if criteria.sectors and sector not in criteria.sectors:
                        continue
                    if criteria.exclude_sectors and sector in criteria.exclude_sectors:
                        continue
                    
                    # IV rank filter
                    iv_rank = data.get('iv_rank', 0)
                    if criteria.min_iv_rank and iv_rank < criteria.min_iv_rank:
                        continue
                    if criteria.max_iv_rank and iv_rank > criteria.max_iv_rank:
                        continue
                    
                    # Days to earnings filter
                    if criteria.days_to_earnings_range:
                        days_to_earnings = data.get('days_to_earnings', 999)
                        min_days, max_days = criteria.days_to_earnings_range
                        if not (min_days <= days_to_earnings <= max_days):
                            continue
                    
                    # Custom filters
                    passes_custom = True
                    for filter_name, filter_value in criteria.custom_filters.items():
                        if filter_name in data:
                            if isinstance(filter_value, dict):
                                # Range filter
                                if 'min' in filter_value and data[filter_name] < filter_value['min']:
                                    passes_custom = False
                                    break
                                if 'max' in filter_value and data[filter_name] > filter_value['max']:
                                    passes_custom = False
                                    break
                            else:
                                # Exact match filter
                                if data[filter_name] != filter_value:
                                    passes_custom = False
                                    break
                    
                    if not passes_custom:
                        continue
                    
                    filtered_symbols.append(symbol)
                
                except Exception as e:
                    self.logger.warning(f"Filter evaluation failed for {symbol}: {str(e)}")
                    continue
            
            self.logger.info(f"Basic filters: {len(filtered_symbols)} symbols passed (filtered from {len(market_data.get('symbols', {}))})")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Basic filtering failed: {str(e)}")
            raise ApplicationError(f"Failed to apply basic filters: {str(e)}")
    
    def _run_scan_strategies(self, symbols: List[str], market_data: Dict[str, Any], criteria: ScanCriteria) -> List[ScanResult]:
        """Run scan strategies on filtered symbols."""
        try:
            scan_results = []
            
            # Use parallel processing for scan strategies
            def scan_symbol(symbol):
                try:
                    symbol_data = market_data['symbols'][symbol]
                    result = self._scan_single_symbol(symbol, symbol_data, criteria)
                    return result
                except Exception as e:
                    self.logger.warning(f"Scan failed for {symbol}: {str(e)}")
                    return None
            
            # Process symbols in parallel
            results = self.performance_optimizer.parallel_process(
                scan_symbol,
                symbols,
                max_workers=8
            )
            
            # Filter out failed scans
            scan_results = [r for r in results if r is not None]
            
            self.logger.info(f"Scan strategies completed: {len(scan_results)} results from {len(symbols)} symbols")
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Scan strategies failed: {str(e)}")
            raise ApplicationError(f"Failed to run scan strategies: {str(e)}")
    
    def _scan_single_symbol(self, symbol: str, symbol_data: Dict[str, Any], criteria: ScanCriteria) -> Optional[ScanResult]:
        """Run scan strategies on a single symbol."""
        try:
            scan_reasons = []
            technical_indicators = {}
            fundamental_metrics = {}
            options_metrics = {}
            risk_metrics = {}
            
            # Run each scan type
            for scan_type in criteria.scan_types:
                if scan_type == ScanType.MOMENTUM:
                    momentum_result = self._scan_momentum(symbol_data)
                    if momentum_result['signal']:
                        scan_reasons.extend(momentum_result['reasons'])
                        technical_indicators.update(momentum_result['indicators'])
                
                elif scan_type == ScanType.MEAN_REVERSION:
                    mean_reversion_result = self._scan_mean_reversion(symbol_data)
                    if mean_reversion_result['signal']:
                        scan_reasons.extend(mean_reversion_result['reasons'])
                        technical_indicators.update(mean_reversion_result['indicators'])
                
                elif scan_type == ScanType.VOLATILITY:
                    volatility_result = self._scan_volatility(symbol_data)
                    if volatility_result['signal']:
                        scan_reasons.extend(volatility_result['reasons'])
                        options_metrics.update(volatility_result['metrics'])
                
                elif scan_type == ScanType.EARNINGS:
                    earnings_result = self._scan_earnings(symbol_data)
                    if earnings_result['signal']:
                        scan_reasons.extend(earnings_result['reasons'])
                        fundamental_metrics.update(earnings_result['metrics'])
                
                elif scan_type == ScanType.TECHNICAL:
                    technical_result = self._scan_technical(symbol_data)
                    if technical_result['signal']:
                        scan_reasons.extend(technical_result['reasons'])
                        technical_indicators.update(technical_result['indicators'])
                
                elif scan_type == ScanType.FUNDAMENTAL:
                    fundamental_result = self._scan_fundamental(symbol_data)
                    if fundamental_result['signal']:
                        scan_reasons.extend(fundamental_result['reasons'])
                        fundamental_metrics.update(fundamental_result['metrics'])
                
                elif scan_type == ScanType.OPTIONS_ACTIVITY:
                    options_result = self._scan_options_activity(symbol_data)
                    if options_result['signal']:
                        scan_reasons.extend(options_result['reasons'])
                        options_metrics.update(options_result['metrics'])
            
            # Only create result if we have scan reasons
            if scan_reasons:
                return ScanResult(
                    symbol=symbol,
                    company_name=symbol_data.get('company_name', ''),
                    sector=symbol_data.get('sector', ''),
                    market_cap=symbol_data.get('market_cap', 0),
                    current_price=symbol_data.get('current_price', 0),
                    volume=symbol_data.get('volume', 0),
                    scan_score=0.0,  # Will be calculated later
                    scan_reasons=scan_reasons,
                    technical_indicators=technical_indicators,
                    fundamental_metrics=fundamental_metrics,
                    options_metrics=options_metrics,
                    risk_metrics=risk_metrics,
                    last_updated=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Single symbol scan failed for {symbol}: {str(e)}")
            return None
    
    def _scan_momentum(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for momentum signals."""
        try:
            technical_data = symbol_data.get('technical', {})
            
            # Get momentum indicators
            rsi = technical_data.get('rsi', 50)
            macd = technical_data.get('macd', 0)
            macd_signal = technical_data.get('macd_signal', 0)
            momentum_z = technical_data.get('momentum_z', 0)
            
            signal = False
            reasons = []
            indicators = {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'momentum_z': momentum_z
            }
            
            # Momentum signals
            if momentum_z > 1.5:
                signal = True
                reasons.append(f"Strong upward momentum (Z-score: {momentum_z:.2f})")
            
            if macd > macd_signal and macd > 0:
                signal = True
                reasons.append("MACD bullish crossover")
            
            if 50 < rsi < 70:
                signal = True
                reasons.append(f"RSI in momentum zone ({rsi:.1f})")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Momentum scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'indicators': {}}
    
    def _scan_mean_reversion(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for mean reversion signals."""
        try:
            technical_data = symbol_data.get('technical', {})
            
            # Get mean reversion indicators
            rsi = technical_data.get('rsi', 50)
            bollinger_position = technical_data.get('bollinger_position', 0.5)
            momentum_z = technical_data.get('momentum_z', 0)
            
            signal = False
            reasons = []
            indicators = {
                'rsi': rsi,
                'bollinger_position': bollinger_position,
                'momentum_z': momentum_z
            }
            
            # Mean reversion signals
            if rsi < 30:
                signal = True
                reasons.append(f"RSI oversold ({rsi:.1f})")
            
            if bollinger_position < 0.2:
                signal = True
                reasons.append(f"Near lower Bollinger Band ({bollinger_position:.2f})")
            
            if momentum_z < -1.5:
                signal = True
                reasons.append(f"Strong downward momentum reversal opportunity (Z-score: {momentum_z:.2f})")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Mean reversion scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'indicators': {}}
    
    def _scan_volatility(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for volatility signals."""
        try:
            options_data = symbol_data.get('options', {})
            
            # Get volatility metrics
            iv_rank = options_data.get('iv_rank', 0)
            iv_percentile = options_data.get('iv_percentile', 0)
            hv_iv_ratio = options_data.get('hv_iv_ratio', 1.0)
            
            signal = False
            reasons = []
            metrics = {
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'hv_iv_ratio': hv_iv_ratio
            }
            
            # High IV signals (good for selling premium)
            if iv_rank > 70:
                signal = True
                reasons.append(f"High IV rank ({iv_rank:.1f}%) - premium selling opportunity")
            
            # Low IV signals (good for buying options)
            if iv_rank < 30:
                signal = True
                reasons.append(f"Low IV rank ({iv_rank:.1f}%) - premium buying opportunity")
            
            # IV vs HV divergence
            if hv_iv_ratio < 0.8:
                signal = True
                reasons.append(f"IV elevated vs HV (ratio: {hv_iv_ratio:.2f})")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Volatility scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'metrics': {}}
    
    def _scan_earnings(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for earnings-related signals."""
        try:
            fundamental_data = symbol_data.get('fundamental', {})
            
            # Get earnings metrics
            days_to_earnings = fundamental_data.get('days_to_earnings', 999)
            earnings_surprise_history = fundamental_data.get('earnings_surprise_history', [])
            revenue_growth = fundamental_data.get('revenue_growth', 0)
            
            signal = False
            reasons = []
            metrics = {
                'days_to_earnings': days_to_earnings,
                'revenue_growth': revenue_growth
            }
            
            # Earnings proximity signals
            if 7 <= days_to_earnings <= 30:
                signal = True
                reasons.append(f"Earnings in {days_to_earnings} days")
            
            # Earnings surprise history
            if earnings_surprise_history:
                avg_surprise = sum(earnings_surprise_history[-4:]) / min(4, len(earnings_surprise_history))
                if avg_surprise > 0.05:  # 5% average beat
                    signal = True
                    reasons.append(f"Strong earnings surprise history (avg: {avg_surprise*100:.1f}%)")
            
            # Revenue growth
            if revenue_growth > 0.15:  # 15% revenue growth
                signal = True
                reasons.append(f"Strong revenue growth ({revenue_growth*100:.1f}%)")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Earnings scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'metrics': {}}
    
    def _scan_technical(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for technical analysis signals."""
        try:
            technical_data = symbol_data.get('technical', {})
            
            # Get technical indicators
            current_price = symbol_data.get('current_price', 0)
            sma_20 = technical_data.get('sma_20', 0)
            sma_50 = technical_data.get('sma_50', 0)
            support_level = technical_data.get('support_level', 0)
            resistance_level = technical_data.get('resistance_level', 0)
            
            signal = False
            reasons = []
            indicators = {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'support_level': support_level,
                'resistance_level': resistance_level
            }
            
            # Moving average signals
            if current_price > sma_20 > sma_50:
                signal = True
                reasons.append("Bullish moving average alignment")
            
            # Support/resistance signals
            if support_level > 0 and abs(current_price - support_level) / current_price < 0.02:
                signal = True
                reasons.append(f"Near support level ({support_level:.2f})")
            
            if resistance_level > 0 and abs(current_price - resistance_level) / current_price < 0.02:
                signal = True
                reasons.append(f"Near resistance level ({resistance_level:.2f})")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Technical scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'indicators': {}}
    
    def _scan_fundamental(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for fundamental analysis signals."""
        try:
            fundamental_data = symbol_data.get('fundamental', {})
            
            # Get fundamental metrics
            pe_ratio = fundamental_data.get('pe_ratio', 0)
            peg_ratio = fundamental_data.get('peg_ratio', 0)
            debt_to_equity = fundamental_data.get('debt_to_equity', 0)
            roe = fundamental_data.get('roe', 0)
            
            signal = False
            reasons = []
            metrics = {
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'debt_to_equity': debt_to_equity,
                'roe': roe
            }
            
            # Value signals
            if 0 < pe_ratio < 15:
                signal = True
                reasons.append(f"Low P/E ratio ({pe_ratio:.1f})")
            
            if 0 < peg_ratio < 1.0:
                signal = True
                reasons.append(f"Attractive PEG ratio ({peg_ratio:.2f})")
            
            # Quality signals
            if roe > 0.15:  # 15% ROE
                signal = True
                reasons.append(f"Strong ROE ({roe*100:.1f}%)")
            
            if debt_to_equity < 0.3:  # Low debt
                signal = True
                reasons.append(f"Low debt-to-equity ({debt_to_equity:.2f})")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'metrics': {}}
    
    def _scan_options_activity(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for unusual options activity."""
        try:
            options_data = symbol_data.get('options', {})
            
            # Get options activity metrics
            options_volume = options_data.get('volume', 0)
            options_volume_avg = options_data.get('volume_avg_20d', 1)
            put_call_ratio = options_data.get('put_call_ratio', 1.0)
            flow_z = options_data.get('flow_z', 0)
            
            signal = False
            reasons = []
            metrics = {
                'options_volume': options_volume,
                'options_volume_avg': options_volume_avg,
                'put_call_ratio': put_call_ratio,
                'flow_z': flow_z
            }
            
            # Unusual volume
            volume_ratio = options_volume / max(options_volume_avg, 1)
            if volume_ratio > 2.0:
                signal = True
                reasons.append(f"Unusual options volume ({volume_ratio:.1f}x average)")
            
            # Unusual flow
            if abs(flow_z) > 2.0:
                direction = "bullish" if flow_z > 0 else "bearish"
                signal = True
                reasons.append(f"Unusual {direction} options flow (Z-score: {flow_z:.2f})")
            
            # Put/call ratio signals
            if put_call_ratio > 1.5:
                signal = True
                reasons.append(f"High put/call ratio ({put_call_ratio:.2f}) - potential contrarian signal")
            
            return {
                'signal': signal,
                'reasons': reasons,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Options activity scan failed: {str(e)}")
            return {'signal': False, 'reasons': [], 'metrics': {}}
    
    def _calculate_composite_scores(self, scan_results: List[ScanResult], criteria: ScanCriteria) -> List[ScanResult]:
        """Calculate composite scores for scan results."""
        try:
            # Score weights based on scan types
            scan_type_weights = {
                ScanType.MOMENTUM: 0.2,
                ScanType.MEAN_REVERSION: 0.2,
                ScanType.VOLATILITY: 0.15,
                ScanType.EARNINGS: 0.15,
                ScanType.TECHNICAL: 0.1,
                ScanType.FUNDAMENTAL: 0.1,
                ScanType.OPTIONS_ACTIVITY: 0.1
            }
            
            for result in scan_results:
                score = 0.0
                
                # Base score from number of scan reasons
                base_score = len(result.scan_reasons) * 10
                
                # Bonus for specific criteria
                for scan_type in criteria.scan_types:
                    weight = scan_type_weights.get(scan_type, 0.1)
                    
                    if scan_type == ScanType.MOMENTUM:
                        momentum_z = result.technical_indicators.get('momentum_z', 0)
                        score += weight * min(abs(momentum_z) * 20, 100)
                    
                    elif scan_type == ScanType.VOLATILITY:
                        iv_rank = result.options_metrics.get('iv_rank', 0)
                        # Higher score for extreme IV ranks
                        iv_score = max(iv_rank, 100 - iv_rank)
                        score += weight * iv_score
                    
                    elif scan_type == ScanType.OPTIONS_ACTIVITY:
                        flow_z = result.options_metrics.get('flow_z', 0)
                        score += weight * min(abs(flow_z) * 25, 100)
                
                # Combine base score and weighted score
                result.scan_score = base_score + score
            
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Composite score calculation failed: {str(e)}")
            return scan_results
    
    def _rank_and_filter_results(self, scan_results: List[ScanResult], criteria: ScanCriteria) -> List[ScanResult]:
        """Rank and filter scan results."""
        try:
            # Sort by score (descending)
            sorted_results = sorted(scan_results, key=lambda x: x.scan_score, reverse=True)
            
            # Take top results (configurable)
            max_results = self.config.get('scanning', {}).get('max_results', 50)
            top_results = sorted_results[:max_results]
            
            # Filter by minimum score
            min_score = self.config.get('scanning', {}).get('min_score', 20)
            filtered_results = [r for r in top_results if r.scan_score >= min_score]
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Result ranking and filtering failed: {str(e)}")
            return scan_results
    
    def _generate_scan_summary(self, results: List[ScanResult], market_data: Dict[str, Any], criteria: ScanCriteria) -> Dict[str, Any]:
        """Generate summary of scan results."""
        try:
            summary = {
                'total_results': len(results),
                'scan_types': [st.value for st in criteria.scan_types],
                'average_score': sum(r.scan_score for r in results) / len(results) if results else 0,
                'top_sectors': self._get_top_sectors(results),
                'scan_reasons_breakdown': self._get_scan_reasons_breakdown(results),
                'market_data_coverage': len(market_data.get('symbols', {})),
                'scan_timestamp': datetime.now().isoformat()
            }
            
            # Add score distribution
            if results:
                scores = [r.scan_score for r in results]
                summary['score_distribution'] = {
                    'min': min(scores),
                    'max': max(scores),
                    'median': sorted(scores)[len(scores)//2],
                    'q1': sorted(scores)[len(scores)//4],
                    'q3': sorted(scores)[3*len(scores)//4]
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return {'error': f"Failed to generate summary: {str(e)}"}
    
    def _get_top_sectors(self, results: List[ScanResult]) -> Dict[str, int]:
        """Get top sectors from scan results."""
        sector_counts = {}
        for result in results:
            sector = result.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Sort by count and return top 10
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_sectors[:10])
    
    def _get_scan_reasons_breakdown(self, results: List[ScanResult]) -> Dict[str, int]:
        """Get breakdown of scan reasons."""
        reason_counts = {}
        for result in results:
            for reason in result.scan_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Sort by count and return top 20
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_reasons[:20])
    
    def _create_error_response(self, criteria: ScanCriteria, errors: List[str], warnings: List[str], start_time: datetime) -> MarketScanResponse:
        """Create error response."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MarketScanResponse(
            results=[],
            summary={'error': 'Market scan failed', 'execution_time': execution_time},
            scan_criteria=criteria,
            execution_time=execution_time,
            data_quality_score=0.0,
            warnings=warnings,
            errors=errors
        )