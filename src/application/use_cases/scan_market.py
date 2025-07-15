"""Market scanning use case."""

from typing import List, Optional
from datetime import datetime
from decimal import Decimal

from ...domain.entities.strategy import StrategyType
from ...data.models.trades import TradeCandidate, TradeFilterCriteria
from ...data.models.market_data import StockQuote
from ...data.models.options import OptionsChain


class MarketScanRequest:
    """Request for market scanning."""
    
    def __init__(
        self,
        scan_type: str = "broad_market",
        symbols: Optional[List[str]] = None,
        strategy_types: Optional[List[StrategyType]] = None,
        filter_criteria: Optional[TradeFilterCriteria] = None
    ):
        self.scan_type = scan_type
        self.symbols = symbols or []
        self.strategy_types = strategy_types or []
        self.filter_criteria = filter_criteria or TradeFilterCriteria()
        self.timestamp = datetime.utcnow()


class MarketScanResponse:
    """Response from market scanning."""
    
    def __init__(self, candidates: List[TradeCandidate], scan_stats: dict):
        self.candidates = candidates
        self.scan_stats = scan_stats
        self.timestamp = datetime.utcnow()


class MarketScanUseCase:
    """Use case for scanning market for trade opportunities."""
    
    def __init__(self, market_repo, options_repo, strategy_generator, scoring_engine):
        self.market_repo = market_repo
        self.options_repo = options_repo
        self.strategy_generator = strategy_generator
        self.scoring_engine = scoring_engine
    
    async def execute(self, request: MarketScanRequest) -> MarketScanResponse:
        """Execute market scan."""
        candidates = []
        stats = {
            "symbols_scanned": 0,
            "strategies_generated": 0,
            "candidates_found": 0,
            "scan_duration": 0
        }
        
        start_time = datetime.utcnow()
        
        # Get symbols to scan
        symbols_to_scan = self._get_scan_symbols(request)
        
        # Scan each symbol
        for symbol in symbols_to_scan:
            symbol_candidates = await self._scan_symbol(symbol, request)
            candidates.extend(symbol_candidates)
            stats["symbols_scanned"] += 1
        
        # Apply final filtering and scoring
        filtered_candidates = self._filter_candidates(candidates, request.filter_criteria)
        
        stats["candidates_found"] = len(filtered_candidates)
        stats["scan_duration"] = (datetime.utcnow() - start_time).total_seconds()
        
        return MarketScanResponse(filtered_candidates, stats)
    
    def _get_scan_symbols(self, request: MarketScanRequest) -> List[str]:
        """Get symbols to scan based on scan type."""
        if request.scan_type == "portfolio":
            return request.symbols
        elif request.scan_type == "broad_market":
            # Return common liquid symbols for broad market scan
            return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        else:
            return request.symbols
    
    async def _scan_symbol(self, symbol: str, request: MarketScanRequest) -> List[TradeCandidate]:
        """Scan a single symbol for opportunities."""
        candidates = []
        
        # Get market data
        stock_quote = await self.market_repo.get_stock_quote(symbol)
        options_chain = await self.options_repo.get_options_chain(symbol)
        
        if not stock_quote or not options_chain:
            return candidates
        
        # Generate strategies
        strategy_types = request.strategy_types or [
            StrategyType.PUT_CREDIT_SPREAD,
            StrategyType.CALL_CREDIT_SPREAD,
            StrategyType.IRON_CONDOR
        ]
        
        for strategy_type in strategy_types:
            strategies = await self.strategy_generator.generate_strategies(
                strategy_type, options_chain, stock_quote
            )
            
            for strategy in strategies:
                candidate = TradeCandidate(
                    strategy=strategy,
                    sector=await self._get_sector(symbol),
                    generated_at=datetime.utcnow()
                )
                candidates.append(candidate)
        
        return candidates
    
    def _filter_candidates(self, candidates: List[TradeCandidate], criteria: TradeFilterCriteria) -> List[TradeCandidate]:
        """Apply filtering criteria to candidates."""
        filtered = []
        
        for candidate in candidates:
            if self._meets_criteria(candidate, criteria):
                filtered.append(candidate)
        
        return filtered
    
    def _meets_criteria(self, candidate: TradeCandidate, criteria: TradeFilterCriteria) -> bool:
        """Check if candidate meets filtering criteria."""
        # Check probability of profit
        if (candidate.probability_of_profit is None or 
            candidate.probability_of_profit < criteria.min_probability_of_profit):
            return False
        
        # Check credit to max loss ratio
        if candidate.strategy.credit_to_max_loss_ratio is None or candidate.strategy.credit_to_max_loss_ratio < criteria.min_credit_to_max_loss:
            return False
        
        # Check max loss per trade
        if candidate.max_loss and candidate.max_loss > criteria.max_loss_per_trade:
            return False
        
        # Check quote age
        if candidate.quote_age_minutes and candidate.quote_age_minutes > criteria.max_quote_age_minutes:
            return False
        
        return True
    
    async def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        # Placeholder implementation
        return "Technology"