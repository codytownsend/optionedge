"""Trade generation use case."""

from typing import List, Optional, Dict
from datetime import datetime
from decimal import Decimal

from ...data.models.trades import TradeCandidate, TradeFilterCriteria
from ...data.models.portfolios import Portfolio
from .scan_market import MarketScanUseCase, MarketScanRequest


class TradeGenerationRequest:
    """Request for trade generation."""
    
    def __init__(
        self,
        nav: Decimal,
        portfolio: Optional[Portfolio] = None,
        max_trades: int = 5,
        scan_type: str = "broad_market",
        symbols: Optional[List[str]] = None,
        custom_criteria: Optional[TradeFilterCriteria] = None
    ):
        self.nav = nav
        self.portfolio = portfolio
        self.max_trades = max_trades
        self.scan_type = scan_type
        self.symbols = symbols or []
        self.custom_criteria = custom_criteria
        self.timestamp = datetime.utcnow()


class TradeGenerationResponse:
    """Response from trade generation."""
    
    def __init__(
        self,
        selected_trades: List[TradeCandidate],
        rejected_trades: List[TradeCandidate],
        portfolio_impact: Dict,
        generation_stats: Dict
    ):
        self.selected_trades = selected_trades
        self.rejected_trades = rejected_trades
        self.portfolio_impact = portfolio_impact
        self.generation_stats = generation_stats
        self.timestamp = datetime.utcnow()


class TradeGenerationUseCase:
    """Use case for generating the top 5 trade recommendations."""
    
    def __init__(
        self,
        market_scan_use_case: MarketScanUseCase,
        scoring_engine,
        risk_calculator,
        portfolio_manager
    ):
        self.market_scan_use_case = market_scan_use_case
        self.scoring_engine = scoring_engine
        self.risk_calculator = risk_calculator
        self.portfolio_manager = portfolio_manager
    
    async def execute(self, request: TradeGenerationRequest) -> TradeGenerationResponse:
        """Execute trade generation process."""
        start_time = datetime.utcnow()
        
        # Step 1: Market scan
        scan_request = MarketScanRequest(
            scan_type=request.scan_type,
            symbols=request.symbols,
            filter_criteria=self._build_filter_criteria(request)
        )
        
        scan_response = await self.market_scan_use_case.execute(scan_request)
        candidates = scan_response.candidates
        
        # Step 2: Score all candidates
        scored_candidates = await self._score_candidates(candidates)
        
        # Step 3: Apply portfolio-level constraints
        portfolio_filtered = self._apply_portfolio_constraints(
            scored_candidates, request.portfolio, request.nav
        )
        
        # Step 4: Rank and select top trades
        ranked_candidates = self._rank_candidates(portfolio_filtered)
        selected_trades = self._select_top_trades(ranked_candidates, request.max_trades)
        
        # Step 5: Calculate portfolio impact
        portfolio_impact = self._calculate_portfolio_impact(
            selected_trades, request.portfolio, request.nav
        )
        
        # Step 6: Generate thesis for selected trades
        for trade in selected_trades:
            trade.thesis = await self._generate_thesis(trade)
        
        # Prepare response
        rejected_trades = [c for c in ranked_candidates if c not in selected_trades]
        generation_stats = {
            "total_candidates": len(candidates),
            "scored_candidates": len(scored_candidates),
            "portfolio_filtered": len(portfolio_filtered),
            "selected_trades": len(selected_trades),
            "generation_time": (datetime.utcnow() - start_time).total_seconds()
        }
        
        return TradeGenerationResponse(
            selected_trades=selected_trades,
            rejected_trades=rejected_trades,
            portfolio_impact=portfolio_impact,
            generation_stats=generation_stats
        )
    
    def _build_filter_criteria(self, request: TradeGenerationRequest) -> TradeFilterCriteria:
        """Build filter criteria from request."""
        if request.custom_criteria:
            return request.custom_criteria
        
        return TradeFilterCriteria(
            nav=request.nav,
            available_capital=request.nav * Decimal('0.05')  # 5% of NAV per trade
        )
    
    async def _score_candidates(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """Score all candidates using the scoring engine."""
        scored = []
        
        for candidate in candidates:
            score = await self.scoring_engine.calculate_score(candidate)
            candidate.model_score = score
            scored.append(candidate)
        
        return scored
    
    def _apply_portfolio_constraints(
        self, 
        candidates: List[TradeCandidate], 
        portfolio: Optional[Portfolio], 
        nav: Decimal
    ) -> List[TradeCandidate]:
        """Apply portfolio-level constraints."""
        if not portfolio:
            return candidates
        
        filtered = []
        sector_counts = {}
        
        for candidate in candidates:
            # Check sector diversification
            sector = candidate.sector or "Unknown"
            if sector_counts.get(sector, 0) >= 2:
                continue
            
            # Check portfolio Greeks limits
            if not self._check_portfolio_greeks(candidate, portfolio, nav):
                continue
            
            filtered.append(candidate)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        return filtered
    
    def _check_portfolio_greeks(
        self, 
        candidate: TradeCandidate, 
        portfolio: Portfolio, 
        nav: Decimal
    ) -> bool:
        """Check if adding this trade would violate portfolio Greeks limits."""
        # Calculate portfolio Greeks after adding this trade
        portfolio_delta = portfolio.total_delta + (candidate.strategy.net_delta or 0)
        portfolio_vega = portfolio.total_vega + (candidate.strategy.net_vega or 0)
        
        # Check delta limits: [-0.30, +0.30] × (NAV/100k)
        nav_factor = nav / Decimal('100000')
        max_delta = float(Decimal('0.30') * nav_factor)
        
        if abs(portfolio_delta) > max_delta:
            return False
        
        # Check vega limits: >= -0.05 × (NAV/100k)
        min_vega = float(Decimal('-0.05') * nav_factor)
        
        if portfolio_vega < min_vega:
            return False
        
        return True
    
    def _rank_candidates(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """Rank candidates by composite score."""
        # Sort by model score (descending)
        ranked = sorted(
            candidates,
            key=lambda c: c.model_score or 0,
            reverse=True
        )
        
        # Assign ranks
        for i, candidate in enumerate(ranked, 1):
            candidate.rank = i
        
        return ranked
    
    def _select_top_trades(
        self, 
        candidates: List[TradeCandidate], 
        max_trades: int
    ) -> List[TradeCandidate]:
        """Select top trades ensuring sector diversification."""
        selected = []
        sector_counts = {}
        
        for candidate in candidates:
            if len(selected) >= max_trades:
                break
            
            # Check sector diversification
            sector = candidate.sector or "Unknown"
            if sector_counts.get(sector, 0) >= 2:
                continue
            
            selected.append(candidate)
            candidate.selected = True
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        return selected
    
    def _calculate_portfolio_impact(
        self, 
        selected_trades: List[TradeCandidate], 
        portfolio: Optional[Portfolio], 
        nav: Decimal
    ) -> Dict:
        """Calculate the impact of selected trades on portfolio."""
        total_capital = sum(
            trade.capital_required or Decimal('0') 
            for trade in selected_trades
        )
        
        total_delta = sum(
            trade.strategy.net_delta or 0 
            for trade in selected_trades
        )
        
        total_vega = sum(
            trade.strategy.net_vega or 0 
            for trade in selected_trades
        )
        
        return {
            "total_capital_required": total_capital,
            "capital_utilization": float(total_capital / nav) if nav > 0 else 0,
            "total_delta": total_delta,
            "total_vega": total_vega,
            "sector_distribution": self._calculate_sector_distribution(selected_trades)
        }
    
    def _calculate_sector_distribution(self, trades: List[TradeCandidate]) -> Dict[str, int]:
        """Calculate sector distribution of selected trades."""
        distribution = {}
        for trade in trades:
            sector = trade.sector or "Unknown"
            distribution[sector] = distribution.get(sector, 0) + 1
        return distribution
    
    async def _generate_thesis(self, trade: TradeCandidate) -> str:
        """Generate trading thesis for a trade."""
        # Placeholder implementation - would use more sophisticated logic
        strategy_name = trade.strategy_type.value.replace("_", " ").title()
        
        thesis_parts = [
            f"{strategy_name} on {trade.underlying}",
            f"POP: {trade.probability_of_profit:.0%}" if trade.probability_of_profit else "",
            f"DTE: {trade.days_to_expiration}" if trade.days_to_expiration else "",
            "High IV rank" if trade.iv_rank and trade.iv_rank > 0.5 else ""
        ]
        
        # Filter out empty parts and join
        thesis = " | ".join(part for part in thesis_parts if part)
        
        # Ensure thesis is under 30 words
        words = thesis.split()
        if len(words) > 30:
            thesis = " ".join(words[:30])
        
        return thesis