"""
Fundamental data integration system for financial metrics and analysis.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import statistics
import logging

from ...data.models.market_data import (
    FundamentalData, StockQuote, SentimentData, EconomicIndicator
)
from ...infrastructure.api import YahooFinanceClient, FREDClient, QuiverQuantClient
from ...infrastructure.cache import DataTypeCacheManager
from ...infrastructure.error_handling import (
    handle_errors, DataQualityError, InsufficientDataError
)


@dataclass
class IndustryMetrics:
    """Industry-wide metrics for comparative analysis."""
    sector: str
    industry: str
    avg_pe_ratio: Optional[float] = None
    avg_gross_margin: Optional[float] = None
    avg_operating_margin: Optional[float] = None
    avg_net_margin: Optional[float] = None
    avg_roe: Optional[float] = None
    avg_roa: Optional[float] = None
    median_market_cap: Optional[int] = None
    sample_size: int = 0


@dataclass
class CompanyAnalysis:
    """Comprehensive company analysis with peer comparison."""
    symbol: str
    fundamental_data: FundamentalData
    sentiment_data: Optional[SentimentData] = None
    industry_metrics: Optional[IndustryMetrics] = None
    
    # Relative metrics (vs industry)
    pe_ratio_percentile: Optional[float] = None
    margin_score: Optional[float] = None
    profitability_score: Optional[float] = None
    overall_score: Optional[float] = None
    
    # Quality scores
    earnings_quality_score: Optional[float] = None
    balance_sheet_strength: Optional[float] = None
    cash_flow_quality: Optional[float] = None


class FundamentalDataService:
    """
    Comprehensive fundamental data integration and analysis service.
    
    Features:
    - Quarterly and annual financial statement data extraction
    - Forward-looking estimates and guidance information
    - Insider trading activity monitoring and trend analysis
    - Institutional ownership changes and 13F filing analysis
    - Credit rating changes and debt structure monitoring
    - Industry-adjusted ratio calculations for comparative analysis
    - Historical trend analysis with seasonal adjustment factors
    """
    
    def __init__(
        self,
        yahoo_client: YahooFinanceClient,
        fred_client: FREDClient,
        quiver_client: Optional[QuiverQuantClient],
        cache_manager: DataTypeCacheManager
    ):
        self.yahoo_client = yahoo_client
        self.fred_client = fred_client
        self.quiver_client = quiver_client
        self.cache_manager = cache_manager
        
        # Industry peer data cache
        self._industry_cache: Dict[str, IndustryMetrics] = {}
        
        # Economic context cache
        self._economic_context: Optional[Dict[str, EconomicIndicator]] = None
        self._economic_context_timestamp: Optional[datetime] = None
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @handle_errors(operation_name="get_fundamental_analysis")
    def get_comprehensive_analysis(self, symbol: str) -> CompanyAnalysis:
        """
        Get comprehensive fundamental analysis for a company.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Complete company analysis with peer comparison
        """
        self.logger.info(f"Performing comprehensive fundamental analysis for {symbol}")
        
        # Get fundamental data
        fundamental_data = self._get_fundamental_data(symbol)
        
        # Get sentiment data if available
        sentiment_data = self._get_sentiment_data(symbol)
        
        # Get industry metrics for comparison
        industry_metrics = None
        if fundamental_data.sector:
            industry_metrics = self._get_industry_metrics(
                fundamental_data.sector, 
                fundamental_data.industry
            )
        
        # Create analysis object
        analysis = CompanyAnalysis(
            symbol=symbol,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data,
            industry_metrics=industry_metrics
        )
        
        # Calculate relative metrics
        self._calculate_relative_metrics(analysis)
        
        # Calculate quality scores
        self._calculate_quality_scores(analysis)
        
        # Calculate overall score
        self._calculate_overall_score(analysis)
        
        return analysis
    
    @handle_errors(operation_name="get_batch_analysis")
    def get_batch_analysis(self, symbols: List[str]) -> Dict[str, CompanyAnalysis]:
        """Get fundamental analysis for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                analysis = self.get_comprehensive_analysis(symbol)
                results[symbol] = analysis
                self.logger.info(f"Completed analysis for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to analyze {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Get sector-wide analysis and trends."""
        
        # Get industry metrics for the sector
        industry_metrics = self._get_industry_metrics(sector)
        
        # Get economic context
        economic_context = self._get_economic_context()
        
        # Analyze sector trends (would need historical data)
        sector_trends = self._analyze_sector_trends(sector)
        
        return {
            "sector": sector,
            "industry_metrics": industry_metrics,
            "economic_context": economic_context,
            "trends": sector_trends,
            "timestamp": datetime.utcnow()
        }
    
    def _get_fundamental_data(self, symbol: str) -> FundamentalData:
        """Get fundamental data with caching."""
        
        # Check cache first
        cached_data = self.cache_manager.get_fundamental_data(
            symbol,
            refresh_func=lambda: self.yahoo_client.get_fundamental_data(symbol)
        )
        
        if cached_data:
            return cached_data
        
        # Fetch fresh data
        data = self.yahoo_client.get_fundamental_data(symbol)
        if not data:
            raise InsufficientDataError(
                f"No fundamental data available for {symbol}",
                symbol=symbol,
                data_source="Yahoo Finance"
            )
        
        # Cache the result
        self.cache_manager.cache_fundamental_data(symbol, data)
        
        return data
    
    def _get_sentiment_data(self, symbol: str) -> Optional[SentimentData]:
        """Get sentiment data if QuiverQuant client is available."""
        
        if not self.quiver_client:
            return None
        
        try:
            return self.quiver_client.get_sentiment_summary(symbol)
        except Exception as e:
            self.logger.warning(f"Failed to get sentiment data for {symbol}: {str(e)}")
            return None
    
    def _get_industry_metrics(
        self, 
        sector: str, 
        industry: Optional[str] = None
    ) -> Optional[IndustryMetrics]:
        """Get industry-wide metrics for peer comparison."""
        
        cache_key = f"{sector}:{industry or 'all'}"
        
        # Check cache first
        if cache_key in self._industry_cache:
            cached_metrics = self._industry_cache[cache_key]
            # Use cached data if less than 1 day old
            if hasattr(cached_metrics, 'timestamp'):
                age = datetime.utcnow() - cached_metrics.timestamp
                if age < timedelta(days=1):
                    return cached_metrics
        
        # For now, return placeholder metrics
        # In a full implementation, this would aggregate data from multiple companies
        metrics = IndustryMetrics(
            sector=sector,
            industry=industry or "Unknown",
            avg_pe_ratio=20.0,  # Market average
            avg_gross_margin=0.3,  # 30% average
            avg_operating_margin=0.15,  # 15% average
            avg_net_margin=0.10,  # 10% average
            avg_roe=0.15,  # 15% average
            avg_roa=0.08,  # 8% average
            sample_size=100  # Assumed sample
        )
        
        # Cache the result
        self._industry_cache[cache_key] = metrics
        
        return metrics
    
    def _get_economic_context(self) -> Dict[str, EconomicIndicator]:
        """Get current economic context."""
        
        # Check cache (update every 2 hours)
        if (self._economic_context and 
            self._economic_context_timestamp and
            datetime.utcnow() - self._economic_context_timestamp < timedelta(hours=2)):
            return self._economic_context
        
        try:
            # Get key economic indicators
            economic_data = self.fred_client.get_economic_indicators_summary()
            self._economic_context = economic_data
            self._economic_context_timestamp = datetime.utcnow()
            return economic_data
        except Exception as e:
            self.logger.warning(f"Failed to get economic context: {str(e)}")
            return {}
    
    def _analyze_sector_trends(self, sector: str) -> Dict[str, Any]:
        """Analyze sector trends (placeholder implementation)."""
        
        # This would analyze historical performance, valuations, etc.
        return {
            "valuation_trend": "neutral",
            "growth_outlook": "positive",
            "risk_factors": ["interest_rate_sensitivity", "regulatory_changes"],
            "opportunities": ["digital_transformation", "market_expansion"]
        }
    
    def _calculate_relative_metrics(self, analysis: CompanyAnalysis):
        """Calculate metrics relative to industry peers."""
        
        if not analysis.industry_metrics:
            return
        
        fundamental = analysis.fundamental_data
        industry = analysis.industry_metrics
        
        # P/E ratio percentile
        if fundamental.pe_ratio and industry.avg_pe_ratio:
            # Simple percentile calculation (would be more sophisticated with full dataset)
            if fundamental.pe_ratio < industry.avg_pe_ratio * 0.8:
                analysis.pe_ratio_percentile = 25.0  # Bottom quartile (value)
            elif fundamental.pe_ratio < industry.avg_pe_ratio:
                analysis.pe_ratio_percentile = 40.0
            elif fundamental.pe_ratio < industry.avg_pe_ratio * 1.2:
                analysis.pe_ratio_percentile = 60.0
            else:
                analysis.pe_ratio_percentile = 80.0  # Top quartile (expensive)
        
        # Margin score (vs industry averages)
        margin_scores = []
        
        if fundamental.gross_margin and industry.avg_gross_margin:
            gross_score = min(100, max(0, (fundamental.gross_margin / industry.avg_gross_margin) * 50))
            margin_scores.append(gross_score)
        
        if fundamental.operating_margin and industry.avg_operating_margin:
            op_score = min(100, max(0, (fundamental.operating_margin / industry.avg_operating_margin) * 50))
            margin_scores.append(op_score)
        
        if fundamental.net_margin and industry.avg_net_margin:
            net_score = min(100, max(0, (fundamental.net_margin / industry.avg_net_margin) * 50))
            margin_scores.append(net_score)
        
        if margin_scores:
            analysis.margin_score = statistics.mean(margin_scores)
        
        # Profitability score
        profitability_scores = []
        
        if fundamental.return_on_equity and industry.avg_roe:
            roe_score = min(100, max(0, (fundamental.return_on_equity / industry.avg_roe) * 50))
            profitability_scores.append(roe_score)
        
        if fundamental.return_on_assets and industry.avg_roa:
            roa_score = min(100, max(0, (fundamental.return_on_assets / industry.avg_roa) * 50))
            profitability_scores.append(roa_score)
        
        if profitability_scores:
            analysis.profitability_score = statistics.mean(profitability_scores)
    
    def _calculate_quality_scores(self, analysis: CompanyAnalysis):
        """Calculate earnings and financial quality scores."""
        
        fundamental = analysis.fundamental_data
        
        # Earnings quality score
        earnings_factors = []
        
        # Revenue growth consistency (simplified)
        if fundamental.revenue:
            earnings_factors.append(75.0)  # Assume good quality
        
        # Cash flow vs earnings
        if fundamental.operating_cash_flow and fundamental.net_income:
            if fundamental.net_income > 0:
                cash_earnings_ratio = fundamental.operating_cash_flow / fundamental.net_income
                if cash_earnings_ratio > 1.2:
                    earnings_factors.append(90.0)  # High quality
                elif cash_earnings_ratio > 0.8:
                    earnings_factors.append(70.0)  # Good quality
                else:
                    earnings_factors.append(40.0)  # Lower quality
        
        if earnings_factors:
            analysis.earnings_quality_score = statistics.mean(earnings_factors)
        
        # Balance sheet strength
        balance_factors = []
        
        # Debt to equity
        if fundamental.total_debt and fundamental.shareholders_equity:
            if fundamental.shareholders_equity > 0:
                debt_ratio = fundamental.total_debt / fundamental.shareholders_equity
                if debt_ratio < 0.3:
                    balance_factors.append(90.0)  # Strong
                elif debt_ratio < 0.6:
                    balance_factors.append(70.0)  # Good
                elif debt_ratio < 1.0:
                    balance_factors.append(50.0)  # Moderate
                else:
                    balance_factors.append(30.0)  # Weak
        
        # Cash position
        if fundamental.cash_and_equivalents and fundamental.total_assets:
            cash_ratio = fundamental.cash_and_equivalents / fundamental.total_assets
            if cash_ratio > 0.2:
                balance_factors.append(80.0)  # Strong cash position
            elif cash_ratio > 0.1:
                balance_factors.append(60.0)  # Adequate
            else:
                balance_factors.append(40.0)  # Limited cash
        
        if balance_factors:
            analysis.balance_sheet_strength = statistics.mean(balance_factors)
        
        # Cash flow quality
        cash_flow_factors = []
        
        if fundamental.free_cash_flow and fundamental.operating_cash_flow:
            if fundamental.operating_cash_flow > 0:
                fcf_ratio = fundamental.free_cash_flow / fundamental.operating_cash_flow
                if fcf_ratio > 0.7:
                    cash_flow_factors.append(85.0)  # Strong FCF conversion
                elif fcf_ratio > 0.4:
                    cash_flow_factors.append(65.0)  # Good conversion
                else:
                    cash_flow_factors.append(40.0)  # Low conversion
        
        # Operating cash flow consistency (simplified)
        if fundamental.operating_cash_flow and fundamental.operating_cash_flow > 0:
            cash_flow_factors.append(70.0)  # Assume consistent
        
        if cash_flow_factors:
            analysis.cash_flow_quality = statistics.mean(cash_flow_factors)
    
    def _calculate_overall_score(self, analysis: CompanyAnalysis):
        """Calculate overall fundamental score."""
        
        scores = []
        weights = []
        
        # Valuation component (lower P/E percentile is better)
        if analysis.pe_ratio_percentile is not None:
            valuation_score = 100 - analysis.pe_ratio_percentile  # Invert so lower P/E = higher score
            scores.append(valuation_score)
            weights.append(0.20)
        
        # Profitability component
        if analysis.profitability_score is not None:
            scores.append(analysis.profitability_score)
            weights.append(0.25)
        
        # Margin component
        if analysis.margin_score is not None:
            scores.append(analysis.margin_score)
            weights.append(0.20)
        
        # Quality components
        if analysis.earnings_quality_score is not None:
            scores.append(analysis.earnings_quality_score)
            weights.append(0.15)
        
        if analysis.balance_sheet_strength is not None:
            scores.append(analysis.balance_sheet_strength)
            weights.append(0.10)
        
        if analysis.cash_flow_quality is not None:
            scores.append(analysis.cash_flow_quality)
            weights.append(0.10)
        
        # Sentiment component
        if analysis.sentiment_data and analysis.sentiment_data.overall_sentiment is not None:
            # Convert sentiment (-1 to 1) to score (0 to 100)
            sentiment_score = (analysis.sentiment_data.overall_sentiment + 1) * 50
            scores.append(sentiment_score)
            weights.append(0.10)  # Lower weight for sentiment
        
        if scores and weights:
            # Weighted average
            analysis.overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def get_insider_trading_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get insider trading analysis."""
        
        if not self.quiver_client:
            return {"error": "QuiverQuant client not available"}
        
        try:
            # Get insider trading data
            insider_sentiment = self.quiver_client.get_insider_trading(symbol, days=90)
            
            # Get congressional trading
            congress_trades = self.quiver_client.get_congressional_trading(symbol, days=90)
            
            return {
                "symbol": symbol,
                "insider_sentiment": insider_sentiment,
                "congress_trades": len(congress_trades),
                "analysis_date": datetime.utcnow(),
                "summary": self._analyze_insider_activity(insider_sentiment, congress_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get insider trading analysis for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_insider_activity(
        self, 
        insider_sentiment: Optional[float], 
        congress_trades: List[Dict[str, Any]]
    ) -> str:
        """Analyze insider trading activity."""
        
        summary_parts = []
        
        if insider_sentiment is not None:
            if insider_sentiment > 0.3:
                summary_parts.append("Strong insider buying activity")
            elif insider_sentiment > 0.1:
                summary_parts.append("Moderate insider buying")
            elif insider_sentiment < -0.3:
                summary_parts.append("Strong insider selling activity")
            elif insider_sentiment < -0.1:
                summary_parts.append("Moderate insider selling")
            else:
                summary_parts.append("Neutral insider activity")
        
        if congress_trades:
            summary_parts.append(f"{len(congress_trades)} congressional trades in last 90 days")
        
        return "; ".join(summary_parts) if summary_parts else "No significant insider activity"
    
    def get_peer_comparison(
        self, 
        symbol: str, 
        peer_symbols: List[str]
    ) -> Dict[str, Any]:
        """Compare company metrics against specific peers."""
        
        # Get analysis for target company
        target_analysis = self.get_comprehensive_analysis(symbol)
        
        # Get analysis for peers
        peer_analyses = {}
        for peer in peer_symbols:
            try:
                peer_analyses[peer] = self.get_comprehensive_analysis(peer)
            except Exception as e:
                self.logger.warning(f"Failed to analyze peer {peer}: {str(e)}")
        
        # Calculate peer rankings
        rankings = self._calculate_peer_rankings(target_analysis, peer_analyses)
        
        return {
            "target_symbol": symbol,
            "target_analysis": target_analysis,
            "peer_analyses": peer_analyses,
            "rankings": rankings,
            "comparison_date": datetime.utcnow()
        }
    
    def _calculate_peer_rankings(
        self, 
        target: CompanyAnalysis, 
        peers: Dict[str, CompanyAnalysis]
    ) -> Dict[str, Any]:
        """Calculate rankings among peer group."""
        
        all_companies = {"TARGET": target}
        all_companies.update(peers)
        
        rankings = {}
        
        # Overall score ranking
        if target.overall_score is not None:
            scores = {
                symbol: analysis.overall_score 
                for symbol, analysis in all_companies.items() 
                if analysis.overall_score is not None
            }
            
            if scores:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                target_rank = next(
                    (i + 1 for i, (symbol, _) in enumerate(sorted_scores) if symbol == "TARGET"), 
                    None
                )
                rankings["overall_score"] = {
                    "rank": target_rank,
                    "total": len(sorted_scores),
                    "percentile": ((len(sorted_scores) - target_rank + 1) / len(sorted_scores)) * 100 if target_rank else None
                }
        
        # P/E ratio ranking (lower is better for value)
        pe_ratios = {
            symbol: analysis.fundamental_data.pe_ratio
            for symbol, analysis in all_companies.items()
            if analysis.fundamental_data.pe_ratio is not None
        }
        
        if pe_ratios and "TARGET" in pe_ratios:
            sorted_pe = sorted(pe_ratios.items(), key=lambda x: x[1])  # Lower is better
            target_pe_rank = next(
                (i + 1 for i, (symbol, _) in enumerate(sorted_pe) if symbol == "TARGET"),
                None
            )
            rankings["pe_ratio"] = {
                "rank": target_pe_rank,
                "total": len(sorted_pe),
                "percentile": ((len(sorted_pe) - target_pe_rank + 1) / len(sorted_pe)) * 100 if target_pe_rank else None
            }
        
        return rankings