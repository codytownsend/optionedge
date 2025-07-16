"""
QuiverQuant API client for alternative data (sentiment, flow, insider trading).
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ...data.models.market_data import SentimentData, ETFFlowData


class QuiverQuantClient(BaseAPIClient):
    """
    QuiverQuant API client implementation.
    
    API Documentation: https://www.quiverquant.com/docs/
    Base URL: https://api.quiverquant.com/beta/
    Authentication: Bearer token in header
    Rate Limit: 100 requests/minute
    """
    
    def __init__(self, api_key: str):
        rate_limit_config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=None,
            burst_allowance=10
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            half_open_max_calls=3
        )
        
        super().__init__(
            base_url="https://api.quiverquant.com/beta",
            api_key=api_key,
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=30,
            max_retries=3
        )
    
    def authenticate(self) -> Dict[str, str]:
        """Return authentication headers for QuiverQuant API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "OptionsEngine/1.0"
        }
    
    def get_provider_name(self) -> str:
        """Return the name of the data provider."""
        return "QuiverQuant"
    
    def health_check(self) -> bool:
        """Perform health check using a simple endpoint."""
        try:
            # Try to get recent congressional trading data (should always have data)
            response = self.get("live/congresstrading")
            return isinstance(response, list)
        except Exception:
            return False
    
    def get_reddit_sentiment(self, symbol: str, days: int = 30) -> Optional[SentimentData]:
        """
        Get Reddit sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of data to retrieve
            
        Returns:
            SentimentData object or None
        """
        try:
            response = self.get(f"live/wallstreetbets/{symbol}")
            
            if not response or not isinstance(response, list):
                self.logger.warning(f"No Reddit sentiment data found for {symbol}")
                return None
            
            # Get recent data within the specified days
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_posts = [
                post for post in response 
                if datetime.fromisoformat(post.get("Date", "").replace("Z", "+00:00")) > cutoff_date
            ]
            
            if not recent_posts:
                return None
            
            # Calculate aggregate sentiment
            total_sentiment = 0
            total_mentions = 0
            
            for post in recent_posts:
                sentiment = post.get("Sentiment", 0)
                if sentiment is not None:
                    total_sentiment += sentiment
                    total_mentions += 1
            
            if total_mentions == 0:
                return None
            
            avg_sentiment = total_sentiment / total_mentions
            
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                reddit_sentiment=max(-1.0, min(1.0, avg_sentiment)),  # Clamp to [-1, 1]
                mention_count=total_mentions,
                overall_sentiment=max(-1.0, min(1.0, avg_sentiment))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get Reddit sentiment for {symbol}: {str(e)}")
            return None
    
    def get_insider_trading(self, symbol: str, days: int = 90) -> Optional[float]:
        """
        Get insider trading sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of data to analyze
            
        Returns:
            Insider sentiment score (-1 to 1) or None
        """
        try:
            response = self.get(f"live/insidertrading/{symbol}")
            
            if not response or not isinstance(response, list):
                self.logger.warning(f"No insider trading data found for {symbol}")
                return None
            
            # Filter recent trades
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = [
                trade for trade in response
                if datetime.strptime(trade.get("Date", ""), "%Y-%m-%d") > cutoff_date.replace(tzinfo=None)
            ]
            
            if not recent_trades:
                return None
            
            # Calculate sentiment based on buy/sell ratio and amounts
            total_buy_value = 0
            total_sell_value = 0
            
            for trade in recent_trades:
                transaction_type = trade.get("Transaction", "").lower()
                shares = trade.get("Shares", 0) or 0
                price = trade.get("Price", 0) or 0
                value = shares * price
                
                if "buy" in transaction_type or "purchase" in transaction_type:
                    total_buy_value += value
                elif "sell" in transaction_type or "sale" in transaction_type:
                    total_sell_value += value
            
            total_value = total_buy_value + total_sell_value
            if total_value == 0:
                return None
            
            # Calculate sentiment: positive for net buying, negative for net selling
            net_sentiment = (total_buy_value - total_sell_value) / total_value
            return max(-1.0, min(1.0, net_sentiment))
            
        except Exception as e:
            self.logger.error(f"Failed to get insider trading data for {symbol}: {str(e)}")
            return None
    
    def get_congressional_trading(self, symbol: Optional[str] = None, days: int = 90) -> List[Dict[str, Any]]:
        """
        Get congressional trading data.
        
        Args:
            symbol: Optional stock symbol to filter by
            days: Number of days of data to retrieve
            
        Returns:
            List of congressional trading records
        """
        try:
            endpoint = f"live/congresstrading/{symbol}" if symbol else "live/congresstrading"
            response = self.get(endpoint)
            
            if not response or not isinstance(response, list):
                return []
            
            # Filter recent trades
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = []
            
            for trade in response:
                try:
                    trade_date = datetime.strptime(trade.get("TransactionDate", ""), "%Y-%m-%d")
                    if trade_date > cutoff_date.replace(tzinfo=None):
                        recent_trades.append(trade)
                except (ValueError, TypeError):
                    continue
            
            return recent_trades
            
        except Exception as e:
            self.logger.error(f"Failed to get congressional trading data: {str(e)}")
            return []
    
    def get_etf_flows(self, etf_symbol: str, days: int = 30) -> Optional[ETFFlowData]:
        """
        Get ETF flow data for a symbol.
        
        Args:
            etf_symbol: ETF symbol
            days: Number of days of data to analyze
            
        Returns:
            ETFFlowData object or None
        """
        try:
            # Note: QuiverQuant may not have direct ETF flow endpoints
            # This is a placeholder implementation that would need to be
            # adapted based on actual available endpoints
            
            response = self.get(f"live/etfflows/{etf_symbol}")
            
            if not response:
                self.logger.warning(f"No ETF flow data found for {etf_symbol}")
                return None
            
            # This would be adapted based on actual QuiverQuant ETF flow data structure
            latest_data = response[0] if isinstance(response, list) else response
            
            return ETFFlowData(
                symbol=etf_symbol,
                date=datetime.strptime(latest_data.get("Date", ""), "%Y-%m-%d").date(),
                net_flow=Decimal(str(latest_data.get("NetFlow", 0))),
                inflow=Decimal(str(latest_data.get("Inflow", 0))) if latest_data.get("Inflow") else None,
                outflow=Decimal(str(latest_data.get("Outflow", 0))) if latest_data.get("Outflow") else None,
                aum=Decimal(str(latest_data.get("AUM", 0))) if latest_data.get("AUM") else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get ETF flow data for {etf_symbol}: {str(e)}")
            return None
    
    def get_options_flow(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get unusual options activity data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of data to retrieve
            
        Returns:
            List of options flow records
        """
        try:
            response = self.get(f"live/optionsflow/{symbol}")
            
            if not response or not isinstance(response, list):
                self.logger.warning(f"No options flow data found for {symbol}")
                return []
            
            # Filter recent activity
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_flow = []
            
            for flow in response:
                try:
                    flow_date = datetime.strptime(flow.get("Date", ""), "%Y-%m-%d")
                    if flow_date > cutoff_date.replace(tzinfo=None):
                        recent_flow.append(flow)
                except (ValueError, TypeError):
                    continue
            
            return recent_flow
            
        except Exception as e:
            self.logger.error(f"Failed to get options flow data for {symbol}: {str(e)}")
            return []
    
    def get_sentiment_summary(self, symbol: str) -> Optional[SentimentData]:
        """
        Get comprehensive sentiment summary for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            SentimentData object with multiple sentiment sources or None
        """
        try:
            # Gather sentiment from multiple sources
            reddit_sentiment_data = self.get_reddit_sentiment(symbol)
            insider_sentiment = self.get_insider_trading(symbol)
            
            # Get congressional sentiment
            congress_trades = self.get_congressional_trading(symbol, days=90)
            congress_sentiment = self._calculate_congressional_sentiment(congress_trades)
            
            if not any([reddit_sentiment_data, insider_sentiment, congress_sentiment]):
                return None
            
            # Combine sentiments
            overall_sentiment = 0
            sentiment_count = 0
            
            if reddit_sentiment_data and reddit_sentiment_data.reddit_sentiment is not None:
                overall_sentiment += reddit_sentiment_data.reddit_sentiment
                sentiment_count += 1
            
            if insider_sentiment is not None:
                overall_sentiment += insider_sentiment
                sentiment_count += 1
            
            if congress_sentiment is not None:
                overall_sentiment += congress_sentiment
                sentiment_count += 1
            
            if sentiment_count == 0:
                return None
            
            avg_sentiment = overall_sentiment / sentiment_count
            
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                overall_sentiment=avg_sentiment,
                reddit_sentiment=reddit_sentiment_data.reddit_sentiment if reddit_sentiment_data else None,
                insider_sentiment=insider_sentiment,
                mention_count=reddit_sentiment_data.mention_count if reddit_sentiment_data else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get sentiment summary for {symbol}: {str(e)}")
            return None
    
    def _calculate_congressional_sentiment(self, trades: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate sentiment from congressional trading data."""
        if not trades:
            return None
        
        total_buy_value = 0
        total_sell_value = 0
        
        for trade in trades:
            transaction_type = trade.get("Transaction", "").lower()
            amount = trade.get("Amount", 0) or 0
            
            if "purchase" in transaction_type or "buy" in transaction_type:
                total_buy_value += amount
            elif "sale" in transaction_type or "sell" in transaction_type:
                total_sell_value += amount
        
        total_value = total_buy_value + total_sell_value
        if total_value == 0:
            return None
        
        # Return sentiment: positive for net buying, negative for net selling
        return (total_buy_value - total_sell_value) / total_value