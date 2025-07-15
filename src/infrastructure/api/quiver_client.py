"""
QuiverQuant API client for alternative data and sentiment analysis.
"""

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, APIError, DataValidator
from ...data.models.market_data import SentimentData

logger = logging.getLogger(__name__)


class QuiverQuantClient(BaseAPIClient):
    """QuiverQuant API client for alternative data sources."""
    
    def __init__(self, 
                 api_key: str,
                 **kwargs):
        """
        Initialize QuiverQuant client.
        
        Args:
            api_key: QuiverQuant API key
            **kwargs: Additional arguments for base client
        """
        super().__init__(
            base_url="https://api.quiverquant.com",
            api_key=api_key,
            rate_limit_per_minute=60,  # QuiverQuant rate limit
            **kwargs
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for QuiverQuant API."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def _validate_response(self, response) -> bool:
        """Validate QuiverQuant API response."""
        try:
            data = response.json()
            
            # Check for API errors
            if isinstance(data, dict):
                if 'error' in data:
                    logger.error(f"QuiverQuant API error: {data['error']}")
                    return False
                
                if 'message' in data and 'error' in data.get('message', '').lower():
                    logger.error(f"QuiverQuant API message: {data['message']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False
    
    def get_reddit_sentiment(self, 
                           symbol: str,
                           lookback_days: int = 7) -> Optional[SentimentData]:
        """
        Get Reddit sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            SentimentData object or None if not found
        """
        params = {
            'symbol': symbol.upper(),
            'days': lookback_days
        }
        
        try:
            response_data = self.get('/beta/social/reddit', params=params)
            return self._parse_reddit_sentiment(symbol, response_data)
            
        except Exception as e:
            logger.error(f"Failed to get Reddit sentiment for {symbol}: {e}")
            return None
    
    def get_twitter_sentiment(self, 
                            symbol: str,
                            lookback_days: int = 7) -> Optional[SentimentData]:
        """
        Get Twitter sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            SentimentData object or None if not found
        """
        params = {
            'symbol': symbol.upper(),
            'days': lookback_days
        }
        
        try:
            response_data = self.get('/beta/social/twitter', params=params)
            return self._parse_twitter_sentiment(symbol, response_data)
            
        except Exception as e:
            logger.error(f"Failed to get Twitter sentiment for {symbol}: {e}")
            return None
    
    def get_insider_trading(self, 
                          symbol: str,
                          lookback_days: int = 30) -> List[Dict[str, Any]]:
        """
        Get insider trading data for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            List of insider trading transactions
        """
        params = {
            'symbol': symbol.upper(),
            'days': lookback_days
        }
        
        try:
            response_data = self.get('/beta/historical/insider-trading', params=params)
            return self._parse_insider_trading(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get insider trading for {symbol}: {e}")
            return []
    
    def get_congress_trading(self, 
                           symbol: str,
                           lookback_days: int = 90) -> List[Dict[str, Any]]:
        """
        Get congressional trading data for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            List of congressional trading transactions
        """
        params = {
            'symbol': symbol.upper(),
            'days': lookback_days
        }
        
        try:
            response_data = self.get('/beta/historical/congress-trading', params=params)
            return self._parse_congress_trading(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get congress trading for {symbol}: {e}")
            return []
    
    def get_institutional_holdings(self, 
                                 symbol: str) -> List[Dict[str, Any]]:
        """
        Get institutional holdings data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of institutional holdings
        """
        params = {'symbol': symbol.upper()}
        
        try:
            response_data = self.get('/beta/historical/institutional-holdings', params=params)
            return self._parse_institutional_holdings(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get institutional holdings for {symbol}: {e}")
            return []
    
    def get_analyst_ratings(self, 
                          symbol: str,
                          lookback_days: int = 90) -> List[Dict[str, Any]]:
        """
        Get analyst ratings and price targets for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            List of analyst ratings
        """
        params = {
            'symbol': symbol.upper(),
            'days': lookback_days
        }
        
        try:
            response_data = self.get('/beta/historical/analyst-ratings', params=params)
            return self._parse_analyst_ratings(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get analyst ratings for {symbol}: {e}")
            return []
    
    def get_options_flow(self, 
                       symbol: str,
                       lookback_days: int = 7) -> List[Dict[str, Any]]:
        """
        Get options flow data for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            List of options flow transactions
        """
        params = {
            'symbol': symbol.upper(),
            'days': lookback_days
        }
        
        try:
            response_data = self.get('/beta/historical/options-flow', params=params)
            return self._parse_options_flow(response_data)
            
        except Exception as e:
            logger.error(f"Failed to get options flow for {symbol}: {e}")
            return []
    
    def get_comprehensive_sentiment(self, 
                                  symbol: str,
                                  lookback_days: int = 7) -> Optional[SentimentData]:
        """
        Get comprehensive sentiment data from multiple sources.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            Combined SentimentData object
        """
        try:
            # Get sentiment from multiple sources
            reddit_sentiment = self.get_reddit_sentiment(symbol, lookback_days)
            twitter_sentiment = self.get_twitter_sentiment(symbol, lookback_days)
            
            # Combine sentiment data
            overall_sentiment = None
            mention_count = 0
            
            if reddit_sentiment and twitter_sentiment:
                # Average the sentiments
                reddit_score = reddit_sentiment.reddit_sentiment or 0
                twitter_score = twitter_sentiment.twitter_sentiment or 0
                overall_sentiment = (reddit_score + twitter_score) / 2
                
                mention_count = (reddit_sentiment.mention_count or 0) + (twitter_sentiment.mention_count or 0)
            elif reddit_sentiment:
                overall_sentiment = reddit_sentiment.reddit_sentiment
                mention_count = reddit_sentiment.mention_count or 0
            elif twitter_sentiment:
                overall_sentiment = twitter_sentiment.twitter_sentiment
                mention_count = twitter_sentiment.mention_count or 0
            
            if overall_sentiment is not None:
                return SentimentData(
                    symbol=symbol.upper(),
                    timestamp=datetime.utcnow(),
                    overall_sentiment=overall_sentiment,
                    reddit_sentiment=reddit_sentiment.reddit_sentiment if reddit_sentiment else None,
                    twitter_sentiment=twitter_sentiment.twitter_sentiment if twitter_sentiment else None,
                    mention_count=mention_count
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive sentiment for {symbol}: {e}")
            return None
    
    def _parse_reddit_sentiment(self, symbol: str, data: Dict) -> Optional[SentimentData]:
        """Parse Reddit sentiment response."""
        try:
            if not data or not isinstance(data, list) or len(data) == 0:
                return None
            
            # Use most recent data point
            recent_data = data[0]
            
            sentiment_score = self._normalize_sentiment(recent_data.get('sentiment', 0))
            mention_count = recent_data.get('mentions', 0)
            
            return SentimentData(
                symbol=symbol.upper(),
                timestamp=datetime.utcnow(),
                reddit_sentiment=sentiment_score,
                mention_count=mention_count
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Reddit sentiment: {e}")
            return None
    
    def _parse_twitter_sentiment(self, symbol: str, data: Dict) -> Optional[SentimentData]:
        """Parse Twitter sentiment response."""
        try:
            if not data or not isinstance(data, list) or len(data) == 0:
                return None
            
            # Use most recent data point
            recent_data = data[0]
            
            sentiment_score = self._normalize_sentiment(recent_data.get('sentiment', 0))
            mention_count = recent_data.get('mentions', 0)
            
            return SentimentData(
                symbol=symbol.upper(),
                timestamp=datetime.utcnow(),
                twitter_sentiment=sentiment_score,
                mention_count=mention_count
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Twitter sentiment: {e}")
            return None
    
    def _parse_insider_trading(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse insider trading response."""
        try:
            if not data or not isinstance(data, list):
                return []
            
            parsed_trades = []
            for trade in data:
                parsed_trade = {
                    'date': trade.get('date'),
                    'insider_name': trade.get('insider'),
                    'title': trade.get('title'),
                    'transaction_type': trade.get('transaction_type'),
                    'shares': trade.get('shares'),
                    'price': trade.get('price'),
                    'value': trade.get('value'),
                    'shares_held': trade.get('shares_held')
                }
                parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            logger.warning(f"Failed to parse insider trading: {e}")
            return []
    
    def _parse_congress_trading(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse congressional trading response."""
        try:
            if not data or not isinstance(data, list):
                return []
            
            parsed_trades = []
            for trade in data:
                parsed_trade = {
                    'date': trade.get('date'),
                    'representative': trade.get('representative'),
                    'transaction_type': trade.get('transaction_type'),
                    'amount_range': trade.get('amount_range'),
                    'filed_date': trade.get('filed_date')
                }
                parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            logger.warning(f"Failed to parse congress trading: {e}")
            return []
    
    def _parse_institutional_holdings(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse institutional holdings response."""
        try:
            if not data or not isinstance(data, list):
                return []
            
            parsed_holdings = []
            for holding in data:
                parsed_holding = {
                    'date': holding.get('date'),
                    'institution': holding.get('institution'),
                    'shares': holding.get('shares'),
                    'value': holding.get('value'),
                    'percent_of_portfolio': holding.get('percent_of_portfolio'),
                    'change_in_shares': holding.get('change_in_shares')
                }
                parsed_holdings.append(parsed_holding)
            
            return parsed_holdings
            
        except Exception as e:
            logger.warning(f"Failed to parse institutional holdings: {e}")
            return []
    
    def _parse_analyst_ratings(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse analyst ratings response."""
        try:
            if not data or not isinstance(data, list):
                return []
            
            parsed_ratings = []
            for rating in data:
                parsed_rating = {
                    'date': rating.get('date'),
                    'analyst': rating.get('analyst'),
                    'rating': rating.get('rating'),
                    'previous_rating': rating.get('previous_rating'),
                    'price_target': rating.get('price_target'),
                    'previous_price_target': rating.get('previous_price_target')
                }
                parsed_ratings.append(parsed_rating)
            
            return parsed_ratings
            
        except Exception as e:
            logger.warning(f"Failed to parse analyst ratings: {e}")
            return []
    
    def _parse_options_flow(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse options flow response."""
        try:
            if not data or not isinstance(data, list):
                return []
            
            parsed_flows = []
            for flow in data:
                parsed_flow = {
                    'date': flow.get('date'),
                    'time': flow.get('time'),
                    'strike': flow.get('strike'),
                    'expiration': flow.get('expiration'),
                    'type': flow.get('type'),
                    'sentiment': flow.get('sentiment'),
                    'volume': flow.get('volume'),
                    'premium': flow.get('premium'),
                    'open_interest': flow.get('open_interest')
                }
                parsed_flows.append(parsed_flow)
            
            return parsed_flows
            
        except Exception as e:
            logger.warning(f"Failed to parse options flow: {e}")
            return []
    
    def _normalize_sentiment(self, raw_sentiment: Any) -> float:
        """Normalize sentiment score to -1 to 1 range."""
        try:
            if raw_sentiment is None:
                return 0.0
            
            score = float(raw_sentiment)
            
            # Assume raw sentiment is already in -1 to 1 range
            # If not, adjust normalization logic here
            return max(-1.0, min(1.0, score))
            
        except (ValueError, TypeError):
            return 0.0
    
    def test_connection(self) -> bool:
        """Test QuiverQuant API connection."""
        try:
            # Test with a simple endpoint
            response_data = self.get('/beta/social/reddit', params={'symbol': 'AAPL', 'days': 1})
            return response_data is not None
        except Exception as e:
            logger.error(f"QuiverQuant connection test failed: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols for sentiment analysis."""
        try:
            response_data = self.get('/beta/symbols')
            if isinstance(response_data, list):
                return [symbol.upper() for symbol in response_data]
            return []
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []