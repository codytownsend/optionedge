"""
Tradier API client for options and market data.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ...data.models.options import OptionQuote, OptionsChain, OptionType, Greeks
from ...data.models.market_data import StockQuote


class TradierClient(BaseAPIClient):
    """
    Tradier API client implementation.
    
    API Documentation: https://documentation.tradier.com/brokerage-api
    Base URL: https://api.tradier.com/v1/
    Authentication: Bearer token in header
    Rate Limit: 120 requests/minute
    """
    
    def __init__(self, api_key: str, sandbox: bool = False):
        base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
        
        rate_limit_config = RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=None,
            burst_allowance=10
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            half_open_max_calls=3
        )
        
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=30,
            max_retries=3
        )
        
        self.sandbox = sandbox
    
    def authenticate(self) -> Dict[str, str]:
        """Return authentication headers for Tradier API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
    
    def get_provider_name(self) -> str:
        """Return the name of the data provider."""
        return f"Tradier{'Sandbox' if self.sandbox else ''}"
    
    def health_check(self) -> bool:
        """Perform health check using user profile endpoint."""
        try:
            response = self.get("user/profile")
            return "profile" in response
        except Exception:
            return False
    
    def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get current stock quote for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            StockQuote object or None if not found
        """
        try:
            response = self.get("markets/quotes", params={"symbols": symbol})
            
            if "quotes" not in response or "quote" not in response["quotes"]:
                self.logger.warning(f"No quote data found for {symbol}")
                return None
            
            quote_data = response["quotes"]["quote"]
            if isinstance(quote_data, list):
                quote_data = quote_data[0] if quote_data else {}
            
            return self._parse_stock_quote(quote_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get stock quote for {symbol}: {str(e)}")
            return None
    
    def get_stock_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """
        Get current stock quotes for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to StockQuote objects
        """
        try:
            symbols_str = ",".join(symbols)
            response = self.get("markets/quotes", params={"symbols": symbols_str})
            
            quotes = {}
            if "quotes" in response and "quote" in response["quotes"]:
                quote_data = response["quotes"]["quote"]
                
                # Handle single vs multiple quotes
                if isinstance(quote_data, dict):
                    quote_data = [quote_data]
                
                for quote in quote_data:
                    parsed_quote = self._parse_stock_quote(quote)
                    if parsed_quote:
                        quotes[parsed_quote.symbol] = parsed_quote
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Failed to get stock quotes: {str(e)}")
            return {}
    
    def get_options_chain(self, symbol: str, expiration: Optional[date] = None) -> Optional[OptionsChain]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration: Specific expiration date (optional)
            
        Returns:
            OptionsChain object or None if not found
        """
        try:
            params = {"symbol": symbol}
            if expiration:
                params["expiration"] = expiration.strftime("%Y-%m-%d")
            
            response = self.get("markets/options/chains", params=params)
            
            if "options" not in response or "option" not in response["options"]:
                self.logger.warning(f"No options data found for {symbol}")
                return None
            
            # Get underlying price
            underlying_quote = self.get_stock_quote(symbol)
            underlying_price = underlying_quote.price if underlying_quote else None
            
            # Create options chain
            chain = OptionsChain(
                underlying=symbol,
                underlying_price=underlying_price,
                data_source="Tradier"
            )
            
            # Parse options data
            options_data = response["options"]["option"]
            if isinstance(options_data, dict):
                options_data = [options_data]
            
            for option_data in options_data:
                option_quote = self._parse_option_quote(option_data, symbol)
                if option_quote:
                    chain.add_option(option_quote)
            
            return chain
            
        except Exception as e:
            self.logger.error(f"Failed to get options chain for {symbol}: {str(e)}")
            return None
    
    def get_options_expirations(self, symbol: str) -> List[date]:
        """
        Get available options expiration dates for a symbol.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            List of expiration dates
        """
        try:
            response = self.get("markets/options/expirations", params={"symbol": symbol})
            
            if "expirations" not in response or "date" not in response["expirations"]:
                return []
            
            expiration_data = response["expirations"]["date"]
            if isinstance(expiration_data, str):
                expiration_data = [expiration_data]
            
            expirations = []
            for exp_str in expiration_data:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    expirations.append(exp_date)
                except ValueError:
                    self.logger.warning(f"Invalid expiration date format: {exp_str}")
            
            return sorted(expirations)
            
        except Exception as e:
            self.logger.error(f"Failed to get options expirations for {symbol}: {str(e)}")
            return []
    
    def get_option_strikes(self, symbol: str, expiration: date) -> List[Decimal]:
        """
        Get available strike prices for a symbol and expiration.
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration date
            
        Returns:
            List of strike prices
        """
        try:
            params = {
                "symbol": symbol,
                "expiration": expiration.strftime("%Y-%m-%d")
            }
            response = self.get("markets/options/strikes", params=params)
            
            if "strikes" not in response or "strike" not in response["strikes"]:
                return []
            
            strike_data = response["strikes"]["strike"]
            if isinstance(strike_data, (int, float, str)):
                strike_data = [strike_data]
            
            strikes = []
            for strike in strike_data:
                try:
                    strikes.append(Decimal(str(strike)))
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid strike price: {strike}")
            
            return sorted(strikes)
            
        except Exception as e:
            self.logger.error(f"Failed to get option strikes for {symbol}: {str(e)}")
            return []
    
    def _parse_stock_quote(self, quote_data: Dict[str, Any]) -> Optional[StockQuote]:
        """Parse stock quote data from Tradier API response."""
        try:
            if not quote_data or "symbol" not in quote_data:
                return None
            
            # Handle missing or invalid price data
            last_price = quote_data.get("last")
            if not last_price or last_price <= 0:
                return None
            
            return StockQuote(
                symbol=quote_data["symbol"],
                price=Decimal(str(last_price)),
                bid=Decimal(str(quote_data["bid"])) if quote_data.get("bid") else None,
                ask=Decimal(str(quote_data["ask"])) if quote_data.get("ask") else None,
                open=Decimal(str(quote_data["open"])) if quote_data.get("open") else None,
                high=Decimal(str(quote_data["high"])) if quote_data.get("high") else None,
                low=Decimal(str(quote_data["low"])) if quote_data.get("low") else None,
                previous_close=Decimal(str(quote_data["prevclose"])) if quote_data.get("prevclose") else None,
                volume=int(quote_data["volume"]) if quote_data.get("volume") else None,
                avg_volume=int(quote_data["average_volume"]) if quote_data.get("average_volume") else None,
                change=Decimal(str(quote_data["change"])) if quote_data.get("change") else None,
                change_percent=float(quote_data["change_percentage"]) if quote_data.get("change_percentage") else None,
                quote_time=datetime.utcnow(),
                exchange=quote_data.get("exchange")
            )
            
        except (ValueError, TypeError, KeyError) as e:
            self.logger.error(f"Failed to parse stock quote: {str(e)}")
            return None
    
    def _parse_option_quote(self, option_data: Dict[str, Any], underlying: str) -> Optional[OptionQuote]:
        """Parse option quote data from Tradier API response."""
        try:
            if not option_data or "symbol" not in option_data:
                return None
            
            # Parse option symbol components
            symbol = option_data["symbol"]
            strike = Decimal(str(option_data.get("strike", 0)))
            option_type = OptionType.CALL if option_data.get("option_type") == "call" else OptionType.PUT
            
            # Parse expiration date
            exp_str = option_data.get("expiration_date")
            if not exp_str:
                return None
            
            try:
                expiration = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                return None
            
            # Parse Greeks
            greeks = None
            if any(key in option_data for key in ["delta", "gamma", "theta", "vega", "rho"]):
                greeks = Greeks(
                    delta=float(option_data["delta"]) if option_data.get("delta") is not None else None,
                    gamma=float(option_data["gamma"]) if option_data.get("gamma") is not None else None,
                    theta=float(option_data["theta"]) if option_data.get("theta") is not None else None,
                    vega=float(option_data["vega"]) if option_data.get("vega") is not None else None,
                    rho=float(option_data["rho"]) if option_data.get("rho") is not None else None
                )
            
            return OptionQuote(
                symbol=symbol,
                underlying=underlying,
                strike=strike,
                expiration=expiration,
                option_type=option_type,
                bid=Decimal(str(option_data["bid"])) if option_data.get("bid") else None,
                ask=Decimal(str(option_data["ask"])) if option_data.get("ask") else None,
                last=Decimal(str(option_data["last"])) if option_data.get("last") else None,
                mark=Decimal(str(option_data["mark"])) if option_data.get("mark") else None,
                volume=int(option_data["volume"]) if option_data.get("volume") else None,
                open_interest=int(option_data["open_interest"]) if option_data.get("open_interest") else None,
                implied_volatility=float(option_data["implied_volatility"]) if option_data.get("implied_volatility") else None,
                greeks=greeks,
                quote_time=datetime.utcnow(),
                exchange=option_data.get("exchange")
            )
            
        except (ValueError, TypeError, KeyError) as e:
            self.logger.error(f"Failed to parse option quote: {str(e)}")
            return None