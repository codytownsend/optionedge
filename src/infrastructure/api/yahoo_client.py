"""
Yahoo Finance API client for fundamental and market data.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ...data.models.market_data import StockQuote, FundamentalData, TechnicalIndicators, OHLCVData


class YahooFinanceClient(BaseAPIClient):
    """
    Yahoo Finance API client implementation.
    
    Base URL: https://query1.finance.yahoo.com/
    No authentication required for basic data
    Rate Limit: ~2000 requests/hour
    """
    
    def __init__(self):
        rate_limit_config = RateLimitConfig(
            requests_per_minute=35,  # Conservative to stay under hourly limit
            requests_per_hour=2000,
            burst_allowance=10
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            half_open_max_calls=3
        )
        
        super().__init__(
            base_url="https://query1.finance.yahoo.com",
            api_key=None,  # No API key required
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=30,
            max_retries=3
        )
    
    def authenticate(self) -> Dict[str, str]:
        """Return headers for Yahoo Finance API (no authentication required)."""
        return {
            "User-Agent": "Mozilla/5.0 (compatible; OptionsEngine/1.0)",
            "Accept": "application/json"
        }
    
    def get_provider_name(self) -> str:
        """Return the name of the data provider."""
        return "Yahoo Finance"
    
    def health_check(self) -> bool:
        """Perform health check using a simple quote request."""
        try:
            response = self.get("v7/finance/quote", params={"symbols": "AAPL"})
            return "quoteResponse" in response
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
            response = self.get("v7/finance/quote", params={"symbols": symbol})
            
            if ("quoteResponse" not in response or 
                "result" not in response["quoteResponse"] or
                not response["quoteResponse"]["result"]):
                self.logger.warning(f"No quote data found for {symbol}")
                return None
            
            quote_data = response["quoteResponse"]["result"][0]
            return self._parse_stock_quote(quote_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get stock quote for {symbol}: {str(e)}")
            return None
    
    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """
        Get fundamental financial data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            FundamentalData object or None if not found
        """
        try:
            # Get key statistics and financial data
            modules = "defaultKeyStatistics,financialData,summaryProfile,incomeStatementHistory,balanceSheetHistory,cashflowStatementHistory"
            response = self.get(
                f"v10/finance/quoteSummary/{symbol}",
                params={"modules": modules}
            )
            
            if ("quoteSummary" not in response or 
                "result" not in response["quoteSummary"] or
                not response["quoteSummary"]["result"]):
                self.logger.warning(f"No fundamental data found for {symbol}")
                return None
            
            data = response["quoteSummary"]["result"][0]
            return self._parse_fundamental_data(symbol, data)
            
        except Exception as e:
            self.logger.error(f"Failed to get fundamental data for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> List[OHLCVData]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            List of OHLCVData objects
        """
        try:
            params = {
                "symbol": symbol,
                "period1": "0",  # Will be calculated based on period
                "period2": str(int(datetime.now().timestamp())),
                "interval": interval
            }
            
            # Calculate period1 based on period parameter
            now = datetime.now()
            if period == "1d":
                period1 = int((now.replace(hour=0, minute=0, second=0, microsecond=0)).timestamp())
            elif period == "5d":
                period1 = int((now - datetime.timedelta(days=5)).timestamp())
            elif period == "1mo":
                period1 = int((now - datetime.timedelta(days=30)).timestamp())
            elif period == "3mo":
                period1 = int((now - datetime.timedelta(days=90)).timestamp())
            elif period == "6mo":
                period1 = int((now - datetime.timedelta(days=180)).timestamp())
            elif period == "1y":
                period1 = int((now - datetime.timedelta(days=365)).timestamp())
            elif period == "2y":
                period1 = int((now - datetime.timedelta(days=730)).timestamp())
            elif period == "5y":
                period1 = int((now - datetime.timedelta(days=1825)).timestamp())
            else:
                period1 = int((now - datetime.timedelta(days=365)).timestamp())  # Default 1y
            
            params["period1"] = str(period1)
            
            response = self.get(f"v8/finance/chart/{symbol}", params=params)
            
            if ("chart" not in response or 
                "result" not in response["chart"] or
                not response["chart"]["result"]):
                self.logger.warning(f"No historical data found for {symbol}")
                return []
            
            chart_data = response["chart"]["result"][0]
            return self._parse_historical_data(symbol, chart_data, interval)
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            return []
    
    def get_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """
        Get technical indicators for a symbol.
        Note: Yahoo Finance doesn't provide pre-calculated technical indicators,
        so this would require calculating them from historical data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            TechnicalIndicators object or None
        """
        try:
            # Get historical data for calculations
            historical_data = self.get_historical_data(symbol, period="1y", interval="1d")
            
            if len(historical_data) < 200:  # Need sufficient data for indicators
                self.logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # This is a placeholder - technical indicators would be calculated
            # from the historical data using libraries like TA-Lib or custom implementations
            return TechnicalIndicators(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                # Indicators would be calculated here
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get technical indicators for {symbol}: {str(e)}")
            return None
    
    def _parse_stock_quote(self, quote_data: Dict[str, Any]) -> Optional[StockQuote]:
        """Parse stock quote data from Yahoo Finance API response."""
        try:
            if not quote_data or "symbol" not in quote_data:
                return None
            
            # Get price data with fallbacks
            price = quote_data.get("regularMarketPrice", quote_data.get("price"))
            if not price:
                return None
            
            return StockQuote(
                symbol=quote_data["symbol"],
                price=Decimal(str(price)),
                bid=Decimal(str(quote_data["bid"])) if quote_data.get("bid") else None,
                ask=Decimal(str(quote_data["ask"])) if quote_data.get("ask") else None,
                open=Decimal(str(quote_data["regularMarketOpen"])) if quote_data.get("regularMarketOpen") else None,
                high=Decimal(str(quote_data["regularMarketDayHigh"])) if quote_data.get("regularMarketDayHigh") else None,
                low=Decimal(str(quote_data["regularMarketDayLow"])) if quote_data.get("regularMarketDayLow") else None,
                previous_close=Decimal(str(quote_data["regularMarketPreviousClose"])) if quote_data.get("regularMarketPreviousClose") else None,
                volume=int(quote_data["regularMarketVolume"]) if quote_data.get("regularMarketVolume") else None,
                avg_volume=int(quote_data["averageDailyVolume3Month"]) if quote_data.get("averageDailyVolume3Month") else None,
                change=Decimal(str(quote_data["regularMarketChange"])) if quote_data.get("regularMarketChange") else None,
                change_percent=float(quote_data["regularMarketChangePercent"]) if quote_data.get("regularMarketChangePercent") else None,
                market_cap=quote_data.get("marketCap"),
                shares_outstanding=quote_data.get("sharesOutstanding"),
                quote_time=datetime.utcnow(),
                exchange=quote_data.get("fullExchangeName")
            )
            
        except (ValueError, TypeError, KeyError) as e:
            self.logger.error(f"Failed to parse stock quote: {str(e)}")
            return None
    
    def _parse_fundamental_data(self, symbol: str, data: Dict[str, Any]) -> Optional[FundamentalData]:
        """Parse fundamental data from Yahoo Finance API response."""
        try:
            # Extract data from various modules
            key_stats = data.get("defaultKeyStatistics", {})
            financial_data = data.get("financialData", {})
            profile = data.get("summaryProfile", {})
            
            # Get most recent financial statements
            income_stmt = None
            balance_sheet = None
            cash_flow = None
            
            if "incomeStatementHistory" in data and data["incomeStatementHistory"].get("incomeStatementHistory"):
                income_stmt = data["incomeStatementHistory"]["incomeStatementHistory"][0]
            
            if "balanceSheetHistory" in data and data["balanceSheetHistory"].get("balanceSheetHistory"):
                balance_sheet = data["balanceSheetHistory"]["balanceSheetHistory"][0]
            
            if "cashflowStatementHistory" in data and data["cashflowStatementHistory"].get("cashflowStatementHistory"):
                cash_flow = data["cashflowStatementHistory"]["cashflowStatementHistory"][0]
            
            # Helper function to extract value
            def get_value(source_dict, key, default=None):
                if source_dict and key in source_dict:
                    value = source_dict[key]
                    if isinstance(value, dict) and "raw" in value:
                        return value["raw"]
                    return value
                return default
            
            return FundamentalData(
                symbol=symbol,
                report_date=date.today(),  # Approximate - would need actual report date
                period_type="ttm",  # Most Yahoo data is TTM
                
                # Income statement
                revenue=get_value(income_stmt, "totalRevenue") if income_stmt else None,
                gross_profit=get_value(income_stmt, "grossProfit") if income_stmt else None,
                operating_income=get_value(income_stmt, "operatingIncome") if income_stmt else None,
                net_income=get_value(income_stmt, "netIncome") if income_stmt else None,
                ebitda=get_value(income_stmt, "ebitda") if income_stmt else None,
                
                # Per-share metrics
                eps=get_value(key_stats, "trailingEps"),
                eps_diluted=get_value(financial_data, "trailingEps"),
                
                # Balance sheet
                total_assets=get_value(balance_sheet, "totalAssets") if balance_sheet else None,
                total_debt=get_value(balance_sheet, "totalDebt") if balance_sheet else None,
                shareholders_equity=get_value(balance_sheet, "totalStockholderEquity") if balance_sheet else None,
                cash_and_equivalents=get_value(balance_sheet, "cash") if balance_sheet else None,
                
                # Cash flow
                operating_cash_flow=get_value(cash_flow, "totalCashFromOperatingActivities") if cash_flow else None,
                free_cash_flow=get_value(financial_data, "freeCashflow"),
                
                # Valuation ratios
                pe_ratio=get_value(key_stats, "trailingPE"),
                peg_ratio=get_value(key_stats, "pegRatio"),
                price_to_book=get_value(key_stats, "priceToBook"),
                price_to_sales=get_value(key_stats, "priceToSalesTrailing12Months"),
                
                # Profitability ratios
                gross_margin=get_value(financial_data, "grossMargins"),
                operating_margin=get_value(financial_data, "operatingMargins"),
                net_margin=get_value(financial_data, "profitMargins"),
                
                # Efficiency ratios
                return_on_assets=get_value(financial_data, "returnOnAssets"),
                return_on_equity=get_value(financial_data, "returnOnEquity"),
                
                # Sector classification
                sector=get_value(profile, "sector"),
                industry=get_value(profile, "industry")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse fundamental data: {str(e)}")
            return None
    
    def _parse_historical_data(self, symbol: str, chart_data: Dict[str, Any], interval: str) -> List[OHLCVData]:
        """Parse historical price data from Yahoo Finance API response."""
        try:
            if "timestamp" not in chart_data or "indicators" not in chart_data:
                return []
            
            timestamps = chart_data["timestamp"]
            quotes = chart_data["indicators"]["quote"][0]
            
            historical_data = []
            
            for i, timestamp in enumerate(timestamps):
                if i >= len(quotes.get("open", [])):
                    break
                
                # Skip incomplete data
                if (quotes["open"][i] is None or 
                    quotes["high"][i] is None or 
                    quotes["low"][i] is None or 
                    quotes["close"][i] is None or
                    quotes["volume"][i] is None):
                    continue
                
                ohlcv = OHLCVData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamp),
                    timeframe=interval,
                    open=Decimal(str(quotes["open"][i])),
                    high=Decimal(str(quotes["high"][i])),
                    low=Decimal(str(quotes["low"][i])),
                    close=Decimal(str(quotes["close"][i])),
                    volume=int(quotes["volume"][i]),
                    adj_close=Decimal(str(chart_data["indicators"]["adjclose"][0]["adjclose"][i])) 
                             if "adjclose" in chart_data["indicators"] and 
                                chart_data["indicators"]["adjclose"][0]["adjclose"][i] is not None 
                             else None
                )
                
                historical_data.append(ohlcv)
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Failed to parse historical data: {str(e)}")
            return []