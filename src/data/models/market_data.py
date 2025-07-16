"""
Market data models for stocks, fundamentals, technicals, and economic indicators.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field, validator


class StockQuote(BaseModel):
    """Real-time stock quote with comprehensive market data."""
    
    symbol: str = Field(..., description="Stock symbol")
    price: Decimal = Field(..., description="Current price")
    
    # Bid/Ask
    bid: Optional[Decimal] = Field(None, description="Bid price")
    ask: Optional[Decimal] = Field(None, description="Ask price")
    
    # OHLC
    open: Optional[Decimal] = Field(None, description="Open price")
    high: Optional[Decimal] = Field(None, description="High price")
    low: Optional[Decimal] = Field(None, description="Low price")
    previous_close: Optional[Decimal] = Field(None, description="Previous close")
    
    # Volume
    volume: Optional[int] = Field(None, description="Volume")
    avg_volume: Optional[int] = Field(None, description="Average volume")
    
    # Change metrics
    change: Optional[Decimal] = Field(None, description="Price change")
    change_percent: Optional[float] = Field(None, description="Percentage change")
    
    # Market cap and fundamental metrics
    market_cap: Optional[int] = Field(None, description="Market capitalization")
    shares_outstanding: Optional[int] = Field(None, description="Shares outstanding")
    
    # Metadata
    quote_time: Optional[datetime] = Field(None, description="Quote timestamp")
    exchange: Optional[str] = Field(None, description="Exchange")
    
    class Config:
        frozen = True
        validate_assignment = True
    
    @validator('price', 'bid', 'ask', 'open', 'high', 'low', 'previous_close')
    def prices_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Stock prices must be positive')
        return v
    
    @validator('volume', 'avg_volume')
    def volume_must_be_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError('Volume cannot be negative')
        return v

    @property
    def bid_ask_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None


class OHLCVData(BaseModel):
    """OHLCV (candlestick) data point for technical analysis."""
    
    symbol: str = Field(..., description="Symbol")
    timestamp: datetime = Field(..., description="Timestamp")
    timeframe: str = Field(..., description="Timeframe (1d, 1h, etc.)")
    
    open: Decimal = Field(..., description="Open price")
    high: Decimal = Field(..., description="High price") 
    low: Decimal = Field(..., description="Low price")
    close: Decimal = Field(..., description="Close price")
    volume: int = Field(..., description="Volume")
    
    # Adjusted data for splits/dividends
    adj_close: Optional[Decimal] = Field(None, description="Adjusted close price")
    
    class Config:
        frozen = True
        validate_assignment = True
    
    @validator('open', 'high', 'low', 'close', 'adj_close')
    def prices_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('OHLC prices must be positive')
        return v
    
    @validator('volume')
    def volume_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Volume cannot be negative')
        return v
    
    def validate_ohlc_logic(self):
        """Validate OHLC price relationships."""
        if not (self.low <= self.open <= self.high and 
                self.low <= self.close <= self.high):
            raise ValueError('Invalid OHLC price relationships')

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3) for VWAP calculations."""
        return (self.high + self.low + self.close) / 3

    @property
    def true_range(self, prev_close: Optional[Decimal] = None) -> Decimal:
        """Calculate true range for ATR calculation."""
        if prev_close is None:
            return self.high - self.low
        
        high_low = self.high - self.low
        high_close_prev = abs(self.high - prev_close)
        low_close_prev = abs(self.low - prev_close)
        
        return max(high_low, high_close_prev, low_close_prev)


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators for a symbol as specified in instructions."""
    
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    
    # Moving averages (required per instructions)
    sma_50: Optional[Decimal] = Field(None, description="50-day simple moving average")
    sma_100: Optional[Decimal] = Field(None, description="100-day simple moving average")
    sma_200: Optional[Decimal] = Field(None, description="200-day simple moving average")
    
    ema_12: Optional[Decimal] = Field(None, description="12-day exponential moving average")
    ema_26: Optional[Decimal] = Field(None, description="26-day exponential moving average")
    
    # Momentum indicators (required per instructions)
    rsi_14: Optional[float] = Field(None, description="14-day RSI")
    macd: Optional[Decimal] = Field(None, description="MACD line")
    macd_signal: Optional[Decimal] = Field(None, description="MACD signal line")
    macd_histogram: Optional[Decimal] = Field(None, description="MACD histogram")
    
    # Volatility indicators (required per instructions)
    bollinger_upper: Optional[Decimal] = Field(None, description="Bollinger upper band")
    bollinger_middle: Optional[Decimal] = Field(None, description="Bollinger middle band")
    bollinger_lower: Optional[Decimal] = Field(None, description="Bollinger lower band")
    atr_14: Optional[Decimal] = Field(None, description="14-day Average True Range")
    
    # Volume indicators (required per instructions)
    vwap: Optional[Decimal] = Field(None, description="Volume Weighted Average Price")
    
    # Historical volatility (required per instructions)
    hv_30: Optional[float] = Field(None, description="30-day historical volatility")
    hv_60: Optional[float] = Field(None, description="60-day historical volatility")
    
    # Momentum metrics (required per instructions)
    momentum_1d: Optional[float] = Field(None, description="1-day momentum")
    momentum_5d: Optional[float] = Field(None, description="5-day momentum")
    momentum_20d: Optional[float] = Field(None, description="20-day momentum")
    
    # Z-scores for ranking (required per instructions)
    momentum_z_score: Optional[float] = Field(None, description="Momentum Z-score")
    
    class Config:
        validate_assignment = True
    
    @validator('rsi_14')
    def rsi_must_be_valid(cls, v):
        if v is not None and not (0 <= v <= 100):
            raise ValueError('RSI must be between 0 and 100')
        return v
    
    @validator('hv_30', 'hv_60')
    def volatility_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Volatility cannot be negative')
        return v

    @property
    def bollinger_percent_b(self) -> Optional[float]:
        """Calculate %B indicator for Bollinger Bands."""
        if (self.bollinger_upper is not None and 
            self.bollinger_lower is not None and 
            hasattr(self, 'current_price')):
            
            width = self.bollinger_upper - self.bollinger_lower
            if width > 0:
                return float((self.current_price - self.bollinger_lower) / width)
        return None

    @property
    def bollinger_width(self) -> Optional[Decimal]:
        """Calculate Bollinger Band width."""
        if (self.bollinger_upper is not None and 
            self.bollinger_lower is not None):
            return self.bollinger_upper - self.bollinger_lower
        return None


class FundamentalData(BaseModel):
    """Fundamental financial data for a company as specified in instructions."""
    
    symbol: str = Field(..., description="Stock symbol")
    report_date: date = Field(..., description="Report date")
    period_type: str = Field(..., description="Period type (quarterly, annual, ttm)")
    
    # Income statement (required per instructions)
    revenue: Optional[int] = Field(None, description="Total revenue")
    gross_profit: Optional[int] = Field(None, description="Gross profit")
    operating_income: Optional[int] = Field(None, description="Operating income")
    net_income: Optional[int] = Field(None, description="Net income")
    ebitda: Optional[int] = Field(None, description="EBITDA")
    
    # Per-share metrics (required per instructions)
    eps: Optional[Decimal] = Field(None, description="Earnings per share")
    eps_diluted: Optional[Decimal] = Field(None, description="Diluted EPS")
    
    # Balance sheet
    total_assets: Optional[int] = Field(None, description="Total assets")
    total_debt: Optional[int] = Field(None, description="Total debt")
    shareholders_equity: Optional[int] = Field(None, description="Shareholders equity")
    cash_and_equivalents: Optional[int] = Field(None, description="Cash and equivalents")
    
    # Cash flow (required per instructions)
    operating_cash_flow: Optional[int] = Field(None, description="Operating cash flow")
    free_cash_flow: Optional[int] = Field(None, description="Free cash flow")
    
    # Valuation ratios (required per instructions)
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio")
    price_to_book: Optional[float] = Field(None, description="Price-to-book ratio")
    price_to_sales: Optional[float] = Field(None, description="Price-to-sales ratio")
    
    # Profitability ratios (required per instructions)
    gross_margin: Optional[float] = Field(None, description="Gross profit margin")
    operating_margin: Optional[float] = Field(None, description="Operating margin")
    net_margin: Optional[float] = Field(None, description="Net profit margin")
    
    # Efficiency ratios
    asset_turnover: Optional[float] = Field(None, description="Asset turnover ratio")
    return_on_assets: Optional[float] = Field(None, description="Return on assets")
    return_on_equity: Optional[float] = Field(None, description="Return on equity")
    
    # Sector classification (required for diversification rules)
    sector: Optional[str] = Field(None, description="GICS sector")
    industry: Optional[str] = Field(None, description="GICS industry")
    
    class Config:
        validate_assignment = True
    
    @validator('pe_ratio', 'peg_ratio', 'price_to_book', 'price_to_sales')
    def ratios_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Valuation ratios cannot be negative')
        return v

    @validator('gross_margin', 'operating_margin', 'net_margin')
    def margins_must_be_reasonable(cls, v):
        if v is not None and not (-2.0 <= v <= 2.0):  # -200% to 200%
            raise ValueError('Margins must be between -200% and 200%')
        return v


class EconomicIndicator(BaseModel):
    """Economic indicator data point from FRED API."""
    
    indicator_id: str = Field(..., description="Indicator ID (e.g., GDP, CPI)")
    name: str = Field(..., description="Human-readable name")
    value: Decimal = Field(..., description="Indicator value")
    date: date = Field(..., description="Data date")
    
    # Metadata
    frequency: str = Field(..., description="Data frequency (daily, monthly, etc.)")
    units: str = Field("", description="Units of measurement")
    source: str = Field("FRED", description="Data source")
    
    # Change metrics
    previous_value: Optional[Decimal] = Field(None, description="Previous period value")
    change: Optional[Decimal] = Field(None, description="Period-over-period change")
    change_percent: Optional[float] = Field(None, description="Percentage change")
    
    class Config:
        validate_assignment = True

    def calculate_change_metrics(self, previous_value: Decimal):
        """Calculate change metrics from previous value."""
        self.previous_value = previous_value
        self.change = self.value - previous_value
        
        if previous_value != 0:
            self.change_percent = float((self.change / previous_value) * 100)
        else:
            self.change_percent = None


class SentimentData(BaseModel):
    """Sentiment data for a symbol from various sources."""
    
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Sentiment timestamp")
    
    # Sentiment scores (-1 to 1 scale)
    overall_sentiment: Optional[float] = Field(None, description="Overall sentiment score")
    reddit_sentiment: Optional[float] = Field(None, description="Reddit sentiment")
    twitter_sentiment: Optional[float] = Field(None, description="Twitter sentiment")
    news_sentiment: Optional[float] = Field(None, description="News sentiment")
    
    # Volume metrics
    mention_count: Optional[int] = Field(None, description="Total mentions")
    sentiment_volume: Optional[int] = Field(None, description="Sentiment-weighted volume")
    
    # Analyst sentiment (required per instructions)
    analyst_rating: Optional[str] = Field(None, description="Consensus analyst rating")
    analyst_score: Optional[float] = Field(None, description="Numeric analyst score")
    target_price: Optional[Decimal] = Field(None, description="Consensus target price")
    
    # Insider sentiment (required per instructions)
    insider_sentiment: Optional[float] = Field(None, description="Insider trading sentiment")
    
    class Config:
        validate_assignment = True
    
    @validator('overall_sentiment', 'reddit_sentiment', 'twitter_sentiment', 'news_sentiment', 'analyst_score', 'insider_sentiment')
    def sentiment_must_be_valid(cls, v):
        if v is not None and not (-1 <= v <= 1):
            raise ValueError('Sentiment scores must be between -1 and 1')
        return v


class ETFFlowData(BaseModel):
    """ETF flow data for tracking institutional movements per instructions."""
    
    symbol: str = Field(..., description="ETF symbol")
    date: date = Field(..., description="Flow date")
    
    # Flow metrics (required per instructions)
    net_flow: Decimal = Field(..., description="Net flow (inflow - outflow)")
    inflow: Optional[Decimal] = Field(None, description="Total inflows")
    outflow: Optional[Decimal] = Field(None, description="Total outflows")
    
    # Assets metrics
    aum: Optional[Decimal] = Field(None, description="Assets under management")
    shares_outstanding: Optional[int] = Field(None, description="Shares outstanding")
    
    # Performance metrics
    nav: Optional[Decimal] = Field(None, description="Net asset value")
    premium_discount: Optional[float] = Field(None, description="Premium/discount to NAV")
    
    # Sector exposure (for sector ETFs, required for diversification)
    sector: Optional[str] = Field(None, description="Primary sector")
    
    # Z-score for ranking (required per instructions)
    flow_z_score: Optional[float] = Field(None, description="Flow Z-score")
    
    class Config:
        validate_assignment = True

    def calculate_flow_z_score(self, historical_flows: List[Decimal]) -> float:
        """Calculate flow Z-score for ranking."""
        if len(historical_flows) < 20:  # Need sufficient history
            return 0.0
            
        import statistics
        mean_flow = statistics.mean(historical_flows)
        std_flow = statistics.stdev(historical_flows)
        
        if std_flow == 0:
            return 0.0
            
        z_score = float((self.net_flow - mean_flow) / std_flow)
        return max(-3.0, min(3.0, z_score))  # Cap at ±3


class MarketRegime(str, Enum):
    """Market regime classification for adaptive parameters."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    NORMAL_VOLATILITY = "normal_volatility"


class MarketData(BaseModel):
    """Comprehensive market data container for a ticker."""
    
    symbol: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data timestamp")
    
    # Core market data
    stock_quote: Optional[StockQuote] = Field(None, description="Current stock quote")
    price_history: List[OHLCVData] = Field(default_factory=list, description="Historical price data")
    
    # Technical analysis
    technical_indicators: Optional[TechnicalIndicators] = Field(None, description="Technical indicators")
    
    # Fundamental data
    fundamental_data: Optional[FundamentalData] = Field(None, description="Fundamental metrics")
    
    # Options data
    options_chain: Optional[Any] = Field(None, description="Options chain data")  # Imported from options.py
    
    # Flow and sentiment
    etf_flow_data: Optional[ETFFlowData] = Field(None, description="ETF flow data")
    sentiment_data: Optional[SentimentData] = Field(None, description="Sentiment data")
    
    # Market regime classification
    market_regime: Optional[MarketRegime] = Field(None, description="Current market regime")
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    def is_complete(self) -> bool:
        """Check if market data has minimum required fields for analysis."""
        return (self.stock_quote is not None and 
                len(self.price_history) >= 60 and  # Need 60+ days for technical analysis
                self.technical_indicators is not None)

    def get_sector(self) -> Optional[str]:
        """Get GICS sector for diversification rules."""
        if self.fundamental_data:
            return self.fundamental_data.sector
        return None

    def calculate_iv_rank(self, lookback_days: int = 252) -> Optional[float]:
        """Calculate IV rank percentile for options."""
        if not self.options_chain or len(self.price_history) < lookback_days:
            return None
            
        # This would be implemented based on options chain data
        # Placeholder for now
        return 0.5  # Default to 50th percentile