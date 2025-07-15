"""
Market data models for stocks, fundamentals, technicals, and economic indicators.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field, validator


class StockQuote(BaseModel):
    """Real-time stock quote."""
    
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


class OHLCVData(BaseModel):
    """OHLCV (candlestick) data point."""
    
    symbol: str = Field(..., description="Symbol")
    timestamp: datetime = Field(..., description="Timestamp")
    timeframe: str = Field(..., description="Timeframe (1d, 1h, etc.)")
    
    open: Decimal = Field(..., description="Open price")
    high: Decimal = Field(..., description="High price") 
    low: Decimal = Field(..., description="Low price")
    close: Decimal = Field(..., description="Close price")
    volume: int = Field(..., description="Volume")
    
    class Config:
        frozen = True
        validate_assignment = True
    
    @validator('open', 'high', 'low', 'close')
    def prices_must_be_positive(cls, v):
        if v <= 0:
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


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators for a symbol."""
    
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    
    # Moving averages
    sma_20: Optional[Decimal] = Field(None, description="20-day simple moving average")
    sma_50: Optional[Decimal] = Field(None, description="50-day simple moving average")
    sma_100: Optional[Decimal] = Field(None, description="100-day simple moving average")
    sma_200: Optional[Decimal] = Field(None, description="200-day simple moving average")
    
    ema_12: Optional[Decimal] = Field(None, description="12-day exponential moving average")
    ema_26: Optional[Decimal] = Field(None, description="26-day exponential moving average")
    
    # Momentum indicators
    rsi_14: Optional[float] = Field(None, description="14-day RSI")
    macd: Optional[Decimal] = Field(None, description="MACD line")
    macd_signal: Optional[Decimal] = Field(None, description="MACD signal line")
    macd_histogram: Optional[Decimal] = Field(None, description="MACD histogram")
    
    # Volatility indicators
    bollinger_upper: Optional[Decimal] = Field(None, description="Bollinger upper band")
    bollinger_middle: Optional[Decimal] = Field(None, description="Bollinger middle band")
    bollinger_lower: Optional[Decimal] = Field(None, description="Bollinger lower band")
    atr_14: Optional[Decimal] = Field(None, description="14-day Average True Range")
    
    # Volume indicators
    vwap: Optional[Decimal] = Field(None, description="Volume Weighted Average Price")
    
    # Historical volatility
    hv_30: Optional[float] = Field(None, description="30-day historical volatility")
    hv_60: Optional[float] = Field(None, description="60-day historical volatility")
    
    # Momentum metrics
    momentum_1d: Optional[float] = Field(None, description="1-day momentum")
    momentum_5d: Optional[float] = Field(None, description="5-day momentum")
    momentum_20d: Optional[float] = Field(None, description="20-day momentum")
    
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


class FundamentalData(BaseModel):
    """Fundamental financial data for a company."""
    
    symbol: str = Field(..., description="Stock symbol")
    report_date: date = Field(..., description="Report date")
    period_type: str = Field(..., description="Period type (quarterly, annual, ttm)")
    
    # Income statement
    revenue: Optional[int] = Field(None, description="Total revenue")
    gross_profit: Optional[int] = Field(None, description="Gross profit")
    operating_income: Optional[int] = Field(None, description="Operating income")
    net_income: Optional[int] = Field(None, description="Net income")
    
    # Per-share metrics
    eps: Optional[Decimal] = Field(None, description="Earnings per share")
    eps_diluted: Optional[Decimal] = Field(None, description="Diluted EPS")
    
    # Balance sheet
    total_assets: Optional[int] = Field(None, description="Total assets")
    total_debt: Optional[int] = Field(None, description="Total debt")
    shareholders_equity: Optional[int] = Field(None, description="Shareholders equity")
    cash_and_equivalents: Optional[int] = Field(None, description="Cash and equivalents")
    
    # Cash flow
    operating_cash_flow: Optional[int] = Field(None, description="Operating cash flow")
    free_cash_flow: Optional[int] = Field(None, description="Free cash flow")
    
    # Valuation ratios
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio")
    price_to_book: Optional[float] = Field(None, description="Price-to-book ratio")
    price_to_sales: Optional[float] = Field(None, description="Price-to-sales ratio")
    
    # Profitability ratios
    gross_margin: Optional[float] = Field(None, description="Gross profit margin")
    operating_margin: Optional[float] = Field(None, description="Operating margin")
    net_margin: Optional[float] = Field(None, description="Net profit margin")
    
    # Efficiency ratios
    asset_turnover: Optional[float] = Field(None, description="Asset turnover ratio")
    return_on_assets: Optional[float] = Field(None, description="Return on assets")
    return_on_equity: Optional[float] = Field(None, description="Return on equity")
    
    class Config:
        validate_assignment = True
    
    @validator('pe_ratio', 'peg_ratio', 'price_to_book', 'price_to_sales')
    def ratios_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Valuation ratios cannot be negative')
        return v


class EconomicIndicator(BaseModel):
    """Economic indicator data point."""
    
    indicator_id: str = Field(..., description="Indicator ID (e.g., GDP, CPI)")
    name: str = Field(..., description="Human-readable name")
    value: Decimal = Field(..., description="Indicator value")
    date: date = Field(..., description="Data date")
    
    # Metadata
    frequency: str = Field(..., description="Data frequency (daily, monthly, etc.)")
    units: str = Field("", description="Units of measurement")
    source: str = Field("", description="Data source")
    
    # Change metrics
    previous_value: Optional[Decimal] = Field(None, description="Previous period value")
    change: Optional[Decimal] = Field(None, description="Period-over-period change")
    change_percent: Optional[float] = Field(None, description="Percentage change")
    
    class Config:
        validate_assignment = True


class SentimentData(BaseModel):
    """Sentiment data for a symbol."""
    
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
    
    # Analyst sentiment
    analyst_rating: Optional[str] = Field(None, description="Consensus analyst rating")
    analyst_score: Optional[float] = Field(None, description="Numeric analyst score")
    target_price: Optional[Decimal] = Field(None, description="Consensus target price")
    
    class Config:
        validate_assignment = True
    
    @validator('overall_sentiment', 'reddit_sentiment', 'twitter_sentiment', 'news_sentiment', 'analyst_score')
    def sentiment_must_be_valid(cls, v):
        if v is not None and not (-1 <= v <= 1):
            raise ValueError('Sentiment scores must be between -1 and 1')
        return v


class ETFFlowData(BaseModel):
    """ETF flow data for tracking institutional movements."""
    
    symbol: str = Field(..., description="ETF symbol")
    date: date = Field(..., description="Flow date")
    
    # Flow metrics
    net_flow: Decimal = Field(..., description="Net flow (inflow - outflow)")
    inflow: Optional[Decimal] = Field(None, description="Total inflows")
    outflow: Optional[Decimal] = Field(None, description="Total outflows")
    
    # Assets metrics
    aum: Optional[Decimal] = Field(None, description="Assets under management")
    shares_outstanding: Optional[int] = Field(None, description="Shares outstanding")
    
    # Performance metrics
    nav: Optional[Decimal] = Field(None, description="Net asset value")
    premium_discount: Optional[float] = Field(None, description="Premium/discount to NAV")
    
    # Sector exposure (for sector ETFs)
    sector: Optional[str] = Field(None, description="Primary sector")
    
    class Config:
        validate_assignment = True