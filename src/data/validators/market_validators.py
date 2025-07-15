"""
Market data validators for ensuring data quality and consistency.
"""

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
import re

from .options_validators import ValidationResult, ValidationSeverity
from ..models.market_data import (
    StockQuote, FundamentalData, TechnicalIndicators, 
    EconomicIndicator, SentimentData, ETFFlowData, OHLCVData
)

logger = logging.getLogger(__name__)


class StockQuoteValidator:
    """Validator for stock quote data."""
    
    @staticmethod
    def validate(quote: StockQuote) -> ValidationResult:
        """Validate stock quote data."""
        result = ValidationResult()
        
        # Symbol validation
        if not quote.symbol:
            result.add_error("Stock symbol is required", "symbol")
        elif not StockQuoteValidator._is_valid_symbol(quote.symbol):
            result.add_warning(f"Symbol '{quote.symbol}' format may be invalid", "symbol")
        
        # Price validation
        if quote.price <= 0:
            result.add_error(f"Stock price {quote.price} must be positive", "price")
        elif quote.price > Decimal('10000'):  # $10,000 per share seems excessive
            result.add_warning(f"Stock price {quote.price} is unusually high", "price")
        elif quote.price < Decimal('0.01'):  # Penny stock
            result.add_info(f"Stock price {quote.price} is a penny stock", "price")
        
        # Bid/Ask validation
        StockQuoteValidator._validate_bid_ask(quote, result)
        
        # OHLC validation
        StockQuoteValidator._validate_ohlc(quote, result)
        
        # Volume validation
        StockQuoteValidator._validate_volume(quote, result)
        
        # Change validation
        StockQuoteValidator._validate_change_metrics(quote, result)
        
        # Quote age validation
        if quote.quote_time:
            age_minutes = (datetime.utcnow() - quote.quote_time).total_seconds() / 60
            if age_minutes > 60:
                result.add_warning(f"Quote is {age_minutes:.1f} minutes old", "quote_time")
            elif age_minutes > 15:
                result.add_info(f"Quote is {age_minutes:.1f} minutes old", "quote_time")
        else:
            result.add_warning("Quote timestamp is missing", "quote_time")
        
        return result
    
    @staticmethod
    def _is_valid_symbol(symbol: str) -> bool:
        """Check if symbol format is valid."""
        # Basic symbol validation (1-5 uppercase letters)
        pattern = r'^[A-Z]{1,5}$'
        return bool(re.match(pattern, symbol))
    
    @staticmethod
    def _validate_bid_ask(quote: StockQuote, result: ValidationResult):
        """Validate bid/ask prices."""
        if quote.bid is not None and quote.ask is not None:
            if quote.bid > quote.ask:
                result.add_error(f"Bid {quote.bid} cannot be greater than ask {quote.ask}", "bid_ask")
            
            # Check spread
            spread = quote.ask - quote.bid
            spread_pct = spread / quote.price if quote.price > 0 else 0
            
            if spread_pct > Decimal('0.05'):  # 5% spread
                result.add_warning(f"Wide bid-ask spread: {spread_pct:.2%}", "bid_ask_spread")
            
            # Check if current price is within bid-ask
            if not (quote.bid <= quote.price <= quote.ask):
                result.add_warning("Current price is outside bid-ask range", "price_range")
        
        # Individual bid/ask validation
        if quote.bid is not None and quote.bid <= 0:
            result.add_error(f"Bid price {quote.bid} must be positive", "bid")
        
        if quote.ask is not None and quote.ask <= 0:
            result.add_error(f"Ask price {quote.ask} must be positive", "ask")
    
    @staticmethod
    def _validate_ohlc(quote: StockQuote, result: ValidationResult):
        """Validate OHLC prices."""
        ohlc_prices = {
            'open': quote.open,
            'high': quote.high,
            'low': quote.low,
            'previous_close': quote.previous_close
        }
        
        # Check for negative prices
        for name, price in ohlc_prices.items():
            if price is not None and price <= 0:
                result.add_error(f"{name} price {price} must be positive", name)
        
        # OHLC relationship validation
        if all(p is not None for p in [quote.open, quote.high, quote.low, quote.price]):
            if not (quote.low <= quote.open <= quote.high):
                result.add_error("Open price not within high-low range", "ohlc_relationship")
            
            if not (quote.low <= quote.price <= quote.high):
                result.add_error("Current price not within high-low range", "ohlc_relationship")
            
            # Check for unrealistic ranges
            range_pct = (quote.high - quote.low) / quote.price if quote.price > 0 else 0
            if range_pct > Decimal('0.20'):  # 20% daily range
                result.add_warning(f"Large daily range: {range_pct:.1%}", "daily_range")
    
    @staticmethod
    def _validate_volume(quote: StockQuote, result: ValidationResult):
        """Validate volume data."""
        if quote.volume is not None:
            if quote.volume < 0:
                result.add_error(f"Volume {quote.volume} cannot be negative", "volume")
            elif quote.volume == 0:
                result.add_info("No volume traded", "volume")
        
        if quote.avg_volume is not None:
            if quote.avg_volume < 0:
                result.add_error(f"Average volume {quote.avg_volume} cannot be negative", "avg_volume")
            elif quote.volume is not None and quote.avg_volume > 0:
                volume_ratio = quote.volume / quote.avg_volume
                if volume_ratio > 5:
                    result.add_info(f"High volume: {volume_ratio:.1f}x average", "volume_spike")
                elif volume_ratio < 0.1:
                    result.add_info(f"Low volume: {volume_ratio:.1f}x average", "volume_low")
    
    @staticmethod
    def _validate_change_metrics(quote: StockQuote, result: ValidationResult):
        """Validate change and percentage change."""
        if quote.change is not None and quote.previous_close is not None:
            expected_change = quote.price - quote.previous_close
            actual_change = quote.change
            
            # Allow small rounding differences
            if abs(expected_change - actual_change) > Decimal('0.01'):
                result.add_warning("Change calculation may be incorrect", "change")
        
        if quote.change_percent is not None:
            if abs(quote.change_percent) > 50:  # 50% daily change
                result.add_warning(f"Large daily change: {quote.change_percent:.1%}", "change_percent")
            
            # Cross-validate with change amount
            if quote.change is not None and quote.previous_close is not None and quote.previous_close > 0:
                expected_pct = float(quote.change / quote.previous_close) * 100
                if abs(expected_pct - quote.change_percent) > 0.1:
                    result.add_warning("Change percentage may be incorrect", "change_percent")


class OHLCVDataValidator:
    """Validator for OHLCV candlestick data."""
    
    @staticmethod
    def validate(ohlcv: OHLCVData) -> ValidationResult:
        """Validate OHLCV data point."""
        result = ValidationResult()
        
        # Symbol validation
        if not ohlcv.symbol:
            result.add_error("Symbol is required", "symbol")
        
        # Timestamp validation
        if not ohlcv.timestamp:
            result.add_error("Timestamp is required", "timestamp")
        
        # Price validation
        prices = [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]
        price_names = ['open', 'high', 'low', 'close']
        
        for price, name in zip(prices, price_names):
            if price <= 0:
                result.add_error(f"{name} price {price} must be positive", name)
        
        # OHLC relationship validation
        if not (ohlcv.low <= ohlcv.open <= ohlcv.high):
            result.add_error("Open price not within high-low range", "ohlc_relationship")
        
        if not (ohlcv.low <= ohlcv.close <= ohlcv.high):
            result.add_error("Close price not within high-low range", "ohlc_relationship")
        
        # Volume validation
        if ohlcv.volume < 0:
            result.add_error(f"Volume {ohlcv.volume} cannot be negative", "volume")
        
        # Range validation
        if ohlcv.high > 0 and ohlcv.low > 0:
            range_pct = float((ohlcv.high - ohlcv.low) / ohlcv.close)
            if range_pct > 0.50:  # 50% range
                result.add_warning(f"Large price range: {range_pct:.1%}", "price_range")
        
        return result


class FundamentalDataValidator:
    """Validator for fundamental financial data."""
    
    @staticmethod
    def validate(fundamentals: FundamentalData) -> ValidationResult:
        """Validate fundamental data."""
        result = ValidationResult()
        
        # Basic validation
        if not fundamentals.symbol:
            result.add_error("Symbol is required", "symbol")
        
        if not fundamentals.report_date:
            result.add_error("Report date is required", "report_date")
        
        # Date validation
        if fundamentals.report_date > date.today():
            result.add_warning("Report date is in the future", "report_date")
        
        # Financial metrics validation
        FundamentalDataValidator._validate_income_statement(fundamentals, result)
        FundamentalDataValidator._validate_balance_sheet(fundamentals, result)
        FundamentalDataValidator._validate_ratios(fundamentals, result)
        FundamentalDataValidator._validate_margins(fundamentals, result)
        
        return result
    
    @staticmethod
    def _validate_income_statement(fundamentals: FundamentalData, result: ValidationResult):
        """Validate income statement metrics."""
        # Revenue validation
        if fundamentals.revenue is not None and fundamentals.revenue < 0:
            result.add_warning("Negative revenue reported", "revenue")
        
        # Profit relationships
        if (fundamentals.revenue is not None and fundamentals.gross_profit is not None and 
            fundamentals.gross_profit > fundamentals.revenue):
            result.add_error("Gross profit cannot exceed revenue", "gross_profit")
        
        if (fundamentals.gross_profit is not None and fundamentals.operating_income is not None and
            fundamentals.operating_income > fundamentals.gross_profit):
            result.add_error("Operating income cannot exceed gross profit", "operating_income")
        
        # EPS validation
        if fundamentals.eps is not None and fundamentals.eps_diluted is not None:
            if fundamentals.eps_diluted > fundamentals.eps:
                result.add_warning("Diluted EPS higher than basic EPS", "eps_diluted")
    
    @staticmethod
    def _validate_balance_sheet(fundamentals: FundamentalData, result: ValidationResult):
        """Validate balance sheet metrics."""
        # Check for negative values that shouldn't be negative
        if fundamentals.total_assets is not None and fundamentals.total_assets < 0:
            result.add_error("Total assets cannot be negative", "total_assets")
        
        if fundamentals.cash_and_equivalents is not None and fundamentals.cash_and_equivalents < 0:
            result.add_warning("Negative cash and equivalents", "cash_and_equivalents")
        
        # Asset relationships
        if (fundamentals.total_assets is not None and fundamentals.cash_and_equivalents is not None and
            fundamentals.cash_and_equivalents > fundamentals.total_assets):
            result.add_error("Cash cannot exceed total assets", "cash_and_equivalents")
    
    @staticmethod
    def _validate_ratios(fundamentals: FundamentalData, result: ValidationResult):
        """Validate financial ratios."""
        # P/E ratio validation
        if fundamentals.pe_ratio is not None:
            if fundamentals.pe_ratio < 0:
                result.add_info("Negative P/E ratio (company has losses)", "pe_ratio")
            elif fundamentals.pe_ratio > 1000:
                result.add_warning(f"Very high P/E ratio: {fundamentals.pe_ratio}", "pe_ratio")
        
        # PEG ratio validation
        if fundamentals.peg_ratio is not None:
            if fundamentals.peg_ratio < 0:
                result.add_warning("Negative PEG ratio", "peg_ratio")
            elif fundamentals.peg_ratio > 10:
                result.add_warning(f"Very high PEG ratio: {fundamentals.peg_ratio}", "peg_ratio")
        
        # Price-to-book validation
        if fundamentals.price_to_book is not None:
            if fundamentals.price_to_book < 0:
                result.add_warning("Negative price-to-book ratio", "price_to_book")
            elif fundamentals.price_to_book > 50:
                result.add_warning(f"Very high price-to-book: {fundamentals.price_to_book}", "price_to_book")
    
    @staticmethod
    def _validate_margins(fundamentals: FundamentalData, result: ValidationResult):
        """Validate profit margins."""
        margins = [
            ('gross_margin', fundamentals.gross_margin),
            ('operating_margin', fundamentals.operating_margin),
            ('net_margin', fundamentals.net_margin)
        ]
        
        for name, margin in margins:
            if margin is not None:
                if margin < -1.0:  # -100%
                    result.add_warning(f"Very low {name}: {margin:.1%}", name)
                elif margin > 1.0:  # 100%
                    result.add_warning(f"Unusually high {name}: {margin:.1%}", name)
        
        # Margin relationships
        if (fundamentals.gross_margin is not None and fundamentals.operating_margin is not None and
            fundamentals.operating_margin > fundamentals.gross_margin):
            result.add_error("Operating margin cannot exceed gross margin", "margin_relationship")


class TechnicalIndicatorValidator:
    """Validator for technical indicators."""
    
    @staticmethod
    def validate(indicators: TechnicalIndicators) -> ValidationResult:
        """Validate technical indicators."""
        result = ValidationResult()
        
        # Basic validation
        if not indicators.symbol:
            result.add_error("Symbol is required", "symbol")
        
        if not indicators.timestamp:
            result.add_error("Timestamp is required", "timestamp")
        
        # RSI validation
        if indicators.rsi_14 is not None:
            if not (0 <= indicators.rsi_14 <= 100):
                result.add_error(f"RSI {indicators.rsi_14} must be between 0 and 100", "rsi_14")
            else:
                if indicators.rsi_14 > 80:
                    result.add_info(f"RSI indicates overbought: {indicators.rsi_14}", "rsi_14")
                elif indicators.rsi_14 < 20:
                    result.add_info(f"RSI indicates oversold: {indicators.rsi_14}", "rsi_14")
        
        # Moving average validation
        TechnicalIndicatorValidator._validate_moving_averages(indicators, result)
        
        # Volatility validation
        TechnicalIndicatorValidator._validate_volatility(indicators, result)
        
        # Momentum validation
        TechnicalIndicatorValidator._validate_momentum(indicators, result)
        
        return result
    
    @staticmethod
    def _validate_moving_averages(indicators: TechnicalIndicators, result: ValidationResult):
        """Validate moving averages."""
        mas = [
            ('sma_20', indicators.sma_20),
            ('sma_50', indicators.sma_50),
            ('sma_100', indicators.sma_100),
            ('sma_200', indicators.sma_200)
        ]
        
        for name, ma in mas:
            if ma is not None and ma <= 0:
                result.add_error(f"{name} {ma} must be positive", name)
        
        # Check MA ordering (longer period should typically be smoother)
        ma_values = [(20, indicators.sma_20), (50, indicators.sma_50), 
                     (100, indicators.sma_100), (200, indicators.sma_200)]
        
        valid_mas = [(period, ma) for period, ma in ma_values if ma is not None]
        
        if len(valid_mas) >= 2:
            # In trending markets, shorter MAs can be above/below longer ones
            # This is just informational
            for i in range(len(valid_mas) - 1):
                period1, ma1 = valid_mas[i]
                period2, ma2 = valid_mas[i + 1]
                
                if abs(float(ma1 - ma2)) / float(ma1) > 0.20:  # 20% difference
                    result.add_info(f"Large difference between MA{period1} and MA{period2}", "ma_divergence")
    
    @staticmethod
    def _validate_volatility(indicators: TechnicalIndicators, result: ValidationResult):
        """Validate volatility indicators."""
        if indicators.hv_30 is not None:
            if indicators.hv_30 < 0:
                result.add_error(f"Historical volatility {indicators.hv_30} cannot be negative", "hv_30")
            elif indicators.hv_30 > 200:  # 200% annualized
                result.add_warning(f"Very high historical volatility: {indicators.hv_30:.1f}%", "hv_30")
        
        if indicators.hv_60 is not None:
            if indicators.hv_60 < 0:
                result.add_error(f"Historical volatility {indicators.hv_60} cannot be negative", "hv_60")
        
        # ATR validation
        if indicators.atr_14 is not None:
            if indicators.atr_14 < 0:
                result.add_error(f"ATR {indicators.atr_14} cannot be negative", "atr_14")
    
    @staticmethod
    def _validate_momentum(indicators: TechnicalIndicators, result: ValidationResult):
        """Validate momentum indicators."""
        momentum_indicators = [
            ('momentum_1d', indicators.momentum_1d),
            ('momentum_5d', indicators.momentum_5d),
            ('momentum_20d', indicators.momentum_20d)
        ]
        
        for name, momentum in momentum_indicators:
            if momentum is not None:
                if abs(momentum) > 50:  # 50% move
                    result.add_warning(f"Large momentum in {name}: {momentum:.1f}%", name)


class EconomicIndicatorValidator:
    """Validator for economic indicators."""
    
    @staticmethod
    def validate(indicator: EconomicIndicator) -> ValidationResult:
        """Validate economic indicator."""
        result = ValidationResult()
        
        # Basic validation
        if not indicator.indicator_id:
            result.add_error("Indicator ID is required", "indicator_id")
        
        if not indicator.name:
            result.add_error("Indicator name is required", "name")
        
        if not indicator.date:
            result.add_error("Date is required", "date")
        
        # Date validation
        if indicator.date > date.today():
            result.add_warning("Indicator date is in the future", "date")
        
        # Value validation (depends on indicator type)
        EconomicIndicatorValidator._validate_by_indicator_type(indicator, result)
        
        # Change validation
        if indicator.change_percent is not None:
            if abs(indicator.change_percent) > 100:  # 100% change
                result.add_warning(f"Large change: {indicator.change_percent:.1f}%", "change_percent")
        
        return result
    
    @staticmethod
    def _validate_by_indicator_type(indicator: EconomicIndicator, result: ValidationResult):
        """Validate based on indicator type."""
        value = float(indicator.value)
        
        # GDP validation
        if 'GDP' in indicator.indicator_id.upper():
            if value < 0:
                result.add_warning("Negative GDP reported", "value")
        
        # Unemployment rate validation
        elif 'UNRATE' in indicator.indicator_id or 'UNEMPLOYMENT' in indicator.name.upper():
            if not (0 <= value <= 100):
                result.add_warning(f"Unemployment rate {value}% outside typical range", "value")
        
        # Interest rate validation
        elif any(term in indicator.indicator_id.upper() for term in ['RATE', 'YIELD', 'FEDFUNDS']):
            if value < 0:
                result.add_info(f"Negative interest rate: {value}%", "value")
            elif value > 25:
                result.add_warning(f"Very high interest rate: {value}%", "value")
        
        # Inflation validation (CPI)
        elif 'CPI' in indicator.indicator_id.upper():
            if value < 0:
                result.add_info(f"Deflation detected: {value}%", "value")
            elif value > 20:
                result.add_warning(f"Very high inflation: {value}%", "value")


class MarketDataValidator:
    """Main validator class for all market data types."""
    
    @staticmethod
    def validate_stock_quote(quote: StockQuote, strict: bool = False) -> ValidationResult:
        """Validate stock quote with configurable strictness."""
        result = StockQuoteValidator.validate(quote)
        
        if strict:
            # Convert warnings to errors in strict mode
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    issue.severity = ValidationSeverity.ERROR
                    result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_ohlcv_data(ohlcv: OHLCVData) -> ValidationResult:
        """Validate OHLCV data."""
        return OHLCVDataValidator.validate(ohlcv)
    
    @staticmethod
    def validate_fundamental_data(fundamentals: FundamentalData) -> ValidationResult:
        """Validate fundamental data."""
        return FundamentalDataValidator.validate(fundamentals)
    
    @staticmethod
    def validate_technical_indicators(indicators: TechnicalIndicators) -> ValidationResult:
        """Validate technical indicators."""
        return TechnicalIndicatorValidator.validate(indicators)
    
    @staticmethod
    def validate_economic_indicator(indicator: EconomicIndicator) -> ValidationResult:
        """Validate economic indicator."""
        return EconomicIndicatorValidator.validate(indicator)
    
    @staticmethod
    def validate_data_freshness(timestamp: datetime, max_age_minutes: float = 15.0) -> ValidationResult:
        """Validate data freshness."""
        result = ValidationResult()
        
        if not timestamp:
            result.add_error("Timestamp is required for freshness check", "timestamp")
            return result
        
        age_minutes = (datetime.utcnow() - timestamp).total_seconds() / 60
        
        if age_minutes > max_age_minutes:
            result.add_error(f"Data is stale: {age_minutes:.1f} minutes old", "timestamp")
        elif age_minutes > max_age_minutes * 0.8:
            result.add_warning(f"Data is getting stale: {age_minutes:.1f} minutes old", "timestamp")
        
        return result
    
    @staticmethod
    def validate_symbol_format(symbol: str) -> ValidationResult:
        """Validate symbol format."""
        result = ValidationResult()
        
        if not symbol:
            result.add_error("Symbol is required", "symbol")
            return result
        
        # Basic format validation
        if not symbol.isupper():
            result.add_warning("Symbol should be uppercase", "symbol")
        
        if not symbol.isalpha():
            result.add_warning("Symbol contains non-alphabetic characters", "symbol")
        
        if len(symbol) > 5:
            result.add_warning("Symbol is longer than typical (>5 characters)", "symbol")
        
        return result