"""
pytest configuration file for the options trading engine tests.
Provides shared fixtures and test configuration.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import tempfile
import os
import logging

from src.data.models.options import OptionQuote, OptionType, Greeks
from src.data.models.trades import (
    TradeCandidate, StrategyDefinition, TradeLeg, StrategyType, TradeDirection
)
from src.data.models.market_data import StockQuote, TechnicalIndicators
from src.domain.services.scoring_engine import ScoredTradeCandidate, ComponentScores
from src.infrastructure.cache.cache_manager import CacheManager
from src.application.config.settings import ConfigurationManager


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Base test configuration."""
    return {
        'nav': Decimal('100000'),
        'available_capital': Decimal('10000'),
        'max_loss_per_trade': Decimal('500'),
        'min_pop': 0.65,
        'min_credit_ratio': 0.33,
        'max_quote_age_minutes': 10,
        'api_keys': {
            'tradier': 'test_tradier_key',
            'yahoo': 'test_yahoo_key',
            'fred': 'test_fred_key',
            'quiver': 'test_quiver_key'
        },
        'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JNJ'],
        'test_mode': True
    }


# Data fixtures
@pytest.fixture
def sample_option_quote():
    """Sample option quote for testing."""
    return OptionQuote(
        symbol="AAPL250117P00150000",
        strike=Decimal('150'),
        expiration=date(2025, 1, 17),
        option_type=OptionType.PUT,
        bid=Decimal('2.50'),
        ask=Decimal('2.55'),
        last=Decimal('2.52'),
        volume=100,
        open_interest=500,
        quote_timestamp=datetime.utcnow(),
        greeks=Greeks(
            delta=-0.25,
            gamma=0.05,
            theta=-0.08,
            vega=0.12,
            rho=-0.02
        ),
        implied_volatility=0.25
    )


@pytest.fixture
def sample_stock_quote():
    """Sample stock quote for testing."""
    return StockQuote(
        symbol="AAPL",
        last=Decimal('150.25'),
        bid=Decimal('150.20'),
        ask=Decimal('150.30'),
        volume=1000000,
        change=Decimal('1.25'),
        change_percentage=Decimal('0.84'),
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_trade_candidate(sample_option_quote):
    """Sample trade candidate for testing."""
    leg = TradeLeg(
        option=sample_option_quote,
        quantity=1,
        direction=TradeDirection.SELL
    )
    
    strategy = StrategyDefinition(
        strategy_type=StrategyType.PUT_CREDIT_SPREAD,
        underlying="AAPL",
        legs=[leg],
        probability_of_profit=0.70,
        net_credit=Decimal('2.0'),
        max_loss=Decimal('300'),
        max_profit=Decimal('200'),
        credit_to_max_loss_ratio=0.67,
        days_to_expiration=30,
        margin_requirement=Decimal('300')
    )
    
    return TradeCandidate(strategy=strategy)


@pytest.fixture
def sample_scored_trade(sample_trade_candidate):
    """Sample scored trade candidate for testing."""
    component_scores = ComponentScores(
        model_score=75.0,
        pop_score=70.0,
        iv_rank_score=65.0,
        momentum_z=0.5,
        flow_z=0.3,
        risk_reward_score=80.0,
        liquidity_score=85.0
    )
    
    return ScoredTradeCandidate(
        trade_candidate=sample_trade_candidate,
        component_scores=component_scores
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        'AAPL': {
            'stock_quote': {
                'symbol': 'AAPL',
                'last': 150.25,
                'bid': 150.20,
                'ask': 150.30,
                'volume': 1000000,
                'timestamp': datetime.utcnow()
            },
            'options_chain': [
                {
                    'symbol': 'AAPL250117P00150000',
                    'strike': 150.0,
                    'expiration': '2025-01-17',
                    'option_type': 'put',
                    'bid': 2.50,
                    'ask': 2.55,
                    'last': 2.52,
                    'volume': 100,
                    'open_interest': 500,
                    'delta': -0.25,
                    'gamma': 0.05,
                    'theta': -0.08,
                    'vega': 0.12,
                    'rho': -0.02,
                    'implied_volatility': 0.25
                },
                {
                    'symbol': 'AAPL250117P00145000',
                    'strike': 145.0,
                    'expiration': '2025-01-17',
                    'option_type': 'put',
                    'bid': 1.80,
                    'ask': 1.85,
                    'last': 1.82,
                    'volume': 80,
                    'open_interest': 400,
                    'delta': -0.18,
                    'gamma': 0.04,
                    'theta': -0.06,
                    'vega': 0.10,
                    'rho': -0.015,
                    'implied_volatility': 0.23
                }
            ],
            'fundamentals': {
                'trailing_pe': 28.5,
                'peg_ratio': 1.2,
                'gross_margins': 0.38,
                'operating_margins': 0.27,
                'sector': 'Technology',
                'market_cap': 2800000000000
            },
            'technical_indicators': {
                'rsi': 65.0,
                'macd': 2.5,
                'sma_50': 148.0,
                'sma_200': 145.0,
                'bollinger_upper': 155.0,
                'bollinger_lower': 145.0,
                'atr': 3.2,
                'momentum_20d': 0.05,
                'volatility_30d': 0.25
            },
            'flow_data': {
                'put_call_ratio': 0.8,
                'volume_vs_oi_ratio': 1.2,
                'etf_flow_z': 0.5,
                'insider_sentiment': 0.2,
                'institutional_flow': 0.3
            },
            'price_history': [
                {'date': date.today() - timedelta(days=i), 
                 'open': 150.0 + i * 0.1, 
                 'high': 151.0 + i * 0.1, 
                 'low': 149.0 + i * 0.1, 
                 'close': 150.5 + i * 0.1, 
                 'volume': 1000000 + i * 1000}
                for i in range(252)
            ]
        }
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical data for backtesting."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    historical_data = {}
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        historical_data[ticker] = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 252)),
            'High': prices * (1 + np.random.uniform(0.0, 0.02, 252)),
            'Low': prices * (1 - np.random.uniform(0.0, 0.02, 252)),
            'Close': prices,
            'Volume': np.random.randint(500000, 2000000, 252)
        }, index=dates)
    
    return historical_data


# Mock fixtures
@pytest.fixture
def mock_tradier_client():
    """Mock Tradier API client."""
    mock_client = Mock()
    
    # Mock options chain response
    mock_client.get_options_chain.return_value = [
        {
            'symbol': 'AAPL250117P00150000',
            'strike': 150.0,
            'expiration': '2025-01-17',
            'option_type': 'put',
            'bid': 2.50,
            'ask': 2.55,
            'last': 2.52,
            'volume': 100,
            'open_interest': 500,
            'delta': -0.25,
            'gamma': 0.05,
            'theta': -0.08,
            'vega': 0.12,
            'rho': -0.02,
            'implied_volatility': 0.25
        }
    ]
    
    # Mock stock quotes response
    mock_client.get_stock_quotes.return_value = [
        {
            'symbol': 'AAPL',
            'last': 150.25,
            'bid': 150.20,
            'ask': 150.30,
            'volume': 1000000,
            'change': 1.25,
            'change_percentage': 0.84
        }
    ]
    
    return mock_client


@pytest.fixture
def mock_yahoo_client():
    """Mock Yahoo Finance API client."""
    mock_client = Mock()
    
    # Mock fundamentals response
    mock_client.get_fundamentals.return_value = {
        'trailing_pe': 28.5,
        'peg_ratio': 1.2,
        'gross_margins': 0.38,
        'operating_margins': 0.27,
        'sector': 'Technology',
        'market_cap': 2800000000000
    }
    
    # Mock price history response
    mock_client.get_price_history.return_value = [
        {'date': date.today() - timedelta(days=i), 
         'open': 150.0, 'high': 151.0, 'low': 149.0, 'close': 150.5, 'volume': 1000000}
        for i in range(30)
    ]
    
    return mock_client


@pytest.fixture
def mock_fred_client():
    """Mock FRED API client."""
    mock_client = Mock()
    
    # Mock economic data response
    mock_client.get_economic_series.return_value = [
        {'date': date.today() - timedelta(days=i * 30), 'value': 3.2 + i * 0.1}
        for i in range(12)
    ]
    
    return mock_client


# Service fixtures
@pytest.fixture
def cache_manager():
    """Cache manager for testing."""
    return CacheManager()


@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_database_url():
    """Test database URL."""
    return "sqlite:///:memory:"


# Test utilities
@pytest.fixture
def create_test_trade():
    """Factory function to create test trades."""
    def _create_test_trade(
        ticker="AAPL",
        strategy_type=StrategyType.PUT_CREDIT_SPREAD,
        pop=0.70,
        net_credit=2.0,
        max_loss=300,
        days_to_expiration=30,
        **kwargs
    ):
        option = OptionQuote(
            symbol=f"{ticker}250117P00150000",
            strike=Decimal('150'),
            expiration=date(2025, 1, 17),
            option_type=OptionType.PUT,
            bid=Decimal('2.50'),
            ask=Decimal('2.55'),
            last=Decimal('2.52'),
            volume=100,
            open_interest=500,
            quote_timestamp=datetime.utcnow(),
            greeks=Greeks(
                delta=-0.25,
                gamma=0.05,
                theta=-0.08,
                vega=0.12,
                rho=-0.02
            ),
            implied_volatility=0.25
        )
        
        leg = TradeLeg(
            option=option,
            quantity=1,
            direction=TradeDirection.SELL
        )
        
        strategy = StrategyDefinition(
            strategy_type=strategy_type,
            underlying=ticker,
            legs=[leg],
            probability_of_profit=pop,
            net_credit=Decimal(str(net_credit)),
            max_loss=Decimal(str(max_loss)),
            max_profit=Decimal(str(net_credit)),
            credit_to_max_loss_ratio=net_credit / max_loss,
            days_to_expiration=days_to_expiration,
            margin_requirement=Decimal(str(max_loss)),
            **kwargs
        )
        
        return TradeCandidate(strategy=strategy)
    
    return _create_test_trade


@pytest.fixture
def create_scored_trade():
    """Factory function to create scored trades."""
    def _create_scored_trade(
        ticker="AAPL",
        model_score=75.0,
        pop_score=70.0,
        momentum_z=0.5,
        flow_z=0.3,
        **kwargs
    ):
        # Create base trade
        option = OptionQuote(
            symbol=f"{ticker}250117P00150000",
            strike=Decimal('150'),
            expiration=date(2025, 1, 17),
            option_type=OptionType.PUT,
            bid=Decimal('2.50'),
            ask=Decimal('2.55'),
            last=Decimal('2.52'),
            volume=100,
            open_interest=500,
            greeks=Greeks(delta=-0.25, gamma=0.05, theta=-0.08, vega=0.12, rho=-0.02)
        )
        
        leg = TradeLeg(option=option, quantity=1, direction=TradeDirection.SELL)
        
        strategy = StrategyDefinition(
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            underlying=ticker,
            legs=[leg],
            probability_of_profit=0.70,
            net_credit=Decimal('2.0'),
            max_loss=Decimal('300')
        )
        
        trade_candidate = TradeCandidate(strategy=strategy)
        
        # Create component scores
        component_scores = ComponentScores(
            model_score=model_score,
            pop_score=pop_score,
            iv_rank_score=65.0,
            momentum_z=momentum_z,
            flow_z=flow_z,
            risk_reward_score=80.0,
            liquidity_score=85.0
        )
        
        return ScoredTradeCandidate(
            trade_candidate=trade_candidate,
            component_scores=component_scores
        )
    
    return _create_scored_trade


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )
    config.addinivalue_line(
        "markers", "backtest: marks tests as backtesting tests"
    )


# Test session setup/teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset caches between tests."""
    # Clear any global caches
    if hasattr(CacheManager, '_instance'):
        CacheManager._instance = None
    
    yield
    
    # Cleanup after test
    if hasattr(CacheManager, '_instance'):
        CacheManager._instance = None


# Performance monitoring
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time = end_time - self.start_time
            memory_delta = end_memory - self.start_memory
            
            return {
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'start_memory': self.start_memory,
                'end_memory': end_memory
            }
    
    return PerformanceMonitor()


# Error injection fixtures
@pytest.fixture
def error_injector():
    """Error injection utility for testing error handling."""
    class ErrorInjector:
        def __init__(self):
            self.active_errors = {}
            
        def inject_api_error(self, client_name, method_name, error_type=Exception, message="Injected error"):
            """Inject API error."""
            key = f"{client_name}.{method_name}"
            self.active_errors[key] = (error_type, message)
            
        def clear_errors(self):
            """Clear all injected errors."""
            self.active_errors.clear()
            
        def should_error(self, client_name, method_name):
            """Check if error should be injected."""
            key = f"{client_name}.{method_name}"
            return key in self.active_errors
            
        def get_error(self, client_name, method_name):
            """Get error to inject."""
            key = f"{client_name}.{method_name}"
            if key in self.active_errors:
                error_type, message = self.active_errors[key]
                return error_type(message)
            return None
    
    return ErrorInjector()


# Test data validation
@pytest.fixture
def data_validator():
    """Data validation utility for tests."""
    class DataValidator:
        def validate_option_quote(self, quote):
            """Validate option quote data."""
            assert quote.symbol is not None
            assert quote.strike > 0
            assert quote.bid >= 0
            assert quote.ask >= quote.bid
            assert quote.volume >= 0
            assert quote.open_interest >= 0
            
        def validate_trade_candidate(self, candidate):
            """Validate trade candidate data."""
            assert candidate.strategy is not None
            assert candidate.strategy.underlying is not None
            assert len(candidate.strategy.legs) > 0
            assert candidate.strategy.probability_of_profit >= 0
            assert candidate.strategy.probability_of_profit <= 1
            
        def validate_portfolio_metrics(self, metrics):
            """Validate portfolio metrics."""
            assert 'portfolio_delta' in metrics
            assert 'portfolio_vega' in metrics
            assert 'total_risk_amount' in metrics
            assert metrics['total_risk_amount'] >= 0
    
    return DataValidator()


# Custom test decorators
def skip_if_no_api_key(api_name):
    """Skip test if API key is not available."""
    def decorator(func):
        api_key = os.getenv(f"{api_name.upper()}_API_KEY")
        return pytest.mark.skipif(
            not api_key,
            reason=f"{api_name} API key not available"
        )(func)
    return decorator


def requires_network(func):
    """Mark test as requiring network access."""
    return pytest.mark.skipif(
        os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
        reason="Network tests skipped"
    )(func)