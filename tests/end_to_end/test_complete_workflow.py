"""
End-to-end tests for complete Options Trading Engine workflow.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal

from src.application.config.settings import ConfigManager
from src.application.use_cases.generate_trades import GenerateTradesUseCase
from src.application.use_cases.scan_market import ScanMarketUseCase
from src.data.repositories.market_repo import MarketDataRepository
from src.data.repositories.options_repo import OptionsRepository
from src.domain.services.scoring_engine import ScoringEngine
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.services.strategy_generation_service import StrategyGenerationService
from src.domain.services.constraint_engine import ConstraintEngine
from src.domain.services.trade_selector import TradeSelector
from src.infrastructure.cache import CacheManager
from src.presentation.cli.main import OptionsEngineCLI
from src.presentation.formatters.trade_formatter import TradeFormatter


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete end-to-end workflow."""
    
    @pytest.fixture
    def mock_api_clients(self):
        """Create mock API clients."""
        tradier_client = Mock()
        yahoo_client = Mock()
        fred_client = Mock()
        quiver_client = Mock()
        
        # Mock market data responses
        tradier_client.get_quote.return_value = {
            'last': 150.25,
            'bid': 150.20,
            'ask': 150.30,
            'volume': 1000000,
            'open': 149.50,
            'high': 151.00,
            'low': 149.00,
            'close': 150.25,
            'prevclose': 149.00,
            'change': 1.25,
            'change_percentage': 0.84
        }
        
        tradier_client.get_option_chain.return_value = {
            'options': [
                {
                    'symbol': 'AAPL250117P00150000',
                    'option_type': 'put',
                    'strike': 150.0,
                    'expiration': '2025-01-17',
                    'bid': 2.50,
                    'ask': 2.55,
                    'last': 2.52,
                    'volume': 1000,
                    'open_interest': 5000,
                    'greeks': {
                        'delta': -0.25,
                        'gamma': 0.05,
                        'theta': -0.08,
                        'vega': 0.12,
                        'rho': -0.02
                    },
                    'implied_volatility': 0.25
                },
                {
                    'symbol': 'AAPL250117P00145000',
                    'option_type': 'put',
                    'strike': 145.0,
                    'expiration': '2025-01-17',
                    'bid': 1.80,
                    'ask': 1.85,
                    'last': 1.82,
                    'volume': 800,
                    'open_interest': 4000,
                    'greeks': {
                        'delta': -0.18,
                        'gamma': 0.04,
                        'theta': -0.06,
                        'vega': 0.10,
                        'rho': -0.015
                    },
                    'implied_volatility': 0.23
                }
            ]
        }
        
        tradier_client.get_option_expirations.return_value = [
            '2025-01-17',
            '2025-02-21',
            '2025-03-21'
        ]
        
        yahoo_client.get_historical_data.return_value = [
            {
                'date': date.today() - timedelta(days=i),
                'open': 150.0 + i * 0.1,
                'high': 151.0 + i * 0.1,
                'low': 149.0 + i * 0.1,
                'close': 150.5 + i * 0.1,
                'volume': 1000000 + i * 1000
            }
            for i in range(252)
        ]
        
        yahoo_client.get_fundamental_data.return_value = {
            'market_cap': 2800000000000,
            'pe_ratio': 28.5,
            'dividend_yield': 0.005,
            'beta': 1.2,
            'week_52_high': 180.0,
            'week_52_low': 120.0
        }
        
        fred_client.get_series.return_value = [
            {'date': date.today() - timedelta(days=i * 30), 'value': 3.2 + i * 0.1}
            for i in range(12)
        ]
        
        quiver_client.get_sentiment_data.return_value = {
            'sentiment_score': 0.6,
            'analyst_rating': 'BUY',
            'price_target': 160.0
        }
        
        # Add health check methods
        for client in [tradier_client, yahoo_client, fred_client, quiver_client]:
            client.health_check.return_value = {'status': 'healthy'}
        
        return {
            'tradier': tradier_client,
            'yahoo': yahoo_client,
            'fred': fred_client,
            'quiver': quiver_client
        }
    
    @pytest.fixture
    def workflow_system(self, mock_api_clients):
        """Create complete workflow system with mocked dependencies."""
        # Create repositories
        market_repo = MarketDataRepository(
            tradier_client=mock_api_clients['tradier'],
            yahoo_client=mock_api_clients['yahoo'],
            fred_client=mock_api_clients['fred'],
            quiver_client=mock_api_clients['quiver'],
            cache_manager=CacheManager()
        )
        
        options_repo = OptionsRepository(
            tradier_client=mock_api_clients['tradier'],
            yahoo_client=mock_api_clients['yahoo'],
            cache_manager=CacheManager()
        )
        
        # Create domain services
        scoring_engine = ScoringEngine()
        risk_calculator = RiskCalculator()
        strategy_service = StrategyGenerationService()
        constraint_engine = ConstraintEngine()
        trade_selector = TradeSelector()
        
        # Create use cases
        generate_trades_uc = GenerateTradesUseCase(
            market_repo=market_repo,
            options_repo=options_repo,
            strategy_service=strategy_service,
            scoring_engine=scoring_engine,
            constraint_engine=constraint_engine,
            trade_selector=trade_selector,
            config_manager=ConfigManager()
        )
        
        scan_market_uc = ScanMarketUseCase(
            market_repo=market_repo,
            options_repo=options_repo,
            config_manager=ConfigManager()
        )
        
        return {
            'market_repo': market_repo,
            'options_repo': options_repo,
            'generate_trades_uc': generate_trades_uc,
            'scan_market_uc': scan_market_uc,
            'scoring_engine': scoring_engine,
            'risk_calculator': risk_calculator,
            'strategy_service': strategy_service,
            'constraint_engine': constraint_engine,
            'trade_selector': trade_selector
        }
    
    def test_complete_trade_generation_workflow(self, workflow_system):
        """Test complete trade generation workflow."""
        # Arrange
        request = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'strategies': ['cash_secured_put', 'covered_call'],
            'max_trades': 5,
            'filters': {
                'min_score': 0.7,
                'max_risk': 0.05,
                'min_dte': 7,
                'max_dte': 60
            }
        }
        
        # Act
        result = workflow_system['generate_trades_uc'].execute(request)
        
        # Assert
        assert 'trades' in result
        assert 'execution_summary' in result
        assert 'performance_metrics' in result
        
        # Verify trades structure
        trades = result['trades']
        assert isinstance(trades, list)
        
        if trades:  # If trades were generated
            trade = trades[0]
            assert 'symbol' in trade
            assert 'strategy' in trade
            assert 'score' in trade
            assert 'max_profit' in trade
            assert 'max_loss' in trade
            assert 'probability_of_profit' in trade
            assert 'days_to_expiration' in trade
            assert 'capital_required' in trade
        
        # Verify execution summary
        summary = result['execution_summary']
        assert 'total_trades_generated' in summary
        assert 'total_capital_required' in summary
        assert 'average_score' in summary
        assert 'execution_ready' in summary
        
        # Verify performance metrics
        metrics = result['performance_metrics']
        assert 'processing_time' in metrics
        assert 'api_calls_made' in metrics
        assert 'cache_hit_rate' in metrics
    
    def test_complete_market_scanning_workflow(self, workflow_system):
        """Test complete market scanning workflow."""
        # Arrange
        request = {
            'scan_type': 'momentum',
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META'],
            'filters': {
                'min_volume': 1000000,
                'min_price': 10.0,
                'max_price': 1000.0
            },
            'limit': 10
        }
        
        # Act
        result = workflow_system['scan_market_uc'].execute(request)
        
        # Assert
        assert 'scan_results' in result
        assert 'scan_summary' in result
        assert 'performance_metrics' in result
        
        # Verify scan results structure
        scan_results = result['scan_results']
        assert isinstance(scan_results, list)
        
        if scan_results:  # If results were found
            scan_result = scan_results[0]
            assert 'symbol' in scan_result
            assert 'price' in scan_result
            assert 'change_percentage' in scan_result
            assert 'volume' in scan_result
            assert 'score' in scan_result
            assert 'reason' in scan_result
        
        # Verify scan summary
        summary = result['scan_summary']
        assert 'total_symbols_scanned' in summary
        assert 'symbols_meeting_criteria' in summary
        assert 'average_score' in summary
        assert 'scan_type' in summary
        
        # Verify performance metrics
        metrics = result['performance_metrics']
        assert 'processing_time' in metrics
        assert 'api_calls_made' in metrics
        assert 'cache_hit_rate' in metrics
    
    def test_data_flow_integration(self, workflow_system):
        """Test data flow between components."""
        # Test market data retrieval
        market_data = workflow_system['market_repo'].get_market_data('AAPL')
        assert market_data is not None
        assert market_data.symbol == 'AAPL'
        assert market_data.price > 0
        assert market_data.technical_indicators is not None
        
        # Test options data retrieval
        option_chain = workflow_system['options_repo'].get_option_chain('AAPL', '2025-01-17')
        assert option_chain is not None
        assert option_chain.symbol == 'AAPL'
        assert len(option_chain.calls) > 0 or len(option_chain.puts) > 0
        
        # Test strategy generation
        strategies = workflow_system['strategy_service'].generate_strategies(
            symbol='AAPL',
            market_data=market_data,
            option_chain=option_chain,
            strategy_types=['cash_secured_put']
        )
        assert isinstance(strategies, list)
        
        # Test scoring
        if strategies:
            scored_strategies = []
            for strategy in strategies:
                scored_strategy = workflow_system['scoring_engine'].score_strategy(
                    strategy, market_data
                )
                scored_strategies.append(scored_strategy)
            
            assert len(scored_strategies) > 0
            assert all(hasattr(s, 'score') for s in scored_strategies)
    
    def test_error_handling_integration(self, workflow_system):
        """Test error handling across the system."""
        # Test with invalid symbol
        request = {
            'symbols': ['INVALID_SYMBOL'],
            'strategies': ['cash_secured_put'],
            'max_trades': 5,
            'filters': {
                'min_score': 0.7,
                'max_risk': 0.05,
                'min_dte': 7,
                'max_dte': 60
            }
        }
        
        # Should not raise exception, but handle gracefully
        result = workflow_system['generate_trades_uc'].execute(request)
        assert 'trades' in result
        assert 'execution_summary' in result
        assert 'errors' in result or 'warnings' in result
    
    def test_cache_integration(self, workflow_system):
        """Test caching integration."""
        # First call should hit API
        market_data1 = workflow_system['market_repo'].get_market_data('AAPL')
        
        # Second call should use cache
        market_data2 = workflow_system['market_repo'].get_market_data('AAPL')
        
        # Both should return data
        assert market_data1 is not None
        assert market_data2 is not None
        assert market_data1.symbol == market_data2.symbol
    
    def test_constraint_validation_integration(self, workflow_system):
        """Test constraint validation integration."""
        # Create a strategy that might violate constraints
        from src.data.models.trades import TradeCandidate, StrategyDefinition, StrategyType
        
        # Mock a high-risk strategy
        high_risk_strategy = StrategyDefinition(
            strategy_type=StrategyType.CASH_SECURED_PUT,
            underlying='AAPL',
            legs=[],
            probability_of_profit=0.4,  # Low probability
            net_credit=Decimal('100'),
            max_loss=Decimal('10000'),  # High loss
            max_profit=Decimal('100'),
            credit_to_max_loss_ratio=0.01,
            days_to_expiration=5,  # Short DTE
            margin_requirement=Decimal('10000')
        )
        
        trade_candidate = TradeCandidate(strategy=high_risk_strategy)
        
        # Test constraint validation
        violations = workflow_system['constraint_engine'].validate_trade_constraints(
            trade_candidate, {'nav': Decimal('100000')}
        )
        
        # Should identify constraint violations
        assert len(violations) > 0
    
    def test_formatting_integration(self, workflow_system):
        """Test formatting integration."""
        # Generate some trades
        request = {
            'symbols': ['AAPL'],
            'strategies': ['cash_secured_put'],
            'max_trades': 1,
            'filters': {
                'min_score': 0.0,  # Lower threshold to ensure trades
                'max_risk': 1.0,   # Higher threshold
                'min_dte': 1,
                'max_dte': 365
            }
        }
        
        result = workflow_system['generate_trades_uc'].execute(request)
        
        # Test formatting
        formatter = TradeFormatter()
        
        if result['trades']:
            # Create a mock selection result
            from src.domain.services.trade_selector import SelectionResult
            
            selection_result = SelectionResult(
                selected_trades=[],
                execution_ready=True,
                selection_summary={'total_candidates': 1, 'selected_count': 1},
                portfolio_metrics={'total_risk_amount': 1000},
                warnings=[]
            )
            
            formatted_output = formatter.format_detailed_output(selection_result)
            assert isinstance(formatted_output, str)
            assert len(formatted_output) > 0
    
    @pytest.mark.slow
    def test_performance_integration(self, workflow_system):
        """Test performance characteristics."""
        import time
        
        # Test with multiple symbols
        request = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META', 'NVDA', 'AMZN'],
            'strategies': ['cash_secured_put', 'covered_call'],
            'max_trades': 10,
            'filters': {
                'min_score': 0.7,
                'max_risk': 0.05,
                'min_dte': 7,
                'max_dte': 60
            }
        }
        
        start_time = time.time()
        result = workflow_system['generate_trades_uc'].execute(request)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # 30 seconds max
        
        # Should return performance metrics
        assert 'performance_metrics' in result
        metrics = result['performance_metrics']
        assert 'processing_time' in metrics
        assert 'api_calls_made' in metrics
        assert 'cache_hit_rate' in metrics
    
    def test_cli_integration(self, workflow_system, mock_api_clients):
        """Test CLI integration."""
        # Test CLI initialization
        with patch('src.presentation.cli.main.TradierClient') as mock_tradier, \
             patch('src.presentation.cli.main.YahooClient') as mock_yahoo, \
             patch('src.presentation.cli.main.FredClient') as mock_fred, \
             patch('src.presentation.cli.main.QuiverClient') as mock_quiver:
            
            # Configure mocks
            mock_tradier.return_value = mock_api_clients['tradier']
            mock_yahoo.return_value = mock_api_clients['yahoo']
            mock_fred.return_value = mock_api_clients['fred']
            mock_quiver.return_value = mock_api_clients['quiver']
            
            # Create CLI instance
            cli = OptionsEngineCLI()
            
            # Test CLI methods
            assert cli.config_manager is not None
            assert cli.market_repo is not None
            assert cli.options_repo is not None
            assert cli.generate_trades_uc is not None
            assert cli.scan_market_uc is not None
    
    def test_configuration_integration(self, workflow_system):
        """Test configuration integration."""
        # Test configuration loading
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert config is not None
        assert hasattr(config, 'trading_parameters')
        assert hasattr(config, 'api_keys')
        assert hasattr(config, 'constraints')
        
        # Test configuration validation
        try:
            config_manager.validate_config()
            # If no exception, validation passed
            assert True
        except Exception as e:
            # Configuration validation failed
            pytest.fail(f"Configuration validation failed: {str(e)}")
    
    def test_end_to_end_workflow_with_real_structure(self, workflow_system):
        """Test end-to-end workflow with realistic data structure."""
        # Step 1: Market Scanning
        scan_request = {
            'scan_type': 'momentum',
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'filters': {
                'min_volume': 1000000,
                'min_price': 10.0,
                'max_price': 1000.0
            },
            'limit': 3
        }
        
        scan_result = workflow_system['scan_market_uc'].execute(scan_request)
        assert 'scan_results' in scan_result
        
        # Step 2: Trade Generation for scanned symbols
        scanned_symbols = [result['symbol'] for result in scan_result['scan_results']]
        
        trade_request = {
            'symbols': scanned_symbols or ['AAPL'],  # Fallback to AAPL if no scan results
            'strategies': ['cash_secured_put', 'covered_call'],
            'max_trades': 5,
            'filters': {
                'min_score': 0.6,
                'max_risk': 0.05,
                'min_dte': 7,
                'max_dte': 60
            }
        }
        
        trade_result = workflow_system['generate_trades_uc'].execute(trade_request)
        assert 'trades' in trade_result
        assert 'execution_summary' in trade_result
        
        # Step 3: Verify workflow completion
        execution_summary = trade_result['execution_summary']
        assert 'total_trades_generated' in execution_summary
        assert 'execution_ready' in execution_summary
        
        # Step 4: Performance validation
        performance_metrics = trade_result['performance_metrics']
        assert 'processing_time' in performance_metrics
        assert performance_metrics['processing_time'] > 0
        
        # End-to-end workflow completed successfully
        assert True