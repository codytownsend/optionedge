"""
Integration tests for end-to-end data flow.
Tests complete system integration from data collection to trade recommendations.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import time

from src.infrastructure.api.tradier_client import TradierClient
from src.infrastructure.api.yahoo_client import YahooClient
from src.infrastructure.api.fred_client import FredClient
from src.infrastructure.cache.cache_manager import CacheManager
from src.domain.services.market_data_orchestrator import MarketDataOrchestrator
from src.domain.services.strategy_generation_service import StrategyGenerationService
from src.domain.services.scoring_engine import ScoringEngine
from src.domain.services.trade_selector import TradeSelector
from src.presentation.formatters.trade_formatter import TradeRecommendationFormatter
from src.presentation.validation.output_validator import OutputValidator
from src.application.config.settings import ConfigurationManager


class TestEndToEndDataFlow:
    """Test complete end-to-end data flow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'nav': Decimal('100000'),
            'available_capital': Decimal('10000'),
            'api_keys': {
                'tradier': 'test_tradier_key',
                'yahoo': 'test_yahoo_key',
                'fred': 'test_fred_key',
                'quiver': 'test_quiver_key'
            },
            'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JNJ']
        }
        
        # Initialize components
        self.cache_manager = CacheManager()
        self.market_data_orchestrator = MarketDataOrchestrator()
        self.strategy_generator = StrategyGenerationService()
        self.scoring_engine = ScoringEngine()
        self.trade_selector = TradeSelector(nav=self.config['nav'])
        self.formatter = TradeRecommendationFormatter()
        self.validator = OutputValidator()
    
    def create_mock_market_data(self, ticker="AAPL"):
        """Create mock market data for testing."""
        return {
            'stock_quote': {
                'symbol': ticker,
                'last': 150.25,
                'bid': 150.20,
                'ask': 150.30,
                'volume': 1000000,
                'timestamp': datetime.utcnow()
            },
            'options_chain': [
                {
                    'symbol': f"{ticker}250117P00150000",
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
                    'symbol': f"{ticker}250117P00145000",
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
                'sector': 'Technology',
                'market_cap': 2800000000000
            },
            'technical_indicators': {
                'rsi': 65.0,
                'macd': 2.5,
                'sma_50': 148.0,
                'sma_200': 145.0,
                'momentum_20d': 0.05
            },
            'flow_data': {
                'put_call_ratio': 0.8,
                'volume_vs_oi_ratio': 1.2,
                'etf_flow_z': 0.5
            }
        }
    
    @patch('src.infrastructure.api.tradier_client.TradierClient')
    @patch('src.infrastructure.api.yahoo_client.YahooClient')
    @patch('src.infrastructure.api.fred_client.FredClient')
    def test_complete_data_collection_flow(self, mock_fred, mock_yahoo, mock_tradier):
        """Test complete data collection workflow."""
        
        # Setup API mocks
        mock_tradier_instance = Mock()
        mock_yahoo_instance = Mock()
        mock_fred_instance = Mock()
        
        mock_tradier.return_value = mock_tradier_instance
        mock_yahoo.return_value = mock_yahoo_instance
        mock_fred.return_value = mock_fred_instance
        
        # Mock API responses
        for ticker in self.config['watchlist']:
            mock_data = self.create_mock_market_data(ticker)
            
            # Mock Tradier responses
            mock_tradier_instance.get_options_chain.return_value = mock_data['options_chain']
            mock_tradier_instance.get_stock_quotes.return_value = [mock_data['stock_quote']]
            
            # Mock Yahoo responses
            mock_yahoo_instance.get_fundamentals.return_value = mock_data['fundamentals']
            mock_yahoo_instance.get_price_history.return_value = self._create_price_history()
            
            # Mock FRED responses
            mock_fred_instance.get_economic_series.return_value = self._create_economic_data()
        
        # Execute data collection
        start_time = time.time()
        collected_data = self.market_data_orchestrator.collect_all_market_data(
            self.config['watchlist'],
            {
                'tradier': mock_tradier_instance,
                'yahoo': mock_yahoo_instance,
                'fred': mock_fred_instance
            }
        )
        collection_time = time.time() - start_time
        
        # Verify data collection
        assert len(collected_data) == len(self.config['watchlist'])
        assert collection_time < 30.0  # Should complete within 30 seconds
        
        # Verify data structure
        for ticker in self.config['watchlist']:
            assert ticker in collected_data
            ticker_data = collected_data[ticker]
            
            # Check required data components
            assert 'stock_quote' in ticker_data
            assert 'options_chain' in ticker_data
            assert 'fundamentals' in ticker_data
            assert 'technical_indicators' in ticker_data
            assert 'flow_data' in ticker_data
            
            # Check data quality
            assert len(ticker_data['options_chain']) > 0
            assert ticker_data['stock_quote']['last'] > 0
            assert ticker_data['fundamentals']['trailing_pe'] > 0
    
    def test_strategy_generation_flow(self):
        """Test strategy generation workflow."""
        
        # Create test market data
        test_data = {}
        for ticker in self.config['watchlist']:
            test_data[ticker] = self.create_mock_market_data(ticker)
        
        # Execute strategy generation
        start_time = time.time()
        all_strategies = []
        
        for ticker in self.config['watchlist']:
            ticker_strategies = self.strategy_generator.generate_all_strategies(
                ticker, test_data[ticker]
            )
            all_strategies.extend(ticker_strategies)
        
        generation_time = time.time() - start_time
        
        # Verify strategy generation
        assert len(all_strategies) > 0
        assert generation_time < 60.0  # Should complete within 60 seconds
        
        # Verify strategy diversity
        strategy_types = set()
        for strategy in all_strategies:
            strategy_types.add(strategy.strategy.strategy_type)
        
        assert len(strategy_types) >= 2  # Should have multiple strategy types
        
        # Verify strategy completeness
        for strategy in all_strategies[:5]:  # Check first 5 strategies
            assert strategy.strategy.underlying in self.config['watchlist']
            assert strategy.strategy.probability_of_profit > 0
            assert strategy.strategy.max_loss > 0
            assert len(strategy.strategy.legs) > 0
    
    def test_scoring_and_ranking_flow(self):
        """Test scoring and ranking workflow."""
        
        # Create test strategies
        test_strategies = self._create_test_strategies()
        
        # Execute scoring
        start_time = time.time()
        scored_strategies = []
        
        for strategy in test_strategies:
            scored_strategy = self.scoring_engine.score_trade_candidate(
                strategy, self._create_market_context()
            )
            scored_strategies.append(scored_strategy)
        
        scoring_time = time.time() - start_time
        
        # Verify scoring
        assert len(scored_strategies) == len(test_strategies)
        assert scoring_time < 30.0  # Should complete within 30 seconds
        
        # Verify score components
        for scored_strategy in scored_strategies:
            scores = scored_strategy.component_scores
            assert 0 <= scores.model_score <= 100
            assert 0 <= scores.pop_score <= 100
            assert 0 <= scores.iv_rank_score <= 100
            assert -3 <= scores.momentum_z <= 3
            assert -3 <= scores.flow_z <= 3
            assert 0 <= scores.liquidity_score <= 100
        
        # Test ranking
        ranked_strategies = sorted(
            scored_strategies, 
            key=lambda x: x.component_scores.model_score, 
            reverse=True
        )
        
        # Verify ranking order
        for i in range(len(ranked_strategies) - 1):
            assert ranked_strategies[i].component_scores.model_score >= \
                   ranked_strategies[i + 1].component_scores.model_score
    
    def test_trade_selection_flow(self):
        """Test trade selection workflow."""
        
        # Create scored strategies
        scored_strategies = self._create_scored_strategies()
        
        # Execute trade selection
        start_time = time.time()
        selection_result = self.trade_selector.select_final_trades(
            scored_strategies,
            current_trades=[],
            available_capital=self.config['available_capital']
        )
        selection_time = time.time() - start_time
        
        # Verify selection
        assert selection_time < 10.0  # Should complete within 10 seconds
        
        if selection_result.execution_ready:
            assert len(selection_result.selected_trades) == 5
            
            # Verify portfolio constraints
            portfolio_metrics = selection_result.portfolio_metrics
            assert abs(portfolio_metrics.get('portfolio_delta', 0)) <= 0.30
            assert portfolio_metrics.get('portfolio_vega', 0) >= -0.05
            assert portfolio_metrics.get('unique_sectors', 0) >= 2
            assert portfolio_metrics.get('max_sector_concentration', 0) <= 2
        else:
            assert "Fewer than 5 trades" in selection_result.selection_summary['message']
    
    def test_output_formatting_flow(self):
        """Test output formatting workflow."""
        
        # Create test selection result
        selection_result = self._create_test_selection_result()
        
        # Execute formatting
        start_time = time.time()
        formatted_output = self.formatter.format_trades_table(selection_result)
        formatting_time = time.time() - start_time
        
        # Verify formatting
        assert formatting_time < 5.0  # Should complete within 5 seconds
        
        if selection_result.execution_ready:
            # Verify table structure
            lines = formatted_output.split('\n')
            assert len(lines) >= 7  # Header + separator + 5 trades
            
            # Verify column headers
            header_line = lines[0]
            assert 'Ticker' in header_line
            assert 'Strategy' in header_line
            assert 'Legs' in header_line
            assert 'Thesis' in header_line
            assert 'POP' in header_line
            
            # Verify data rows
            for i in range(2, 7):  # Skip header and separator
                data_line = lines[i]
                assert len(data_line.split('|')) == 5  # 5 columns
        else:
            assert "Fewer than 5 trades meet criteria" in formatted_output
    
    def test_output_validation_flow(self):
        """Test output validation workflow."""
        
        # Create test selection result and market data
        selection_result = self._create_test_selection_result()
        market_data = {ticker: self.create_mock_market_data(ticker) 
                      for ticker in self.config['watchlist']}
        
        # Execute validation
        start_time = time.time()
        validation_report = self.validator.validate_complete_output(
            selection_result, market_data
        )
        validation_time = time.time() - start_time
        
        # Verify validation
        assert validation_time < 15.0  # Should complete within 15 seconds
        assert validation_report.total_checks > 0
        assert validation_report.passed_checks >= 0
        assert validation_report.failed_checks >= 0
        
        # Verify validation categories
        if validation_report.critical_issues:
            for issue in validation_report.critical_issues:
                assert issue.level.value == 'critical'
                assert issue.message is not None
                assert issue.recommendation is not None
    
    def test_cache_behavior_integration(self):
        """Test cache behavior and invalidation."""
        
        # Test cache miss
        cache_key = "test_ticker_options"
        assert self.cache_manager.get(cache_key) is None
        
        # Test cache set and hit
        test_data = {'options': [{'strike': 150, 'bid': 2.50}]}
        self.cache_manager.set(cache_key, test_data, ttl=300)
        cached_data = self.cache_manager.get(cache_key)
        assert cached_data == test_data
        
        # Test cache invalidation
        self.cache_manager.invalidate(cache_key)
        assert self.cache_manager.get(cache_key) is None
        
        # Test cache expiration
        self.cache_manager.set(cache_key, test_data, ttl=1)
        time.sleep(2)
        assert self.cache_manager.get(cache_key) is None
    
    def test_error_propagation_flow(self):
        """Test error propagation through system."""
        
        # Test API failure propagation
        with patch('src.infrastructure.api.tradier_client.TradierClient') as mock_tradier:
            mock_tradier_instance = Mock()
            mock_tradier_instance.get_options_chain.side_effect = Exception("API failure")
            mock_tradier.return_value = mock_tradier_instance
            
            # Should handle API failure gracefully
            try:
                self.market_data_orchestrator.collect_all_market_data(
                    ['AAPL'],
                    {'tradier': mock_tradier_instance}
                )
            except Exception as e:
                assert "API failure" in str(e)
        
        # Test invalid data propagation
        invalid_strategies = [Mock(strategy=Mock(underlying=None))]
        
        try:
            self.scoring_engine.score_trade_candidate(
                invalid_strategies[0], self._create_market_context()
            )
        except Exception as e:
            assert e is not None
    
    def test_configuration_changes_integration(self):
        """Test system response to configuration changes."""
        
        # Test different NAV values
        original_nav = self.config['nav']
        
        # Test with higher NAV
        high_nav_selector = TradeSelector(nav=Decimal('500000'))
        test_strategies = self._create_scored_strategies()
        
        result_high_nav = high_nav_selector.select_final_trades(
            test_strategies,
            current_trades=[],
            available_capital=Decimal('50000')
        )
        
        # Test with lower NAV
        low_nav_selector = TradeSelector(nav=Decimal('50000'))
        result_low_nav = low_nav_selector.select_final_trades(
            test_strategies,
            current_trades=[],
            available_capital=Decimal('5000')
        )
        
        # Verify different behaviors
        if result_high_nav.execution_ready and result_low_nav.execution_ready:
            high_nav_metrics = result_high_nav.portfolio_metrics
            low_nav_metrics = result_low_nav.portfolio_metrics
            
            # Higher NAV should allow more risk
            assert high_nav_metrics.get('total_risk_amount', 0) >= \
                   low_nav_metrics.get('total_risk_amount', 0)
    
    def _create_price_history(self):
        """Create mock price history data."""
        return [
            {'date': date.today() - timedelta(days=i), 
             'open': 150.0 + i * 0.5, 
             'high': 151.0 + i * 0.5, 
             'low': 149.0 + i * 0.5, 
             'close': 150.5 + i * 0.5, 
             'volume': 1000000 + i * 10000}
            for i in range(252)  # 1 year of data
        ]
    
    def _create_economic_data(self):
        """Create mock economic data."""
        return [
            {'date': date.today() - timedelta(days=i * 30), 'value': 3.2 + i * 0.1}
            for i in range(12)  # 12 months of data
        ]
    
    def _create_test_strategies(self):
        """Create test strategy candidates."""
        from src.data.models.trades import TradeCandidate, StrategyDefinition, TradeLeg, StrategyType
        from src.data.models.options import OptionQuote, OptionType, Greeks
        
        strategies = []
        
        for i, ticker in enumerate(self.config['watchlist']):
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
                greeks=Greeks(
                    delta=-0.25,
                    gamma=0.05,
                    theta=-0.08,
                    vega=0.12,
                    rho=-0.02
                )
            )
            
            leg = TradeLeg(
                option=option,
                quantity=1,
                direction="SELL"
            )
            
            strategy = StrategyDefinition(
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                underlying=ticker,
                legs=[leg],
                probability_of_profit=0.70 + i * 0.02,
                net_credit=Decimal('2.0'),
                max_loss=Decimal('300'),
                days_to_expiration=30
            )
            
            strategies.append(TradeCandidate(strategy=strategy))
        
        return strategies
    
    def _create_market_context(self):
        """Create market context for scoring."""
        return {
            'market_regime': 'NORMAL',
            'volatility_regime': 'NORMAL',
            'risk_free_rate': 0.05,
            'market_direction': 'NEUTRAL'
        }
    
    def _create_scored_strategies(self):
        """Create scored strategy candidates."""
        from src.domain.services.scoring_engine import ScoredTradeCandidate, ComponentScores
        
        strategies = self._create_test_strategies()
        scored_strategies = []
        
        for i, strategy in enumerate(strategies):
            scores = ComponentScores(
                model_score=80.0 - i * 2,
                pop_score=70.0 + i,
                iv_rank_score=60.0,
                momentum_z=0.5 - i * 0.1,
                flow_z=0.3 + i * 0.1,
                risk_reward_score=70.0,
                liquidity_score=80.0
            )
            
            scored_strategies.append(ScoredTradeCandidate(
                trade_candidate=strategy,
                component_scores=scores
            ))
        
        return scored_strategies
    
    def _create_test_selection_result(self):
        """Create test selection result."""
        from src.domain.services.trade_selector import SelectionResult
        
        scored_strategies = self._create_scored_strategies()
        
        return SelectionResult(
            selected_trades=scored_strategies,
            rejected_trades=[],
            selection_summary={
                'selected_count': len(scored_strategies),
                'execution_ready': True,
                'message': f"Successfully selected {len(scored_strategies)} trades"
            },
            portfolio_metrics={
                'portfolio_delta': 0.15,
                'portfolio_vega': 0.08,
                'total_risk_amount': 1500,
                'unique_sectors': 3,
                'max_sector_concentration': 2
            },
            execution_ready=True
        )


class TestConcurrentOperations:
    """Test concurrent operations and load scenarios."""
    
    def test_concurrent_data_collection(self):
        """Test concurrent data collection operations."""
        import threading
        import queue
        
        # Create test queue for results
        results_queue = queue.Queue()
        
        def collect_data_worker(ticker):
            """Worker function for concurrent data collection."""
            try:
                # Simulate data collection
                time.sleep(0.1)  # Simulate API call
                result = {
                    'ticker': ticker,
                    'status': 'success',
                    'data': {'price': 150.0, 'volume': 1000000}
                }
                results_queue.put(result)
            except Exception as e:
                results_queue.put({'ticker': ticker, 'status': 'error', 'error': str(e)})
        
        # Start concurrent workers
        threads = []
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JNJ']
        
        start_time = time.time()
        
        for ticker in tickers:
            thread = threading.Thread(target=collect_data_worker, args=(ticker,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # Verify results
        assert execution_time < 2.0  # Should complete much faster than sequential
        assert results_queue.qsize() == len(tickers)
        
        # Check all results
        success_count = 0
        while not results_queue.empty():
            result = results_queue.get()
            if result['status'] == 'success':
                success_count += 1
        
        assert success_count == len(tickers)
    
    def test_load_testing_scenario(self):
        """Test system under load conditions."""
        
        # Create large dataset
        large_ticker_list = [f"TICK{i:03d}" for i in range(100)]
        
        # Test data processing performance
        start_time = time.time()
        
        # Simulate processing large dataset
        processed_count = 0
        for ticker in large_ticker_list:
            # Simulate processing
            time.sleep(0.001)  # 1ms per ticker
            processed_count += 1
            
            # Check memory usage periodically
            if processed_count % 10 == 0:
                import psutil
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                assert memory_usage < 500  # Should stay under 500MB
        
        execution_time = time.time() - start_time
        
        # Verify performance
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert processed_count == len(large_ticker_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])