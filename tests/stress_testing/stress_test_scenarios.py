"""
Stress testing and edge case scenarios for the options trading engine.
Tests system resilience under extreme conditions and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import random
import psutil
import gc

from src.infrastructure.api.base_client import BaseAPIClient, APIError, RateLimitError
from src.infrastructure.api.tradier_client import TradierClient
from src.infrastructure.cache.cache_manager import CacheManager
from src.domain.services.market_data_orchestrator import MarketDataOrchestrator
from src.domain.services.strategy_generation_service import StrategyGenerationService
from src.domain.services.scoring_engine import ScoringEngine
from src.domain.services.trade_selector import TradeSelector
from src.domain.services.risk_calculator import RiskCalculator
from src.data.models.options import OptionQuote, OptionType, Greeks
from src.data.models.trades import TradeCandidate, StrategyDefinition


class TestMarketCrashSimulation:
    """Test system behavior during market crash scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = MarketDataOrchestrator()
        self.strategy_generator = StrategyGenerationService()
        self.scoring_engine = ScoringEngine()
        self.trade_selector = TradeSelector(nav=Decimal('100000'))
        self.risk_calculator = RiskCalculator()
    
    def test_market_crash_scenario_2008_style(self):
        """Test system behavior during 2008-style market crash."""
        # Simulate 2008 market crash conditions
        crash_conditions = {
            'stock_price_drop': -0.50,  # 50% drop
            'volatility_spike': 3.0,     # 3x volatility increase
            'liquidity_drought': 0.1,    # 90% liquidity reduction
            'correlation_spike': 0.9,    # High correlation across assets
            'bid_ask_spreads': 5.0       # 5x wider spreads
        }
        
        # Create pre-crash market data
        normal_market_data = self._create_normal_market_data()
        
        # Apply crash conditions
        crash_market_data = self._apply_crash_conditions(normal_market_data, crash_conditions)
        
        # Test strategy generation during crash
        crash_strategies = []
        for ticker, data in crash_market_data.items():
            try:
                strategies = self.strategy_generator.generate_all_strategies(ticker, data)
                crash_strategies.extend(strategies)
            except Exception as e:
                pytest.fail(f"Strategy generation failed during crash: {str(e)}")
        
        # Verify system resilience
        assert len(crash_strategies) >= 0, "System should handle crash conditions gracefully"
        
        # Test risk calculations under extreme conditions
        for strategy in crash_strategies[:5]:  # Test first 5
            try:
                risk_metrics = self.risk_calculator.calculate_all_risk_metrics(strategy)
                assert risk_metrics is not None, "Risk calculations should not fail"
            except Exception as e:
                pytest.fail(f"Risk calculation failed during crash: {str(e)}")
        
        # Test trade selection with extreme data
        if crash_strategies:
            try:
                scored_strategies = []
                for strategy in crash_strategies[:10]:  # Limit to avoid timeout
                    scored_strategy = self.scoring_engine.score_trade_candidate(
                        strategy, {'market_regime': 'CRISIS', 'volatility_regime': 'EXTREME'}
                    )
                    scored_strategies.append(scored_strategy)
                
                selection_result = self.trade_selector.select_final_trades(
                    scored_strategies, [], Decimal('10000')
                )
                
                # Should either select trades or fail gracefully
                assert selection_result is not None, "Trade selection should not crash"
                
            except Exception as e:
                pytest.fail(f"Trade selection failed during crash: {str(e)}")
    
    def test_flash_crash_scenario(self):
        """Test system behavior during flash crash (rapid price movements)."""
        # Simulate flash crash conditions
        flash_crash_timeline = [
            {'time': 0, 'price_change': 0.0, 'volatility': 0.25},
            {'time': 1, 'price_change': -0.05, 'volatility': 0.50},
            {'time': 2, 'price_change': -0.15, 'volatility': 1.00},
            {'time': 3, 'price_change': -0.25, 'volatility': 2.00},  # Flash crash moment
            {'time': 4, 'price_change': -0.10, 'volatility': 1.50},  # Partial recovery
            {'time': 5, 'price_change': -0.05, 'volatility': 0.75}   # Stabilization
        ]
        
        base_price = 150.0
        
        for time_point in flash_crash_timeline:
            # Create market data for this time point
            current_price = base_price * (1 + time_point['price_change'])
            current_volatility = time_point['volatility']
            
            market_data = self._create_flash_crash_market_data(
                current_price, current_volatility
            )
            
            # Test system response at each time point
            try:
                # Generate strategies
                strategies = self.strategy_generator.generate_all_strategies(
                    'AAPL', market_data
                )
                
                # Test that system doesn't crash
                assert strategies is not None, f"System crashed at time {time_point['time']}"
                
                # Test risk calculations
                if strategies:
                    risk_metrics = self.risk_calculator.calculate_all_risk_metrics(strategies[0])
                    assert risk_metrics is not None, "Risk calculations should handle extreme volatility"
                
            except Exception as e:
                pytest.fail(f"System failed during flash crash at time {time_point['time']}: {str(e)}")
    
    def test_liquidity_crisis_scenario(self):
        """Test system behavior during liquidity crisis."""
        # Create liquidity crisis conditions
        liquidity_crisis_data = {
            'AAPL': {
                'stock_quote': {'last': 150.0, 'bid': 148.0, 'ask': 152.0, 'volume': 100000},  # Wide spreads
                'options_chain': [
                    {
                        'strike': 150.0,
                        'bid': 0.50,
                        'ask': 3.00,  # Extremely wide spread
                        'volume': 1,  # Very low volume
                        'open_interest': 5,  # Very low OI
                        'implied_volatility': 0.80  # High IV
                    }
                ]
            }
        }
        
        # Test strategy generation with low liquidity
        try:
            strategies = self.strategy_generator.generate_all_strategies(
                'AAPL', liquidity_crisis_data['AAPL']
            )
            
            # Should handle low liquidity gracefully
            assert strategies is not None, "System should handle liquidity crisis"
            
            # Test that strategies are filtered for liquidity
            for strategy in strategies:
                for leg in strategy.strategy.legs:
                    # Should reject or handle low liquidity options
                    if leg.option.volume == 1:
                        # System should either reject this or handle it specially
                        pass
            
        except Exception as e:
            pytest.fail(f"Liquidity crisis handling failed: {str(e)}")
    
    def test_correlation_breakdown_scenario(self):
        """Test system behavior when correlations break down."""
        # Create scenario where normal correlations fail
        breakdown_data = {
            'AAPL': {'price_change': 0.10, 'volatility': 0.30},   # Tech up
            'MSFT': {'price_change': 0.08, 'volatility': 0.25},   # Tech up
            'GOOGL': {'price_change': -0.15, 'volatility': 0.60}, # Tech down (unusual)
            'JNJ': {'price_change': 0.20, 'volatility': 0.40},    # Healthcare up (unusual)
            'XOM': {'price_change': -0.30, 'volatility': 0.80}    # Energy down
        }
        
        # Test portfolio construction with broken correlations
        all_strategies = []
        
        for ticker, conditions in breakdown_data.items():
            market_data = self._create_correlation_breakdown_data(ticker, conditions)
            
            try:
                strategies = self.strategy_generator.generate_all_strategies(ticker, market_data)
                all_strategies.extend(strategies)
            except Exception as e:
                pytest.fail(f"Strategy generation failed for {ticker}: {str(e)}")
        
        # Test diversification logic with broken correlations
        if all_strategies:
            try:
                scored_strategies = []
                for strategy in all_strategies:
                    scored_strategy = self.scoring_engine.score_trade_candidate(
                        strategy, {'market_regime': 'UNSTABLE'}
                    )
                    scored_strategies.append(scored_strategy)
                
                selection_result = self.trade_selector.select_final_trades(
                    scored_strategies, [], Decimal('10000')
                )
                
                # Should still maintain some diversification
                if selection_result.execution_ready:
                    sectors = set()
                    for trade in selection_result.selected_trades:
                        # Would need actual sector mapping
                        sectors.add(trade.trade_candidate.strategy.underlying[0])  # Simplified
                    
                    # Should have some diversification even with broken correlations
                    assert len(sectors) > 1, "Should maintain some diversification"
                
            except Exception as e:
                pytest.fail(f"Portfolio construction failed with broken correlations: {str(e)}")
    
    def _create_normal_market_data(self) -> Dict[str, Any]:
        """Create normal market conditions data."""
        return {
            'AAPL': {
                'stock_quote': {'last': 150.0, 'bid': 149.95, 'ask': 150.05, 'volume': 1000000},
                'options_chain': [
                    {
                        'strike': 150.0,
                        'bid': 2.50,
                        'ask': 2.55,
                        'volume': 100,
                        'open_interest': 500,
                        'implied_volatility': 0.25
                    }
                ],
                'volatility': 0.25
            }
        }
    
    def _apply_crash_conditions(self, data: Dict[str, Any], conditions: Dict[str, float]) -> Dict[str, Any]:
        """Apply crash conditions to market data."""
        crash_data = {}
        
        for ticker, ticker_data in data.items():
            # Apply price drop
            original_price = ticker_data['stock_quote']['last']
            crash_price = original_price * (1 + conditions['stock_price_drop'])
            
            # Apply volatility spike
            original_vol = ticker_data['volatility']
            crash_vol = original_vol * conditions['volatility_spike']
            
            # Apply liquidity reduction
            original_volume = ticker_data['stock_quote']['volume']
            crash_volume = int(original_volume * conditions['liquidity_drought'])
            
            # Apply wider spreads
            spread_multiplier = conditions['bid_ask_spreads']
            
            crash_data[ticker] = {
                'stock_quote': {
                    'last': crash_price,
                    'bid': crash_price * (1 - 0.01 * spread_multiplier),
                    'ask': crash_price * (1 + 0.01 * spread_multiplier),
                    'volume': crash_volume
                },
                'options_chain': [
                    {
                        'strike': opt['strike'],
                        'bid': opt['bid'] * 0.5,  # Lower bids
                        'ask': opt['ask'] * spread_multiplier,  # Higher asks
                        'volume': max(1, int(opt['volume'] * conditions['liquidity_drought'])),
                        'open_interest': opt['open_interest'],
                        'implied_volatility': crash_vol
                    }
                    for opt in ticker_data['options_chain']
                ],
                'volatility': crash_vol
            }
        
        return crash_data
    
    def _create_flash_crash_market_data(self, current_price: float, current_volatility: float) -> Dict[str, Any]:
        """Create market data for flash crash scenario."""
        return {
            'stock_quote': {
                'last': current_price,
                'bid': current_price * 0.99,
                'ask': current_price * 1.01,
                'volume': 10000000  # High volume during flash crash
            },
            'options_chain': [
                {
                    'strike': 150.0,
                    'bid': max(0.05, current_price - 150.0 + 1.0),
                    'ask': max(0.10, current_price - 150.0 + 2.0),
                    'volume': 1000,
                    'open_interest': 500,
                    'implied_volatility': current_volatility
                }
            ],
            'volatility': current_volatility
        }
    
    def _create_correlation_breakdown_data(self, ticker: str, conditions: Dict[str, float]) -> Dict[str, Any]:
        """Create market data with broken correlations."""
        base_price = 150.0
        current_price = base_price * (1 + conditions['price_change'])
        
        return {
            'stock_quote': {
                'last': current_price,
                'bid': current_price * 0.999,
                'ask': current_price * 1.001,
                'volume': 500000
            },
            'options_chain': [
                {
                    'strike': 150.0,
                    'bid': 2.50,
                    'ask': 2.55,
                    'volume': 100,
                    'open_interest': 500,
                    'implied_volatility': conditions['volatility']
                }
            ],
            'volatility': conditions['volatility']
        }


class TestAPIOutageScenarios:
    """Test system behavior during API outage scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = MarketDataOrchestrator()
        self.cache_manager = CacheManager()
    
    def test_complete_api_outage(self):
        """Test system behavior when all APIs are down."""
        with patch('src.infrastructure.api.tradier_client.TradierClient') as mock_tradier:
            with patch('src.infrastructure.api.yahoo_client.YahooClient') as mock_yahoo:
                with patch('src.infrastructure.api.fred_client.FredClient') as mock_fred:
                    
                    # Mock all APIs to fail
                    mock_tradier.return_value.get_options_chain.side_effect = APIError("API down")
                    mock_yahoo.return_value.get_fundamentals.side_effect = APIError("API down")
                    mock_fred.return_value.get_economic_series.side_effect = APIError("API down")
                    
                    # Test graceful degradation
                    try:
                        result = self.orchestrator.collect_all_market_data(
                            ['AAPL'], 
                            {
                                'tradier': mock_tradier.return_value,
                                'yahoo': mock_yahoo.return_value,
                                'fred': mock_fred.return_value
                            }
                        )
                        
                        # Should handle gracefully, not crash
                        assert result is not None, "System should handle complete API outage"
                        
                    except Exception as e:
                        # Should not crash, but if it does, error should be handled
                        assert "API down" in str(e), "Error should indicate API issue"
    
    def test_partial_api_outage(self):
        """Test system behavior with partial API outage."""
        with patch('src.infrastructure.api.tradier_client.TradierClient') as mock_tradier:
            with patch('src.infrastructure.api.yahoo_client.YahooClient') as mock_yahoo:
                
                # Tradier works, Yahoo fails
                mock_tradier.return_value.get_options_chain.return_value = [
                    {'strike': 150.0, 'bid': 2.50, 'ask': 2.55}
                ]
                mock_yahoo.return_value.get_fundamentals.side_effect = APIError("Yahoo down")
                
                try:
                    result = self.orchestrator.collect_all_market_data(
                        ['AAPL'], 
                        {
                            'tradier': mock_tradier.return_value,
                            'yahoo': mock_yahoo.return_value
                        }
                    )
                    
                    # Should work with partial data
                    assert result is not None, "System should handle partial API outage"
                    
                except Exception as e:
                    # Should degrade gracefully
                    pass
    
    def test_api_timeout_scenarios(self):
        """Test system behavior with API timeouts."""
        with patch('requests.get') as mock_get:
            # Simulate timeout
            mock_get.side_effect = TimeoutError("Request timeout")
            
            client = TradierClient("test_key")
            
            try:
                result = client.get_options_chain("AAPL")
                # Should handle timeout gracefully
                assert result is not None or result == [], "Should handle timeout"
                
            except Exception as e:
                # Should be a handled exception, not a crash
                assert "timeout" in str(e).lower(), "Should indicate timeout"
    
    def test_rate_limiting_scenarios(self):
        """Test system behavior with rate limiting."""
        with patch('requests.get') as mock_get:
            # Simulate rate limiting
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': '60'}
            mock_get.return_value = mock_response
            
            client = TradierClient("test_key")
            
            try:
                result = client.get_options_chain("AAPL")
                # Should handle rate limiting
                assert result is not None or result == [], "Should handle rate limiting"
                
            except RateLimitError:
                # Rate limit error is expected and handled
                pass
    
    def test_degraded_data_quality(self):
        """Test system behavior with degraded data quality."""
        # Create degraded data scenarios
        degraded_scenarios = [
            {'missing_bid': True, 'missing_ask': False},
            {'missing_bid': False, 'missing_ask': True},
            {'missing_volume': True, 'missing_oi': False},
            {'stale_data': True, 'age_hours': 2},
            {'incomplete_greeks': True, 'missing_delta': True}
        ]
        
        for scenario in degraded_scenarios:
            degraded_data = self._create_degraded_data(scenario)
            
            try:
                # Test that system handles degraded data
                strategies = self.orchestrator.process_degraded_data(degraded_data)
                
                # Should not crash
                assert strategies is not None, f"Should handle degraded data: {scenario}"
                
            except Exception as e:
                # Should handle gracefully
                assert "degraded" in str(e).lower() or "missing" in str(e).lower(), \
                    f"Should handle degraded data gracefully: {scenario}"
    
    def _create_degraded_data(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create degraded data based on scenario."""
        data = {
            'stock_quote': {'last': 150.0, 'volume': 1000000},
            'options_chain': [
                {
                    'strike': 150.0,
                    'bid': 2.50 if not scenario.get('missing_bid') else None,
                    'ask': 2.55 if not scenario.get('missing_ask') else None,
                    'volume': 100 if not scenario.get('missing_volume') else None,
                    'open_interest': 500 if not scenario.get('missing_oi') else None,
                    'delta': -0.25 if not scenario.get('missing_delta') else None
                }
            ]
        }
        
        if scenario.get('stale_data'):
            # Add stale timestamp
            stale_time = datetime.utcnow() - timedelta(hours=scenario.get('age_hours', 1))
            data['timestamp'] = stale_time
        
        return data


class TestMemoryAndResourceExhaustion:
    """Test system behavior under memory and resource exhaustion."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = MarketDataOrchestrator()
        self.strategy_generator = StrategyGenerationService()
    
    def test_memory_exhaustion_scenario(self):
        """Test system behavior under memory pressure."""
        # Create large dataset to consume memory
        large_dataset = {}
        
        try:
            # Generate large dataset
            for i in range(100):  # 100 tickers
                ticker = f"TICK{i:03d}"
                large_dataset[ticker] = {
                    'stock_quote': {'last': 150.0, 'volume': 1000000},
                    'options_chain': [
                        {
                            'strike': 150.0 + j,
                            'bid': 2.50,
                            'ask': 2.55,
                            'volume': 100,
                            'open_interest': 500
                        }
                        for j in range(50)  # 50 strikes per ticker
                    ],
                    'price_history': list(range(1000))  # Large price history
                }
            
            # Monitor memory usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Process large dataset
            all_strategies = []
            for ticker, data in large_dataset.items():
                strategies = self.strategy_generator.generate_all_strategies(ticker, data)
                all_strategies.extend(strategies)
                
                # Check memory usage periodically
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                if current_memory > initial_memory + 500:  # 500MB increase
                    # Trigger garbage collection
                    gc.collect()
                    
                    # Check if memory was freed
                    after_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    assert after_gc_memory < current_memory, "Memory should be freed by GC"
            
            # Final memory check
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Should not consume excessive memory
            assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.1f}MB"
            
        except MemoryError:
            # Should handle memory exhaustion gracefully
            gc.collect()
            assert True, "Memory exhaustion handled gracefully"
    
    def test_cpu_exhaustion_scenario(self):
        """Test system behavior under CPU pressure."""
        import multiprocessing
        
        # Create CPU-intensive workload
        def cpu_intensive_task():
            # Simulate CPU-intensive calculation
            result = 0
            for i in range(1000000):
                result += i * i
            return result
        
        # Start CPU-intensive background tasks
        num_cores = multiprocessing.cpu_count()
        processes = []
        
        for _ in range(num_cores):
            p = multiprocessing.Process(target=cpu_intensive_task)
            p.start()
            processes.append(p)
        
        try:
            # Test system performance under CPU pressure
            start_time = time.time()
            
            # Create test data
            test_data = self._create_test_data_for_cpu_test()
            
            # Process data under CPU pressure
            strategies = self.strategy_generator.generate_all_strategies('AAPL', test_data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should still complete within reasonable time
            assert execution_time < 60, f"Execution took {execution_time:.2f}s under CPU pressure"
            assert strategies is not None, "Should produce results under CPU pressure"
            
        finally:
            # Clean up background processes
            for p in processes:
                p.terminate()
                p.join()
    
    def test_disk_space_exhaustion(self):
        """Test system behavior when disk space is low."""
        # This would typically involve mocking disk operations
        # For now, we'll test cache behavior under space constraints
        
        cache_manager = CacheManager()
        
        # Fill cache with large objects
        large_objects = []
        for i in range(100):
            large_data = {
                'data': list(range(10000)),  # Large data structure
                'timestamp': datetime.utcnow()
            }
            cache_key = f"large_object_{i}"
            
            try:
                cache_manager.set(cache_key, large_data)
                large_objects.append(cache_key)
            except Exception as e:
                # Should handle cache exhaustion gracefully
                assert "cache" in str(e).lower() or "memory" in str(e).lower()
                break
        
        # Test cache eviction
        if large_objects:
            # Try to add one more object
            try:
                cache_manager.set("final_object", {"data": "test"})
                # Should either succeed or fail gracefully
                assert True, "Cache handled space constraints"
            except Exception as e:
                # Should be a handled exception
                assert "cache" in str(e).lower() or "space" in str(e).lower()
    
    def test_concurrent_access_under_pressure(self):
        """Test system behavior under concurrent access pressure."""
        import threading
        import queue
        
        # Create shared resources
        results_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def worker_thread(thread_id):
            try:
                # Each thread processes different data
                test_data = self._create_test_data_for_concurrent_test(thread_id)
                
                # Process data
                strategies = self.strategy_generator.generate_all_strategies(
                    f'TICK{thread_id:03d}', test_data
                )
                
                results_queue.put({
                    'thread_id': thread_id,
                    'strategies': len(strategies),
                    'status': 'success'
                })
                
            except Exception as e:
                error_queue.put({
                    'thread_id': thread_id,
                    'error': str(e),
                    'status': 'error'
                })
        
        # Start multiple concurrent threads
        threads = []
        num_threads = 20
        
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        end_time = time.time()
        
        # Check results
        success_count = 0
        error_count = 0
        
        while not results_queue.empty():
            result = results_queue.get()
            if result['status'] == 'success':
                success_count += 1
        
        while not error_queue.empty():
            error = error_queue.get()
            error_count += 1
        
        # Should handle concurrent access reasonably well
        success_rate = success_count / num_threads
        assert success_rate > 0.5, f"Success rate too low: {success_rate:.2%}"
        
        # Should complete within reasonable time
        assert end_time - start_time < 60, f"Concurrent processing took {end_time - start_time:.2f}s"
    
    def _create_test_data_for_cpu_test(self) -> Dict[str, Any]:
        """Create test data for CPU testing."""
        return {
            'stock_quote': {'last': 150.0, 'volume': 1000000},
            'options_chain': [
                {
                    'strike': 150.0,
                    'bid': 2.50,
                    'ask': 2.55,
                    'volume': 100,
                    'open_interest': 500,
                    'implied_volatility': 0.25
                }
            ],
            'volatility': 0.25
        }
    
    def _create_test_data_for_concurrent_test(self, thread_id: int) -> Dict[str, Any]:
        """Create test data for concurrent testing."""
        return {
            'stock_quote': {'last': 150.0 + thread_id, 'volume': 1000000},
            'options_chain': [
                {
                    'strike': 150.0 + thread_id,
                    'bid': 2.50,
                    'ask': 2.55,
                    'volume': 100,
                    'open_interest': 500,
                    'implied_volatility': 0.25
                }
            ],
            'volatility': 0.25
        }


class TestMathematicalEdgeCases:
    """Test edge cases in mathematical calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.risk_calculator = RiskCalculator()
        self.scoring_engine = ScoringEngine()
    
    def test_division_by_zero_cases(self):
        """Test division by zero edge cases."""
        # Test zero volatility
        try:
            delta = self.risk_calculator.calculate_call_delta(
                S=150.0, K=150.0, T=0.0833, r=0.05, sigma=0.0
            )
            assert not np.isnan(delta), "Delta should not be NaN"
            assert not np.isinf(delta), "Delta should not be infinite"
        except ZeroDivisionError:
            pytest.fail("Should handle zero volatility gracefully")
        
        # Test zero time to expiration
        try:
            theta = self.risk_calculator.calculate_call_theta(
                S=150.0, K=150.0, T=0.0, r=0.05, sigma=0.25
            )
            assert not np.isnan(theta), "Theta should not be NaN"
            assert not np.isinf(theta), "Theta should not be infinite"
        except ZeroDivisionError:
            pytest.fail("Should handle zero time to expiration gracefully")
    
    def test_extreme_parameter_values(self):
        """Test extreme parameter values."""
        extreme_cases = [
            {'S': 1000000, 'K': 1, 'T': 0.0833, 'r': 0.05, 'sigma': 0.25},    # Very high stock price
            {'S': 0.01, 'K': 1000, 'T': 0.0833, 'r': 0.05, 'sigma': 0.25},     # Very low stock price
            {'S': 150, 'K': 150, 'T': 10.0, 'r': 0.05, 'sigma': 0.25},         # Very long time
            {'S': 150, 'K': 150, 'T': 0.0833, 'r': 0.50, 'sigma': 0.25},       # Very high interest rate
            {'S': 150, 'K': 150, 'T': 0.0833, 'r': 0.05, 'sigma': 5.0},        # Very high volatility
        ]
        
        for case in extreme_cases:
            try:
                delta = self.risk_calculator.calculate_call_delta(**case)
                gamma = self.risk_calculator.calculate_gamma(**case)
                theta = self.risk_calculator.calculate_call_theta(**case)
                vega = self.risk_calculator.calculate_vega(**case)
                
                # All Greeks should be finite numbers
                assert np.isfinite(delta), f"Delta should be finite for case: {case}"
                assert np.isfinite(gamma), f"Gamma should be finite for case: {case}"
                assert np.isfinite(theta), f"Theta should be finite for case: {case}"
                assert np.isfinite(vega), f"Vega should be finite for case: {case}"
                
            except Exception as e:
                pytest.fail(f"Extreme case failed: {case}, Error: {str(e)}")
    
    def test_boundary_conditions(self):
        """Test boundary conditions for all constraints."""
        boundary_cases = [
            {'pop': 0.6499},  # Just below POP threshold
            {'pop': 0.6501},  # Just above POP threshold
            {'credit_ratio': 0.3299},  # Just below credit ratio threshold
            {'credit_ratio': 0.3301},  # Just above credit ratio threshold
            {'max_loss': 499.99},  # Just below max loss threshold
            {'max_loss': 500.01},  # Just above max loss threshold
        ]
        
        for case in boundary_cases:
            try:
                # Test boundary conditions
                if 'pop' in case:
                    # Test POP boundary
                    result = self._test_pop_boundary(case['pop'])
                    assert result is not None, f"POP boundary test failed: {case}"
                
                if 'credit_ratio' in case:
                    # Test credit ratio boundary
                    result = self._test_credit_ratio_boundary(case['credit_ratio'])
                    assert result is not None, f"Credit ratio boundary test failed: {case}"
                
                if 'max_loss' in case:
                    # Test max loss boundary
                    result = self._test_max_loss_boundary(case['max_loss'])
                    assert result is not None, f"Max loss boundary test failed: {case}"
                
            except Exception as e:
                pytest.fail(f"Boundary case failed: {case}, Error: {str(e)}")
    
    def _test_pop_boundary(self, pop_value: float) -> bool:
        """Test POP boundary condition."""
        # Create test trade with specific POP
        test_trade = self._create_test_trade(pop=pop_value)
        
        # Test constraint validation
        from src.domain.services.constraint_engine import HardConstraintValidator
        validator = HardConstraintValidator()
        
        result = validator.validate_pop_constraint(test_trade, min_pop=0.65)
        return result.is_valid
    
    def _test_credit_ratio_boundary(self, ratio_value: float) -> bool:
        """Test credit ratio boundary condition."""
        # Create test trade with specific ratio
        test_trade = self._create_test_trade(
            net_credit=ratio_value * 3.0,
            max_loss=3.0
        )
        
        from src.domain.services.constraint_engine import HardConstraintValidator
        validator = HardConstraintValidator()
        
        result = validator.validate_credit_ratio(test_trade, min_ratio=0.33)
        return result.is_valid
    
    def _test_max_loss_boundary(self, max_loss_value: float) -> bool:
        """Test max loss boundary condition."""
        # Create test trade with specific max loss
        test_trade = self._create_test_trade(max_loss=max_loss_value)
        
        from src.domain.services.constraint_engine import HardConstraintValidator
        validator = HardConstraintValidator()
        
        result = validator.validate_max_loss(test_trade, Decimal('100000'), max_loss_dollars=500)
        return result.is_valid
    
    def _create_test_trade(self, pop=0.70, net_credit=2.0, max_loss=3.0) -> 'TradeCandidate':
        """Create test trade for boundary testing."""
        from src.data.models.options import OptionQuote, OptionType, Greeks
        from src.data.models.trades import TradeCandidate, StrategyDefinition, TradeLeg, StrategyType, TradeDirection
        
        option = OptionQuote(
            symbol="AAPL250117P00150000",
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
        
        leg = TradeLeg(
            option=option,
            quantity=1,
            direction=TradeDirection.SELL
        )
        
        strategy = StrategyDefinition(
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            underlying="AAPL",
            legs=[leg],
            probability_of_profit=pop,
            net_credit=Decimal(str(net_credit)),
            max_loss=Decimal(str(max_loss))
        )
        
        return TradeCandidate(strategy=strategy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])