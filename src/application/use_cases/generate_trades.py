"""
Generate trades use case for the Options Trading Engine.
Coordinates the complete trade generation workflow.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..config.settings import get_config, get_trading_config, get_constraints_config
from ...domain.services.market_data_orchestrator import MarketDataOrchestrator
from ...domain.services.strategy_generation_service import StrategyGenerationService
from ...domain.services.scoring_engine import ScoringEngine
from ...domain.services.risk_calculator import RiskCalculator
from ...domain.services.trade_selector import TradeSelector
from ...domain.services.constraint_engine import ConstraintEngine
from ...domain.services.portfolio_risk_controller import PortfolioRiskController
from ...infrastructure.quality_assurance.qa_pipeline import QAPipeline
from ...infrastructure.error_handling import handle_errors, ApplicationError
from ...infrastructure.monitoring.monitoring_system import get_monitoring_system
from ...infrastructure.performance.performance_optimizer import get_performance_optimizer


@dataclass
class TradeGenerationRequest:
    """Request for trade generation."""
    watchlist: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    max_trades: Optional[int] = None
    risk_level: Optional[str] = None
    force_refresh: bool = False
    
    def __post_init__(self):
        """Initialize default values from configuration."""
        if self.watchlist is None:
            config = get_config()
            self.watchlist = config['trading']['watchlist']
        
        if self.strategies is None:
            config = get_config()
            self.strategies = config['trading']['strategies']
        
        if self.max_trades is None:
            config = get_config()
            self.max_trades = config['execution']['max_concurrent_trades']


@dataclass
class TradeGenerationResponse:
    """Response from trade generation."""
    trades: List[Dict[str, Any]]
    summary: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    execution_time: float
    data_quality_score: float
    
    @property
    def success(self) -> bool:
        """Check if trade generation was successful."""
        return len(self.errors) == 0 and len(self.trades) > 0


class GenerateTradesUseCase:
    """
    Use case for generating trades.
    
    This orchestrates the complete trade generation workflow:
    1. Market data collection and validation
    2. Strategy generation for each symbol
    3. Risk assessment and scoring
    4. Constraint validation
    5. Trade selection and ranking
    6. Quality assurance validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize services
        self.market_data_orchestrator = MarketDataOrchestrator()
        self.strategy_generation_service = StrategyGenerationService()
        self.scoring_engine = ScoringEngine()
        self.risk_calculator = RiskCalculator()
        self.trade_selector = TradeSelector()
        self.constraint_engine = ConstraintEngine()
        self.portfolio_risk_controller = PortfolioRiskController()
        self.qa_pipeline = QAPipeline()
        
        # Infrastructure services
        self.monitoring_system = get_monitoring_system()
        self.performance_optimizer = get_performance_optimizer()
        
        # Configuration
        self.config = get_config()
        self.trading_config = get_trading_config()
        self.constraints_config = get_constraints_config()
    
    @handle_errors(operation_name="generate_trades")
    def execute(self, request: TradeGenerationRequest) -> TradeGenerationResponse:
        """
        Execute the trade generation workflow.
        
        Args:
            request: Trade generation request
            
        Returns:
            TradeGenerationResponse with generated trades
        """
        start_time = datetime.now()
        warnings = []
        errors = []
        
        try:
            with self.performance_optimizer.performance_tracking("generate_trades_workflow"):
                # Step 1: Collect market data
                self.logger.info("Step 1: Collecting market data...")
                market_data = self._collect_market_data(request.watchlist, request.force_refresh)
                
                # Validate data quality
                data_quality_score = self.market_data_orchestrator.validate_data_quality(market_data)
                if data_quality_score < 0.8:
                    warnings.append(f"Low data quality score: {data_quality_score:.2f}")
                
                # Step 2: Generate strategies
                self.logger.info("Step 2: Generating strategies...")
                strategies = self._generate_strategies(market_data, request.strategies)
                
                if not strategies:
                    errors.append("No strategies generated")
                    return self._create_error_response(errors, warnings, start_time)
                
                # Step 3: Calculate risks and score strategies
                self.logger.info("Step 3: Calculating risks and scoring...")
                scored_strategies = self._score_strategies(strategies, market_data)
                
                if not scored_strategies:
                    errors.append("No strategies passed scoring")
                    return self._create_error_response(errors, warnings, start_time)
                
                # Step 4: Apply constraints
                self.logger.info("Step 4: Applying constraints...")
                filtered_strategies = self._apply_constraints(scored_strategies, market_data)
                
                if not filtered_strategies:
                    errors.append("No strategies passed constraint filtering")
                    return self._create_error_response(errors, warnings, start_time)
                
                # Step 5: Portfolio risk validation
                self.logger.info("Step 5: Validating portfolio risk...")
                portfolio_validated_strategies = self._validate_portfolio_risk(filtered_strategies)
                
                if not portfolio_validated_strategies:
                    errors.append("No strategies passed portfolio risk validation")
                    return self._create_error_response(errors, warnings, start_time)
                
                # Step 6: Select final trades
                self.logger.info("Step 6: Selecting final trades...")
                selected_trades = self._select_final_trades(portfolio_validated_strategies, request.max_trades)
                
                if not selected_trades:
                    errors.append("No trades selected")
                    return self._create_error_response(errors, warnings, start_time)
                
                # Step 7: Quality assurance
                self.logger.info("Step 7: Running quality assurance...")
                qa_validated_trades = self._run_quality_assurance(selected_trades)
                
                if not qa_validated_trades:
                    errors.append("No trades passed quality assurance")
                    return self._create_error_response(errors, warnings, start_time)
                
                # Step 8: Generate summary
                summary = self._generate_summary(qa_validated_trades, market_data)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                self.logger.info(f"Trade generation completed successfully. Generated {len(qa_validated_trades)} trades in {execution_time:.2f}s")
                
                return TradeGenerationResponse(
                    trades=qa_validated_trades,
                    summary=summary,
                    warnings=warnings,
                    errors=errors,
                    execution_time=execution_time,
                    data_quality_score=data_quality_score
                )
        
        except Exception as e:
            errors.append(f"Trade generation failed: {str(e)}")
            return self._create_error_response(errors, warnings, start_time)
    
    def _collect_market_data(self, watchlist: List[str], force_refresh: bool) -> Dict[str, Any]:
        """Collect market data for watchlist symbols."""
        try:
            # Collect comprehensive market data
            market_data = self.market_data_orchestrator.collect_comprehensive_data(
                symbols=watchlist,
                force_refresh=force_refresh
            )
            
            # Validate we have data for all symbols
            missing_symbols = [symbol for symbol in watchlist if symbol not in market_data.get('symbols', {})]
            if missing_symbols:
                self.logger.warning(f"Missing market data for symbols: {missing_symbols}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Market data collection failed: {str(e)}")
            raise ApplicationError(f"Failed to collect market data: {str(e)}")
    
    def _generate_strategies(self, market_data: Dict[str, Any], enabled_strategies: List[str]) -> List[Dict[str, Any]]:
        """Generate strategies for all symbols."""
        try:
            all_strategies = []
            
            # Use parallel processing for strategy generation
            def generate_for_symbol(symbol_data):
                symbol, data = symbol_data
                return self.strategy_generation_service.generate_strategies(
                    symbol=symbol,
                    market_data=data,
                    enabled_strategies=enabled_strategies
                )
            
            # Process symbols in parallel
            symbol_items = list(market_data.get('symbols', {}).items())
            strategy_results = self.performance_optimizer.parallel_process(
                generate_for_symbol,
                symbol_items,
                max_workers=8
            )
            
            # Flatten results
            for strategies in strategy_results:
                if strategies:
                    all_strategies.extend(strategies)
            
            self.logger.info(f"Generated {len(all_strategies)} strategies across {len(symbol_items)} symbols")
            return all_strategies
            
        except Exception as e:
            self.logger.error(f"Strategy generation failed: {str(e)}")
            raise ApplicationError(f"Failed to generate strategies: {str(e)}")
    
    def _score_strategies(self, strategies: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score and rank strategies."""
        try:
            # Use parallel processing for scoring
            def score_strategy(strategy):
                try:
                    # Calculate risk metrics
                    risk_metrics = self.risk_calculator.calculate_comprehensive_risk(strategy, market_data)
                    strategy.update(risk_metrics)
                    
                    # Score strategy
                    score = self.scoring_engine.score_strategy(strategy, market_data)
                    strategy['model_score'] = score
                    
                    return strategy
                except Exception as e:
                    self.logger.warning(f"Failed to score strategy: {str(e)}")
                    return None
            
            scored_strategies = self.performance_optimizer.parallel_process(
                score_strategy,
                strategies,
                max_workers=8
            )
            
            # Filter out failed scoring attempts
            valid_strategies = [s for s in scored_strategies if s is not None and s.get('model_score', 0) > 0]
            
            # Sort by score (descending)
            valid_strategies.sort(key=lambda x: x.get('model_score', 0), reverse=True)
            
            self.logger.info(f"Scored {len(valid_strategies)} strategies (filtered from {len(strategies)})")
            return valid_strategies
            
        except Exception as e:
            self.logger.error(f"Strategy scoring failed: {str(e)}")
            raise ApplicationError(f"Failed to score strategies: {str(e)}")
    
    def _apply_constraints(self, strategies: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply hard constraints to filter strategies."""
        try:
            filtered_strategies = []
            
            for strategy in strategies:
                try:
                    # Check hard constraints
                    constraint_results = self.constraint_engine.validate_hard_constraints(strategy, market_data)
                    
                    if constraint_results['passed']:
                        strategy['constraint_results'] = constraint_results
                        filtered_strategies.append(strategy)
                    else:
                        # Log constraint violations for debugging
                        violations = constraint_results.get('violations', [])
                        self.logger.debug(f"Strategy {strategy.get('id', 'unknown')} failed constraints: {violations}")
                
                except Exception as e:
                    self.logger.warning(f"Constraint validation failed for strategy: {str(e)}")
                    continue
            
            self.logger.info(f"Applied constraints: {len(filtered_strategies)} strategies passed (filtered from {len(strategies)})")
            return filtered_strategies
            
        except Exception as e:
            self.logger.error(f"Constraint application failed: {str(e)}")
            raise ApplicationError(f"Failed to apply constraints: {str(e)}")
    
    def _validate_portfolio_risk(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate strategies against portfolio-level risk controls."""
        try:
            # Get current portfolio state (this would typically come from a portfolio service)
            current_portfolio = self._get_current_portfolio()
            
            # Validate portfolio risk for each strategy
            validated_strategies = []
            
            for strategy in strategies:
                try:
                    # Check portfolio risk constraints
                    risk_validation = self.portfolio_risk_controller.validate_new_position(
                        strategy,
                        current_portfolio
                    )
                    
                    if risk_validation['valid']:
                        strategy['portfolio_risk_validation'] = risk_validation
                        validated_strategies.append(strategy)
                    else:
                        # Log risk validation failures
                        reasons = risk_validation.get('reasons', [])
                        self.logger.debug(f"Strategy {strategy.get('id', 'unknown')} failed portfolio risk: {reasons}")
                
                except Exception as e:
                    self.logger.warning(f"Portfolio risk validation failed for strategy: {str(e)}")
                    continue
            
            self.logger.info(f"Portfolio risk validation: {len(validated_strategies)} strategies passed (filtered from {len(strategies)})")
            return validated_strategies
            
        except Exception as e:
            self.logger.error(f"Portfolio risk validation failed: {str(e)}")
            raise ApplicationError(f"Failed to validate portfolio risk: {str(e)}")
    
    def _select_final_trades(self, strategies: List[Dict[str, Any]], max_trades: int) -> List[Dict[str, Any]]:
        """Select final trades using trade selector."""
        try:
            # Use trade selector to pick optimal trades
            selected_trades = self.trade_selector.select_optimal_trades(
                strategies,
                max_trades=max_trades
            )
            
            self.logger.info(f"Selected {len(selected_trades)} final trades from {len(strategies)} strategies")
            return selected_trades
            
        except Exception as e:
            self.logger.error(f"Trade selection failed: {str(e)}")
            raise ApplicationError(f"Failed to select trades: {str(e)}")
    
    def _run_quality_assurance(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run quality assurance on selected trades."""
        try:
            # Run QA pipeline
            qa_results = self.qa_pipeline.run_comprehensive_qa(trades)
            
            # Filter trades that passed QA
            validated_trades = []
            for trade, result in zip(trades, qa_results):
                if result.get('passed', False):
                    trade['qa_results'] = result
                    validated_trades.append(trade)
                else:
                    # Log QA failures
                    issues = result.get('issues', [])
                    self.logger.debug(f"Trade {trade.get('id', 'unknown')} failed QA: {issues}")
            
            self.logger.info(f"Quality assurance: {len(validated_trades)} trades passed (filtered from {len(trades)})")
            return validated_trades
            
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {str(e)}")
            raise ApplicationError(f"Failed to run quality assurance: {str(e)}")
    
    def _generate_summary(self, trades: List[Dict[str, Any]], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the trade generation."""
        try:
            summary = {
                'total_trades': len(trades),
                'unique_symbols': len(set(trade.get('symbol', '') for trade in trades)),
                'strategies_used': list(set(trade.get('strategy_type', '') for trade in trades)),
                'average_score': sum(trade.get('model_score', 0) for trade in trades) / len(trades) if trades else 0,
                'total_credit': sum(trade.get('max_profit', 0) for trade in trades),
                'total_max_loss': sum(trade.get('max_loss', 0) for trade in trades),
                'average_pop': sum(trade.get('probability_of_profit', 0) for trade in trades) / len(trades) if trades else 0,
                'portfolio_allocation': sum(trade.get('capital_required', 0) for trade in trades),
                'data_quality_score': self.market_data_orchestrator.validate_data_quality(market_data),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # Add strategy breakdown
            strategy_breakdown = {}
            for trade in trades:
                strategy_type = trade.get('strategy_type', 'unknown')
                if strategy_type not in strategy_breakdown:
                    strategy_breakdown[strategy_type] = {'count': 0, 'total_score': 0}
                strategy_breakdown[strategy_type]['count'] += 1
                strategy_breakdown[strategy_type]['total_score'] += trade.get('model_score', 0)
            
            # Calculate average scores per strategy
            for strategy_type, data in strategy_breakdown.items():
                data['average_score'] = data['total_score'] / data['count'] if data['count'] > 0 else 0
            
            summary['strategy_breakdown'] = strategy_breakdown
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return {'error': f"Failed to generate summary: {str(e)}"}
    
    def _get_current_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        # This would typically fetch from a portfolio service or database
        # For now, return empty portfolio
        return {
            'positions': [],
            'total_nav': self.trading_config.nav,
            'available_capital': self.trading_config.available_capital,
            'current_allocation': 0.0,
            'aggregate_greeks': {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        }
    
    def _create_error_response(self, errors: List[str], warnings: List[str], start_time: datetime) -> TradeGenerationResponse:
        """Create error response."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return TradeGenerationResponse(
            trades=[],
            summary={'error': 'Trade generation failed', 'execution_time': execution_time},
            warnings=warnings,
            errors=errors,
            execution_time=execution_time,
            data_quality_score=0.0
        )