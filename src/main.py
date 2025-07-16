"""
Main application orchestration for the Options Trading Engine.
Coordinates all components and manages the complete trading workflow.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
import json

from src.application.config.settings import ConfigManager
from src.domain.services.market_data_orchestrator import MarketDataOrchestrator
from src.domain.services.strategy_generation_service import StrategyGenerationService
from src.domain.services.scoring_engine import ScoringEngine
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.services.trade_selector import TradeSelector
from src.infrastructure.monitoring.monitoring_system import get_monitoring_system
from src.infrastructure.performance.performance_optimizer import get_performance_optimizer
from src.infrastructure.continuous_improvement.feedback_system import get_improvement_system
from src.infrastructure.error_handling import handle_errors, ApplicationError
from src.presentation.formatters.trade_formatter import TradeFormatter
from src.presentation.validation.output_validator import OutputValidator
from src.infrastructure.quality_assurance.qa_pipeline import QAPipeline


@dataclass
class ApplicationState:
    """Application state tracking."""
    status: str = "stopped"  # stopped, starting, running, stopping, error
    start_time: Optional[datetime] = None
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class OptionsEngineOrchestrator:
    """
    Main orchestrator for the Options Trading Engine.
    
    Coordinates all components and manages the complete trading workflow:
    1. Market data collection and validation
    2. Strategy generation and scoring
    3. Risk calculation and constraint validation
    4. Trade selection and ranking
    5. Output formatting and validation
    6. Performance monitoring and optimization
    7. Continuous improvement feedback collection
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Application state
        self.state = ApplicationState()
        self.shutdown_requested = False
        
        # Initialize core components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("Options Engine Orchestrator initialized")
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Infrastructure components
            self.monitoring_system = get_monitoring_system()
            self.performance_optimizer = get_performance_optimizer()
            self.improvement_system = get_improvement_system()
            
            # Core business components
            self.market_data_orchestrator = MarketDataOrchestrator()
            self.strategy_generation_service = StrategyGenerationService()
            self.scoring_engine = ScoringEngine()
            self.risk_calculator = RiskCalculator()
            self.trade_selector = TradeSelector()
            
            # Presentation components
            self.trade_formatter = TradeFormatter()
            self.output_validator = OutputValidator()
            self.qa_pipeline = QAPipeline()
            
            # Configure monitoring
            self._configure_monitoring()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise ApplicationError(f"Component initialization failed: {str(e)}")
    
    def _configure_monitoring(self):
        """Configure monitoring system with API endpoints and thresholds."""
        # Register API endpoints for monitoring
        api_endpoints = {
            'tradier': {'url': 'https://api.tradier.com/v1/', 'timeout': 30},
            'yahoo': {'url': 'https://query1.finance.yahoo.com/', 'timeout': 30},
            'fred': {'url': 'https://api.stlouisfed.org/fred/', 'timeout': 30},
            'quiver': {'url': 'https://api.quiverquant.com/', 'timeout': 30}
        }
        
        for name, config in api_endpoints.items():
            self.monitoring_system.register_api_endpoint(name, config)
        
        # Add alert handlers
        if self.config.get('alerts', {}).get('email', {}).get('enabled', False):
            from src.infrastructure.monitoring.monitoring_system import EmailAlertHandler
            email_config = self.config['alerts']['email']
            email_handler = EmailAlertHandler(email_config)
            self.monitoring_system.add_alert_handler(email_handler)
        
        if self.config.get('alerts', {}).get('slack', {}).get('enabled', False):
            from src.infrastructure.monitoring.monitoring_system import SlackAlertHandler
            slack_config = self.config['alerts']['slack']
            slack_handler = SlackAlertHandler(slack_config['webhook_url'])
            self.monitoring_system.add_alert_handler(slack_handler)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @handle_errors(operation_name="start_application")
    def start(self):
        """Start the Options Engine application."""
        if self.state.status == "running":
            self.logger.warning("Application is already running")
            return
        
        self.state.status = "starting"
        self.state.start_time = datetime.now()
        
        try:
            # Start infrastructure components
            self.monitoring_system.start_monitoring()
            self.improvement_system.start()
            
            # Initialize market data connections
            self.market_data_orchestrator.initialize_connections()
            
            self.state.status = "running"
            self.logger.info("Options Engine started successfully")
            
            # Start main execution loop
            self._run_main_loop()
            
        except Exception as e:
            self.state.status = "error"
            self.state.last_error = str(e)
            self.state.error_count += 1
            self.logger.error(f"Failed to start application: {str(e)}")
            raise
    
    def _run_main_loop(self):
        """Run the main application loop."""
        self.logger.info("Starting main execution loop")
        
        # Get execution schedule from config
        execution_schedule = self.config.get('execution', {})
        interval = execution_schedule.get('interval_minutes', 15) * 60  # Convert to seconds
        market_hours_only = execution_schedule.get('market_hours_only', True)
        
        while not self.shutdown_requested:
            try:
                # Check if we should run during market hours only
                if market_hours_only and not self._is_market_hours():
                    self.logger.info("Outside market hours, sleeping...")
                    time.sleep(300)  # Sleep 5 minutes and check again
                    continue
                
                # Execute main trading workflow
                self._execute_trading_workflow()
                
                # Update run statistics
                self.state.run_count += 1
                self.state.last_run = datetime.now()
                
                # Sleep until next execution
                self.logger.info(f"Workflow complete, sleeping for {interval} seconds...")
                time.sleep(interval)
                
            except Exception as e:
                self.state.error_count += 1
                self.state.last_error = str(e)
                self.logger.error(f"Error in main loop: {str(e)}")
                
                # Sleep before retrying
                time.sleep(60)
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    @handle_errors(operation_name="execute_trading_workflow")
    def _execute_trading_workflow(self):
        """Execute the complete trading workflow."""
        self.logger.info("Starting trading workflow execution")
        
        with self.performance_optimizer.performance_tracking("complete_workflow"):
            # Step 1: Market Data Collection
            self.logger.info("Step 1: Collecting market data...")
            market_data = self._collect_market_data()
            
            # Step 2: Strategy Generation
            self.logger.info("Step 2: Generating trading strategies...")
            strategies = self._generate_strategies(market_data)
            
            # Step 3: Strategy Scoring
            self.logger.info("Step 3: Scoring strategies...")
            scored_strategies = self._score_strategies(strategies, market_data)
            
            # Step 4: Risk Calculation
            self.logger.info("Step 4: Calculating risks...")
            risk_assessed_strategies = self._calculate_risks(scored_strategies, market_data)
            
            # Step 5: Trade Selection
            self.logger.info("Step 5: Selecting trades...")
            selected_trades = self._select_trades(risk_assessed_strategies)
            
            # Step 6: Quality Assurance
            self.logger.info("Step 6: Running quality assurance...")
            qa_validated_trades = self._run_quality_assurance(selected_trades)
            
            # Step 7: Output Generation
            self.logger.info("Step 7: Generating output...")
            formatted_output = self._generate_output(qa_validated_trades)
            
            # Step 8: Performance Feedback
            self.logger.info("Step 8: Collecting performance feedback...")
            self._collect_performance_feedback(formatted_output)
            
            self.logger.info("Trading workflow execution completed successfully")
    
    def _collect_market_data(self) -> Dict[str, Any]:
        """Collect and validate market data."""
        with self.performance_optimizer.performance_tracking("market_data_collection"):
            try:
                # Get market data from orchestrator
                market_data = self.market_data_orchestrator.collect_comprehensive_data()
                
                # Validate data quality
                data_quality_score = self.market_data_orchestrator.validate_data_quality(market_data)
                
                if data_quality_score < 0.8:  # 80% quality threshold
                    self.logger.warning(f"Low data quality score: {data_quality_score}")
                    
                    # Submit feedback about data quality
                    from src.infrastructure.continuous_improvement.feedback_system import FeedbackEntry, FeedbackType, FeedbackPriority
                    feedback = FeedbackEntry(
                        id=f"data_quality_{int(time.time())}",
                        feedback_type=FeedbackType.SYSTEM_ERROR,
                        priority=FeedbackPriority.HIGH,
                        title="Low data quality detected",
                        description=f"Data quality score: {data_quality_score:.2f}",
                        context={'quality_score': data_quality_score, 'timestamp': datetime.now().isoformat()},
                        timestamp=datetime.now(),
                        source="market_data_collection"
                    )
                    self.improvement_system.submit_feedback(feedback)
                
                return market_data
                
            except Exception as e:
                self.logger.error(f"Market data collection failed: {str(e)}")
                raise
    
    def _generate_strategies(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading strategies based on market data."""
        with self.performance_optimizer.performance_tracking("strategy_generation"):
            try:
                # Use parallel processing for strategy generation
                def generate_for_symbol(symbol_data):
                    return self.strategy_generation_service.generate_strategies(symbol_data)
                
                # Process symbols in parallel
                symbol_data_list = list(market_data.get('symbols', {}).values())
                strategy_results = self.performance_optimizer.parallel_process(
                    generate_for_symbol,
                    symbol_data_list,
                    max_workers=8
                )
                
                # Flatten results
                all_strategies = []
                for strategies in strategy_results:
                    if strategies:
                        all_strategies.extend(strategies)
                
                self.logger.info(f"Generated {len(all_strategies)} strategies")
                return all_strategies
                
            except Exception as e:
                self.logger.error(f"Strategy generation failed: {str(e)}")
                raise
    
    def _score_strategies(self, strategies: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score trading strategies."""
        with self.performance_optimizer.performance_tracking("strategy_scoring"):
            try:
                # Use parallel processing for scoring
                def score_strategy(strategy):
                    return self.scoring_engine.score_strategy(strategy, market_data)
                
                scored_strategies = self.performance_optimizer.parallel_process(
                    score_strategy,
                    strategies,
                    max_workers=8
                )
                
                # Filter out failed scoring attempts
                valid_strategies = [s for s in scored_strategies if s is not None]
                
                self.logger.info(f"Scored {len(valid_strategies)} strategies")
                return valid_strategies
                
            except Exception as e:
                self.logger.error(f"Strategy scoring failed: {str(e)}")
                raise
    
    def _calculate_risks(self, strategies: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate risks for strategies."""
        with self.performance_optimizer.performance_tracking("risk_calculation"):
            try:
                # Use parallel processing for risk calculation
                def calculate_risk(strategy):
                    return self.risk_calculator.calculate_comprehensive_risk(strategy, market_data)
                
                risk_assessed_strategies = self.performance_optimizer.parallel_process(
                    calculate_risk,
                    strategies,
                    max_workers=8
                )
                
                # Filter out failed risk calculations
                valid_strategies = [s for s in risk_assessed_strategies if s is not None]
                
                self.logger.info(f"Risk calculated for {len(valid_strategies)} strategies")
                return valid_strategies
                
            except Exception as e:
                self.logger.error(f"Risk calculation failed: {str(e)}")
                raise
    
    def _select_trades(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select final trades based on strategies."""
        with self.performance_optimizer.performance_tracking("trade_selection"):
            try:
                # Use trade selector to pick best trades
                selected_trades = self.trade_selector.select_optimal_trades(strategies)
                
                self.logger.info(f"Selected {len(selected_trades)} trades")
                return selected_trades
                
            except Exception as e:
                self.logger.error(f"Trade selection failed: {str(e)}")
                raise
    
    def _run_quality_assurance(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run quality assurance on selected trades."""
        with self.performance_optimizer.performance_tracking("quality_assurance"):
            try:
                # Run QA pipeline
                qa_results = self.qa_pipeline.run_comprehensive_qa(trades)
                
                # Filter trades that passed QA
                validated_trades = [
                    trade for trade, result in zip(trades, qa_results) 
                    if result.get('passed', False)
                ]
                
                self.logger.info(f"QA validated {len(validated_trades)} trades")
                return validated_trades
                
            except Exception as e:
                self.logger.error(f"Quality assurance failed: {str(e)}")
                raise
    
    def _generate_output(self, trades: List[Dict[str, Any]]) -> str:
        """Generate formatted output for trades."""
        with self.performance_optimizer.performance_tracking("output_generation"):
            try:
                # Format trades for output
                formatted_output = self.trade_formatter.format_trades_for_console(trades)
                
                # Validate output
                validation_result = self.output_validator.validate_output(formatted_output, trades)
                
                if not validation_result.get('valid', False):
                    self.logger.warning(f"Output validation failed: {validation_result.get('errors', [])}")
                
                self.logger.info("Output generated and validated successfully")
                return formatted_output
                
            except Exception as e:
                self.logger.error(f"Output generation failed: {str(e)}")
                raise
    
    def _collect_performance_feedback(self, output: str):
        """Collect performance feedback for continuous improvement."""
        with self.performance_optimizer.performance_tracking("feedback_collection"):
            try:
                # Get performance metrics
                perf_analysis = self.performance_optimizer.get_performance_analysis()
                
                # Submit performance feedback
                from src.infrastructure.continuous_improvement.feedback_system import FeedbackEntry, FeedbackType, FeedbackPriority
                feedback = FeedbackEntry(
                    id=f"performance_{int(time.time())}",
                    feedback_type=FeedbackType.PERFORMANCE,
                    priority=FeedbackPriority.MEDIUM,
                    title="Workflow execution performance",
                    description=f"Complete workflow execution metrics",
                    context={
                        'performance_summary': perf_analysis.get('summary', {}),
                        'output_length': len(output),
                        'timestamp': datetime.now().isoformat()
                    },
                    timestamp=datetime.now(),
                    source="workflow_execution"
                )
                self.improvement_system.submit_feedback(feedback)
                
                self.logger.info("Performance feedback collected")
                
            except Exception as e:
                self.logger.error(f"Performance feedback collection failed: {str(e)}")
    
    def stop(self):
        """Stop the Options Engine application."""
        self.logger.info("Stopping Options Engine...")
        
        self.state.status = "stopping"
        self.shutdown_requested = True
        
        try:
            # Stop infrastructure components
            self.monitoring_system.stop_monitoring()
            self.improvement_system.stop()
            
            # Cleanup performance optimizer
            self.performance_optimizer.cleanup()
            
            # Close market data connections
            self.market_data_orchestrator.close_connections()
            
            self.state.status = "stopped"
            self.logger.info("Options Engine stopped successfully")
            
        except Exception as e:
            self.state.status = "error"
            self.state.last_error = str(e)
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current application status."""
        return {
            'status': self.state.status,
            'start_time': self.state.start_time.isoformat() if self.state.start_time else None,
            'last_run': self.state.last_run.isoformat() if self.state.last_run else None,
            'run_count': self.state.run_count,
            'error_count': self.state.error_count,
            'last_error': self.state.last_error,
            'performance_metrics': self.state.performance_metrics,
            'uptime': (datetime.now() - self.state.start_time).total_seconds() if self.state.start_time else 0
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get comprehensive health check."""
        try:
            # Get system health
            system_health = self.monitoring_system.get_system_health()
            
            # Get component status
            component_status = {
                'market_data_orchestrator': self.market_data_orchestrator.get_health_status(),
                'monitoring_system': system_health.overall_status,
                'performance_optimizer': 'healthy',
                'improvement_system': 'healthy'
            }
            
            # Overall health
            overall_healthy = all(
                status in ['healthy', 'warning'] 
                for status in component_status.values()
            )
            
            return {
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'components': component_status,
                'system_health': {
                    'overall_status': system_health.overall_status,
                    'cpu_usage': system_health.cpu_usage,
                    'memory_usage': system_health.memory_usage,
                    'error_rate': system_health.error_rate,
                    'alerts_count': system_health.alerts_count
                },
                'application_state': self.get_status()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main application entry point."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/options_engine.log')
            ]
        )
        
        # Create and start orchestrator
        orchestrator = OptionsEngineOrchestrator()
        orchestrator.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()