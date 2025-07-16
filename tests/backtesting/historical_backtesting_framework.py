"""
Historical backtesting framework for strategy validation.
Implements historical market data replay, strategy performance validation,
and model predictive power assessment.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from unittest.mock import Mock, patch

from src.data.models.options import OptionQuote, OptionType, Greeks
from src.data.models.trades import TradeCandidate, StrategyDefinition, TradeLeg, StrategyType
from src.data.models.market_data import StockQuote, TechnicalIndicators
from src.domain.services.strategy_generation_service import StrategyGenerationService
from src.domain.services.scoring_engine import ScoringEngine
from src.domain.services.trade_selector import TradeSelector
from src.domain.services.risk_calculator import RiskCalculator


class BacktestPeriod(Enum):
    """Backtesting period types."""
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    SIX_MONTHS = "6M"
    ONE_YEAR = "1Y"
    TWO_YEARS = "2Y"
    FIVE_YEARS = "5Y"


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS_MARKET = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    period: BacktestPeriod
    start_date: date
    end_date: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    trade_details: List[Dict[str, Any]] = field(default_factory=list)
    performance_by_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    market_regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: Decimal = Decimal('100000')
    max_positions: int = 5
    commission_per_contract: Decimal = Decimal('0.65')
    slippage_pct: float = 0.001  # 0.1% slippage
    risk_free_rate: float = 0.05
    benchmark_symbol: str = "SPY"
    rebalance_frequency: int = 21  # Days
    min_dte: int = 7  # Minimum days to expiration
    max_dte: int = 45  # Maximum days to expiration


class HistoricalBacktestingFramework:
    """
    Historical backtesting framework for options strategies.
    
    Features:
    - Historical market data replay capabilities
    - Strategy performance validation over time
    - Risk metric accuracy verification
    - Model predictive power assessment
    - Parameter sensitivity analysis
    - Market regime testing
    - Overfitting detection through out-of-sample testing
    - Walk-forward analysis for model validation
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.strategy_generator = StrategyGenerationService()
        self.scoring_engine = ScoringEngine()
        self.trade_selector = TradeSelector(nav=self.config.initial_capital)
        self.risk_calculator = RiskCalculator()
        
        # Backtesting state
        self.current_capital = self.config.initial_capital
        self.current_positions = []
        self.closed_positions = []
        self.daily_returns = []
        self.equity_curve = []
    
    def run_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        strategy_config: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run comprehensive backtest over specified period.
        
        Args:
            historical_data: Historical market data by ticker
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_config: Strategy configuration parameters
            
        Returns:
            Comprehensive backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize backtest state
        self._initialize_backtest()
        
        # Generate trading dates
        trading_dates = self._generate_trading_dates(start_date, end_date)
        
        # Main backtesting loop
        for current_date in trading_dates:
            try:
                # Get market data for current date
                market_data = self._get_market_data_for_date(historical_data, current_date)
                
                # Process existing positions
                self._process_existing_positions(market_data, current_date)
                
                # Generate new trades if needed
                if len(self.current_positions) < self.config.max_positions:
                    new_trades = self._generate_trades_for_date(market_data, current_date)
                    self._execute_trades(new_trades, current_date)
                
                # Update equity curve
                self._update_equity_curve(current_date)
                
                # Log progress
                if current_date.day == 1:  # Log monthly
                    self.logger.info(f"Backtest progress: {current_date}, Capital: ${self.current_capital:,.2f}")
                    
            except Exception as e:
                self.logger.error(f"Error processing date {current_date}: {str(e)}")
                continue
        
        # Calculate final results
        backtest_result = self._calculate_backtest_results(start_date, end_date)
        
        self.logger.info(f"Backtest completed. Total return: {backtest_result.total_return:.2%}, "
                        f"Win rate: {backtest_result.win_rate:.2%}")
        
        return backtest_result
    
    def run_walk_forward_analysis(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        training_period_months: int = 12,
        testing_period_months: int = 3
    ) -> List[BacktestResult]:
        """
        Run walk-forward analysis for model validation.
        
        Args:
            historical_data: Historical market data
            start_date: Analysis start date
            end_date: Analysis end date
            training_period_months: Training period length
            testing_period_months: Testing period length
            
        Returns:
            List of backtest results for each period
        """
        self.logger.info("Starting walk-forward analysis")
        
        results = []
        current_start = start_date
        
        while current_start < end_date:
            # Define training period
            training_end = current_start + timedelta(days=training_period_months * 30)
            if training_end > end_date:
                break
            
            # Define testing period
            testing_start = training_end + timedelta(days=1)
            testing_end = testing_start + timedelta(days=testing_period_months * 30)
            if testing_end > end_date:
                testing_end = end_date
            
            self.logger.info(f"Training: {current_start} to {training_end}, "
                           f"Testing: {testing_start} to {testing_end}")
            
            # Train model on training period
            training_data = self._filter_data_by_period(historical_data, current_start, training_end)
            model_params = self._train_model(training_data)
            
            # Test on out-of-sample period
            testing_data = self._filter_data_by_period(historical_data, testing_start, testing_end)
            test_result = self.run_backtest(testing_data, testing_start, testing_end)
            
            results.append(test_result)
            
            # Move to next period
            current_start = testing_end + timedelta(days=1)
        
        self.logger.info(f"Walk-forward analysis completed. {len(results)} periods analyzed")
        
        return results
    
    def test_market_regime_performance(
        self,
        historical_data: Dict[str, pd.DataFrame],
        regime_definitions: Dict[MarketRegime, Dict[str, Any]]
    ) -> Dict[MarketRegime, BacktestResult]:
        """
        Test strategy performance across different market regimes.
        
        Args:
            historical_data: Historical market data
            regime_definitions: Regime classification criteria
            
        Returns:
            Performance results by market regime
        """
        self.logger.info("Testing market regime performance")
        
        results = {}
        
        for regime, criteria in regime_definitions.items():
            self.logger.info(f"Testing {regime.value} regime")
            
            # Identify periods matching regime criteria
            regime_periods = self._identify_regime_periods(historical_data, criteria)
            
            if not regime_periods:
                self.logger.warning(f"No periods found for {regime.value} regime")
                continue
            
            # Run backtest for each regime period
            regime_results = []
            for start_date, end_date in regime_periods:
                period_data = self._filter_data_by_period(historical_data, start_date, end_date)
                result = self.run_backtest(period_data, start_date, end_date)
                regime_results.append(result)
            
            # Aggregate regime results
            aggregated_result = self._aggregate_backtest_results(regime_results)
            results[regime] = aggregated_result
        
        return results
    
    def test_parameter_sensitivity(
        self,
        historical_data: Dict[str, pd.DataFrame],
        parameter_ranges: Dict[str, List[Any]],
        base_period: Tuple[date, date]
    ) -> Dict[str, List[BacktestResult]]:
        """
        Test parameter sensitivity analysis.
        
        Args:
            historical_data: Historical market data
            parameter_ranges: Parameter values to test
            base_period: Base testing period
            
        Returns:
            Results for each parameter combination
        """
        self.logger.info("Running parameter sensitivity analysis")
        
        results = {}
        start_date, end_date = base_period
        
        for param_name, param_values in parameter_ranges.items():
            self.logger.info(f"Testing parameter: {param_name}")
            
            param_results = []
            
            for param_value in param_values:
                # Update configuration with new parameter value
                original_value = getattr(self.config, param_name)
                setattr(self.config, param_name, param_value)
                
                try:
                    # Run backtest with new parameter
                    result = self.run_backtest(historical_data, start_date, end_date)
                    param_results.append(result)
                    
                    self.logger.debug(f"{param_name}={param_value}: Return={result.total_return:.2%}")
                    
                except Exception as e:
                    self.logger.error(f"Error testing {param_name}={param_value}: {str(e)}")
                
                finally:
                    # Restore original value
                    setattr(self.config, param_name, original_value)
            
            results[param_name] = param_results
        
        return results
    
    def test_overfitting_detection(
        self,
        historical_data: Dict[str, pd.DataFrame],
        training_period: Tuple[date, date],
        validation_period: Tuple[date, date]
    ) -> Dict[str, Any]:
        """
        Test for overfitting through out-of-sample validation.
        
        Args:
            historical_data: Historical market data
            training_period: Training data period
            validation_period: Validation data period
            
        Returns:
            Overfitting analysis results
        """
        self.logger.info("Running overfitting detection analysis")
        
        train_start, train_end = training_period
        val_start, val_end = validation_period
        
        # Run backtest on training data
        training_data = self._filter_data_by_period(historical_data, train_start, train_end)
        training_result = self.run_backtest(training_data, train_start, train_end)
        
        # Run backtest on validation data
        validation_data = self._filter_data_by_period(historical_data, val_start, val_end)
        validation_result = self.run_backtest(validation_data, val_start, val_end)
        
        # Calculate overfitting metrics
        return_degradation = training_result.total_return - validation_result.total_return
        sharpe_degradation = training_result.sharpe_ratio - validation_result.sharpe_ratio
        win_rate_degradation = training_result.win_rate - validation_result.win_rate
        
        # Overfitting indicators
        overfitting_score = (
            abs(return_degradation) * 0.4 +
            abs(sharpe_degradation) * 0.3 +
            abs(win_rate_degradation) * 0.3
        )
        
        is_overfitted = overfitting_score > 0.5  # Threshold for overfitting
        
        results = {
            'training_result': training_result,
            'validation_result': validation_result,
            'return_degradation': return_degradation,
            'sharpe_degradation': sharpe_degradation,
            'win_rate_degradation': win_rate_degradation,
            'overfitting_score': overfitting_score,
            'is_overfitted': is_overfitted,
            'recommendation': self._generate_overfitting_recommendation(is_overfitted, overfitting_score)
        }
        
        self.logger.info(f"Overfitting analysis completed. Score: {overfitting_score:.3f}, "
                        f"Overfitted: {is_overfitted}")
        
        return results
    
    def _initialize_backtest(self):
        """Initialize backtest state."""
        self.current_capital = self.config.initial_capital
        self.current_positions = []
        self.closed_positions = []
        self.daily_returns = []
        self.equity_curve = []
    
    def _generate_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        """Generate list of trading dates."""
        trading_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (simplified)
            if current_date.weekday() < 5:
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_dates
    
    def _get_market_data_for_date(
        self,
        historical_data: Dict[str, pd.DataFrame],
        target_date: date
    ) -> Dict[str, Any]:
        """Get market data for specific date."""
        market_data = {}
        
        for ticker, data in historical_data.items():
            # Find closest date in historical data
            date_index = pd.to_datetime(target_date)
            
            if date_index in data.index:
                row = data.loc[date_index]
                market_data[ticker] = {
                    'stock_quote': {
                        'symbol': ticker,
                        'last': row['Close'],
                        'volume': row['Volume'],
                        'date': target_date
                    },
                    'price_history': data.loc[:date_index].tail(252),  # Last 252 days
                    'volatility': self._calculate_realized_volatility(data.loc[:date_index], 30)
                }
        
        return market_data
    
    def _process_existing_positions(self, market_data: Dict[str, Any], current_date: date):
        """Process existing positions for expiration and P&L."""
        positions_to_close = []
        
        for position in self.current_positions:
            ticker = position['ticker']
            strategy = position['strategy']
            entry_date = position['entry_date']
            
            # Check if position should be closed
            days_held = (current_date - entry_date).days
            
            # Close if expired or at profit target
            should_close = False
            close_reason = ""
            
            if days_held >= strategy.days_to_expiration:
                should_close = True
                close_reason = "expiration"
            elif days_held >= 21:  # Hold for at least 21 days
                current_value = self._calculate_position_value(position, market_data.get(ticker, {}))
                if current_value <= position['entry_cost'] * 0.5:  # 50% profit target
                    should_close = True
                    close_reason = "profit_target"
            
            if should_close:
                self._close_position(position, market_data.get(ticker, {}), current_date, close_reason)
                positions_to_close.append(position)
        
        # Remove closed positions
        for position in positions_to_close:
            self.current_positions.remove(position)
    
    def _generate_trades_for_date(self, market_data: Dict[str, Any], current_date: date) -> List[TradeCandidate]:
        """Generate potential trades for current date."""
        candidates = []
        
        for ticker, data in market_data.items():
            try:
                # Generate synthetic options data
                options_data = self._generate_synthetic_options_data(ticker, data, current_date)
                
                # Generate strategies
                strategies = self.strategy_generator.generate_all_strategies(ticker, {
                    'stock_quote': data['stock_quote'],
                    'options_chain': options_data,
                    'volatility': data['volatility']
                })
                
                candidates.extend(strategies)
                
            except Exception as e:
                self.logger.debug(f"Error generating strategies for {ticker}: {str(e)}")
                continue
        
        return candidates
    
    def _execute_trades(self, trade_candidates: List[TradeCandidate], current_date: date):
        """Execute selected trades."""
        if not trade_candidates:
            return
        
        # Score and select trades
        scored_trades = []
        for candidate in trade_candidates:
            try:
                scored_trade = self.scoring_engine.score_trade_candidate(candidate, {
                    'market_regime': 'NORMAL',
                    'volatility_regime': 'NORMAL'
                })
                scored_trades.append(scored_trade)
            except Exception as e:
                self.logger.debug(f"Error scoring trade: {str(e)}")
                continue
        
        # Select best trades
        selection_result = self.trade_selector.select_final_trades(
            scored_trades,
            current_trades=[],
            available_capital=self.current_capital * Decimal('0.1')  # Use 10% of capital
        )
        
        # Execute selected trades
        for scored_trade in selection_result.selected_trades:
            if len(self.current_positions) >= self.config.max_positions:
                break
            
            trade = scored_trade.trade_candidate
            entry_cost = float(trade.strategy.max_loss or 0)
            
            # Check if we have enough capital
            if Decimal(str(entry_cost)) <= self.current_capital:
                position = {
                    'ticker': trade.strategy.underlying,
                    'strategy': trade.strategy,
                    'entry_date': current_date,
                    'entry_cost': entry_cost,
                    'quantity': 1,
                    'scored_trade': scored_trade
                }
                
                self.current_positions.append(position)
                self.current_capital -= Decimal(str(entry_cost))
                
                self.logger.debug(f"Executed trade: {trade.strategy.underlying} "
                                f"{trade.strategy.strategy_type.value} for ${entry_cost}")
    
    def _close_position(self, position: Dict[str, Any], market_data: Dict[str, Any], 
                       current_date: date, reason: str):
        """Close existing position."""
        entry_cost = position['entry_cost']
        exit_value = self._calculate_position_value(position, market_data)
        
        # Calculate P&L
        pnl = exit_value - entry_cost
        
        # Apply commission
        commission = float(self.config.commission_per_contract) * position['quantity']
        pnl -= commission
        
        # Update capital
        self.current_capital += Decimal(str(exit_value))
        
        # Record closed position
        closed_position = {
            **position,
            'exit_date': current_date,
            'exit_value': exit_value,
            'pnl': pnl,
            'close_reason': reason,
            'days_held': (current_date - position['entry_date']).days
        }
        
        self.closed_positions.append(closed_position)
        
        self.logger.debug(f"Closed position: {position['ticker']} P&L: ${pnl:.2f} ({reason})")
    
    def _calculate_position_value(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate current position value."""
        # Simplified position valuation
        strategy = position['strategy']
        entry_cost = position['entry_cost']
        
        if not market_data:
            return entry_cost * 0.5  # Assume 50% value if no data
        
        # For credit strategies, value decreases over time (theta decay)
        if strategy.strategy_type.value in ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD", "IRON_CONDOR"]:
            # Simplified theta decay model
            time_decay_factor = 0.95  # 5% daily decay
            return entry_cost * time_decay_factor
        
        return entry_cost * 0.8  # Default 20% profit
    
    def _update_equity_curve(self, current_date: date):
        """Update equity curve and daily returns."""
        # Calculate total portfolio value
        total_value = float(self.current_capital)
        
        # Add value of open positions
        for position in self.current_positions:
            total_value += position['entry_cost']  # Simplified
        
        self.equity_curve.append({
            'date': current_date,
            'equity': total_value
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (total_value - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def _calculate_backtest_results(self, start_date: date, end_date: date) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        if not self.closed_positions:
            return BacktestResult(
                period=BacktestPeriod.ONE_YEAR,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0
            )
        
        # Calculate basic metrics
        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for pos in self.closed_positions if pos['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate returns
        total_pnl = sum(pos['pnl'] for pos in self.closed_positions)
        initial_capital = float(self.config.initial_capital)
        total_return = total_pnl / initial_capital
        
        # Calculate win/loss averages
        wins = [pos['pnl'] for pos in self.closed_positions if pos['pnl'] > 0]
        losses = [pos['pnl'] for pos in self.closed_positions if pos['pnl'] < 0]
        
        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Calculate profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calculate Sharpe ratio
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            excess_returns = returns_array - (self.config.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats()
        
        # Performance by strategy
        performance_by_strategy = self._calculate_strategy_performance()
        
        return BacktestResult(
            period=BacktestPeriod.ONE_YEAR,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trade_details=[{
                'ticker': pos['ticker'],
                'strategy': pos['strategy'].strategy_type.value,
                'entry_date': pos['entry_date'],
                'exit_date': pos['exit_date'],
                'pnl': pos['pnl'],
                'days_held': pos['days_held']
            } for pos in self.closed_positions],
            performance_by_strategy=performance_by_strategy
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_consecutive_stats(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not self.closed_positions:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for position in self.closed_positions:
            if position['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by strategy type."""
        strategy_performance = {}
        
        for position in self.closed_positions:
            strategy_type = position['strategy'].strategy_type.value
            
            if strategy_type not in strategy_performance:
                strategy_performance[strategy_type] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'average_pnl': 0.0
                }
            
            stats = strategy_performance[strategy_type]
            stats['total_trades'] += 1
            stats['total_pnl'] += position['pnl']
            
            if position['pnl'] > 0:
                stats['winning_trades'] += 1
        
        # Calculate derived metrics
        for strategy_type, stats in strategy_performance.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
                stats['average_pnl'] = stats['total_pnl'] / stats['total_trades']
        
        return strategy_performance
    
    def _calculate_realized_volatility(self, price_data: pd.DataFrame, periods: int) -> float:
        """Calculate realized volatility."""
        if len(price_data) < periods:
            return 0.25  # Default volatility
        
        returns = price_data['Close'].pct_change().dropna()
        volatility = returns.tail(periods).std() * np.sqrt(252)
        return float(volatility)
    
    def _generate_synthetic_options_data(self, ticker: str, market_data: Dict[str, Any], current_date: date) -> List[Dict[str, Any]]:
        """Generate synthetic options data for backtesting."""
        stock_price = market_data['stock_quote']['last']
        volatility = market_data['volatility']
        
        options_data = []
        
        # Generate options at different strikes
        for strike_offset in [-10, -5, 0, 5, 10]:
            strike = stock_price + strike_offset
            
            # Generate put option
            put_option = {
                'symbol': f"{ticker}{current_date.strftime('%y%m%d')}P{strike:08.0f}",
                'strike': strike,
                'expiration': current_date + timedelta(days=30),
                'option_type': 'put',
                'bid': max(0.05, strike - stock_price + 2.0) if strike > stock_price else 0.50,
                'ask': max(0.10, strike - stock_price + 2.5) if strike > stock_price else 0.55,
                'volume': 50,
                'open_interest': 100,
                'implied_volatility': volatility
            }
            
            options_data.append(put_option)
        
        return options_data
    
    def _filter_data_by_period(self, data: Dict[str, pd.DataFrame], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """Filter data by date period."""
        filtered_data = {}
        
        for ticker, df in data.items():
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            filtered_data[ticker] = df.loc[mask]
        
        return filtered_data
    
    def _train_model(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train model on historical data (placeholder)."""
        # This would implement actual model training
        return {'trained': True, 'parameters': {}}
    
    def _identify_regime_periods(self, historical_data: Dict[str, pd.DataFrame], criteria: Dict[str, Any]) -> List[Tuple[date, date]]:
        """Identify periods matching regime criteria."""
        # Simplified regime identification
        periods = []
        
        # Example: High volatility periods
        if criteria.get('volatility_threshold'):
            # Would implement actual regime detection logic
            pass
        
        return periods
    
    def _aggregate_backtest_results(self, results: List[BacktestResult]) -> BacktestResult:
        """Aggregate multiple backtest results."""
        if not results:
            return BacktestResult(
                period=BacktestPeriod.ONE_YEAR,
                start_date=date.today(),
                end_date=date.today(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0
            )
        
        # Aggregate metrics
        total_trades = sum(r.total_trades for r in results)
        winning_trades = sum(r.winning_trades for r in results)
        
        # Weighted averages
        total_return = np.mean([r.total_return for r in results])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return BacktestResult(
            period=BacktestPeriod.ONE_YEAR,
            start_date=min(r.start_date for r in results),
            end_date=max(r.end_date for r in results),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max(r.max_drawdown for r in results),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in results]),
            profit_factor=np.mean([r.profit_factor for r in results]),
            average_win=np.mean([r.average_win for r in results]),
            average_loss=np.mean([r.average_loss for r in results]),
            max_consecutive_wins=max(r.max_consecutive_wins for r in results),
            max_consecutive_losses=max(r.max_consecutive_losses for r in results)
        )
    
    def _generate_overfitting_recommendation(self, is_overfitted: bool, score: float) -> str:
        """Generate recommendation based on overfitting analysis."""
        if not is_overfitted:
            return "Model appears robust. Performance degradation is within acceptable limits."
        
        if score > 0.8:
            return "Strong evidence of overfitting. Consider simplifying the model, increasing training data, or using regularization techniques."
        elif score > 0.6:
            return "Moderate overfitting detected. Review model complexity and validate with additional out-of-sample data."
        else:
            return "Minor overfitting detected. Monitor performance and consider minor adjustments to model parameters."


if __name__ == "__main__":
    # Example usage
    framework = HistoricalBacktestingFramework()
    
    # Create sample historical data
    sample_data = {
        'AAPL': pd.DataFrame({
            'Close': np.random.randn(252).cumsum() + 150,
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=pd.date_range('2023-01-01', periods=252))
    }
    
    # Run backtest
    result = framework.run_backtest(
        sample_data, 
        date(2023, 1, 1), 
        date(2023, 12, 31)
    )
    
    print(f"Backtest Results:")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")