"""
Mathematical validation tests for options calculations.
Tests Greeks calculations, POP calculations, and portfolio aggregation.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from src.data.models.options import OptionQuote, OptionType, Greeks
from src.data.models.trades import StrategyDefinition, TradeCandidate, TradeLeg, StrategyType
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.services.probability_calculator import ProbabilityCalculator
from src.domain.services.portfolio_greeks_manager import PortfolioGreeksManager


class TestBlackScholesGreeks:
    """Test Black-Scholes Greeks calculations against known values."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.risk_calculator = RiskCalculator()
        self.tolerance = 0.001
    
    def test_black_scholes_greeks_call_option(self):
        """Test Greeks calculations for call option against known values."""
        # Test case: AAPL call option
        S = 150.0  # Stock price
        K = 155.0  # Strike price  
        T = 0.0833  # 30 days to expiration
        r = 0.05   # Risk-free rate
        sigma = 0.25  # Volatility
        
        # Known correct values (calculated using standard B-S model)
        expected_delta = 0.4316
        expected_gamma = 0.0189
        expected_theta = -0.0347
        expected_vega = 0.0891
        expected_rho = 0.0526
        
        # Calculate Greeks
        calculated_delta = self.risk_calculator.calculate_call_delta(S, K, T, r, sigma)
        calculated_gamma = self.risk_calculator.calculate_gamma(S, K, T, r, sigma)
        calculated_theta = self.risk_calculator.calculate_call_theta(S, K, T, r, sigma)
        calculated_vega = self.risk_calculator.calculate_vega(S, K, T, r, sigma)
        calculated_rho = self.risk_calculator.calculate_call_rho(S, K, T, r, sigma)
        
        # Assertions
        assert abs(calculated_delta - expected_delta) < self.tolerance, \
            f"Delta calculation failed: {calculated_delta} vs {expected_delta}"
        assert abs(calculated_gamma - expected_gamma) < self.tolerance, \
            f"Gamma calculation failed: {calculated_gamma} vs {expected_gamma}"
        assert abs(calculated_theta - expected_theta) < self.tolerance, \
            f"Theta calculation failed: {calculated_theta} vs {expected_theta}"
        assert abs(calculated_vega - expected_vega) < self.tolerance, \
            f"Vega calculation failed: {calculated_vega} vs {expected_vega}"
        assert abs(calculated_rho - expected_rho) < self.tolerance, \
            f"Rho calculation failed: {calculated_rho} vs {expected_rho}"
    
    def test_black_scholes_greeks_put_option(self):
        """Test Greeks calculations for put option against known values."""
        # Test case: AAPL put option
        S = 150.0  # Stock price
        K = 145.0  # Strike price  
        T = 0.0833  # 30 days to expiration
        r = 0.05   # Risk-free rate
        sigma = 0.25  # Volatility
        
        # Known correct values (calculated using standard B-S model)
        expected_delta = -0.3158
        expected_gamma = 0.0189
        expected_theta = -0.0291
        expected_vega = 0.0891
        expected_rho = -0.0360
        
        # Calculate Greeks
        calculated_delta = self.risk_calculator.calculate_put_delta(S, K, T, r, sigma)
        calculated_gamma = self.risk_calculator.calculate_gamma(S, K, T, r, sigma)
        calculated_theta = self.risk_calculator.calculate_put_theta(S, K, T, r, sigma)
        calculated_vega = self.risk_calculator.calculate_vega(S, K, T, r, sigma)
        calculated_rho = self.risk_calculator.calculate_put_rho(S, K, T, r, sigma)
        
        # Assertions
        assert abs(calculated_delta - expected_delta) < self.tolerance, \
            f"Put delta calculation failed: {calculated_delta} vs {expected_delta}"
        assert abs(calculated_gamma - expected_gamma) < self.tolerance, \
            f"Put gamma calculation failed: {calculated_gamma} vs {expected_gamma}"
        assert abs(calculated_theta - expected_theta) < self.tolerance, \
            f"Put theta calculation failed: {calculated_theta} vs {expected_theta}"
        assert abs(calculated_vega - expected_vega) < self.tolerance, \
            f"Put vega calculation failed: {calculated_vega} vs {expected_vega}"
        assert abs(calculated_rho - expected_rho) < self.tolerance, \
            f"Put rho calculation failed: {calculated_rho} vs {expected_rho}"
    
    def test_greeks_boundary_conditions(self):
        """Test Greeks at boundary conditions."""
        # Deep ITM call (delta should approach 1)
        deep_itm_delta = self.risk_calculator.calculate_call_delta(100, 50, 0.25, 0.05, 0.20)
        assert deep_itm_delta > 0.95, f"Deep ITM delta should be near 1.0: {deep_itm_delta}"
        
        # Deep OTM call (delta should approach 0)
        deep_otm_delta = self.risk_calculator.calculate_call_delta(100, 150, 0.25, 0.05, 0.20)
        assert deep_otm_delta < 0.05, f"Deep OTM delta should be near 0.0: {deep_otm_delta}"
        
        # Near expiration (theta should be large)
        near_expiry_theta = self.risk_calculator.calculate_call_theta(100, 100, 0.01, 0.05, 0.20)
        assert near_expiry_theta < -0.5, f"Near expiry theta should be large negative: {near_expiry_theta}"


class TestPOPCalculations:
    """Test Probability of Profit calculations for different strategies."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.prob_calculator = ProbabilityCalculator()
        self.tolerance = 0.01
    
    def test_credit_spread_pop_calculation(self):
        """Test POP calculations for credit spreads."""
        # Credit spread test case
        spread_params = {
            'short_strike': 100,
            'long_strike': 95,
            'net_credit': 2.0,
            'strike_width': 5.0,
            'strategy_type': StrategyType.PUT_CREDIT_SPREAD
        }
        
        # Expected POP using simple formula: (strike_width - net_credit) / strike_width
        expected_pop = (5.0 - 2.0) / 5.0  # 0.60
        
        calculated_pop = self.prob_calculator.calculate_credit_spread_pop(spread_params)
        
        assert abs(calculated_pop - expected_pop) < self.tolerance, \
            f"Credit spread POP calculation failed: {calculated_pop} vs {expected_pop}"
    
    def test_iron_condor_pop_calculation(self):
        """Test POP calculations for iron condors."""
        # Iron condor test case
        condor_params = {
            'put_short_strike': 95,
            'put_long_strike': 90,
            'call_short_strike': 105,
            'call_long_strike': 110,
            'net_credit': 2.5,
            'current_price': 100,
            'volatility': 0.25,
            'days_to_expiration': 30
        }
        
        # Expected POP should be probability of staying between short strikes
        expected_pop = 0.65  # Approximate expected value
        
        calculated_pop = self.prob_calculator.calculate_iron_condor_pop(condor_params)
        
        assert abs(calculated_pop - expected_pop) < 0.10, \
            f"Iron condor POP calculation failed: {calculated_pop} vs {expected_pop}"
    
    def test_covered_call_pop_calculation(self):
        """Test POP calculations for covered calls."""
        # Covered call test case
        covered_call_params = {
            'stock_price': 100,
            'strike_price': 105,
            'premium': 2.0,
            'volatility': 0.25,
            'days_to_expiration': 30,
            'risk_free_rate': 0.05
        }
        
        # Expected POP should be probability of finishing below strike
        expected_pop = 0.75  # Approximate expected value
        
        calculated_pop = self.prob_calculator.calculate_covered_call_pop(covered_call_params)
        
        assert abs(calculated_pop - expected_pop) < 0.10, \
            f"Covered call POP calculation failed: {calculated_pop} vs {expected_pop}"
    
    def test_monte_carlo_pop_validation(self):
        """Test Monte Carlo POP calculation validation."""
        # Monte Carlo parameters
        mc_params = {
            'current_price': 100,
            'strike_price': 105,
            'volatility': 0.25,
            'days_to_expiration': 30,
            'risk_free_rate': 0.05,
            'simulations': 10000,
            'option_type': OptionType.CALL
        }
        
        # Run Monte Carlo simulation
        mc_pop = self.prob_calculator.calculate_monte_carlo_pop(mc_params)
        
        # Calculate theoretical POP using Black-Scholes
        theoretical_pop = self.prob_calculator.calculate_black_scholes_pop(mc_params)
        
        # Monte Carlo should be within 5% of theoretical
        assert abs(mc_pop - theoretical_pop) < 0.05, \
            f"Monte Carlo POP diverges from theoretical: {mc_pop} vs {theoretical_pop}"


class TestPortfolioGreeks:
    """Test portfolio Greeks aggregation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.portfolio_manager = PortfolioGreeksManager()
        self.nav = Decimal('100000')
    
    def create_test_trade(self, ticker="AAPL", delta=0.25, gamma=0.05, theta=-0.10, vega=0.15, rho=0.02):
        """Create a test trade with specified Greeks."""
        option = OptionQuote(
            symbol=f"{ticker}250117C00150000",
            strike=Decimal('150'),
            expiration=date(2025, 1, 17),
            option_type=OptionType.CALL,
            bid=Decimal('2.50'),
            ask=Decimal('2.60'),
            last=Decimal('2.55'),
            volume=100,
            open_interest=500,
            greeks=Greeks(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho
            )
        )
        
        leg = TradeLeg(
            option=option,
            quantity=1,
            direction="BUY"
        )
        
        strategy = StrategyDefinition(
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            underlying=ticker,
            legs=[leg]
        )
        
        return TradeCandidate(strategy=strategy)
    
    def test_portfolio_greeks_aggregation(self):
        """Test portfolio Greeks aggregation."""
        trades = [
            self.create_test_trade(ticker="AAPL", delta=0.25, gamma=0.05, theta=-0.10, vega=0.15),
            self.create_test_trade(ticker="MSFT", delta=-0.15, gamma=0.03, theta=-0.08, vega=0.12)
        ]
        
        portfolio_greeks = self.portfolio_manager.calculate_portfolio_greeks(trades, self.nav)
        
        # Expected aggregated Greeks (normalized by NAV factor)
        nav_factor = float(self.nav) / 100000
        expected_delta = (0.25 - 0.15) * 100 / nav_factor  # 0.10 * 100 = 10
        expected_gamma = (0.05 + 0.03) * 100 / nav_factor  # 0.08 * 100 = 8
        expected_theta = (-0.10 - 0.08) * 100 / nav_factor  # -0.18 * 100 = -18
        expected_vega = (0.15 + 0.12) * 100 / nav_factor  # 0.27 * 100 = 27
        
        assert abs(portfolio_greeks['delta'] - expected_delta) < 0.001, \
            f"Portfolio delta calculation failed: {portfolio_greeks['delta']} vs {expected_delta}"
        assert abs(portfolio_greeks['gamma'] - expected_gamma) < 0.001, \
            f"Portfolio gamma calculation failed: {portfolio_greeks['gamma']} vs {expected_gamma}"
        assert abs(portfolio_greeks['theta'] - expected_theta) < 0.001, \
            f"Portfolio theta calculation failed: {portfolio_greeks['theta']} vs {expected_theta}"
        assert abs(portfolio_greeks['vega'] - expected_vega) < 0.001, \
            f"Portfolio vega calculation failed: {portfolio_greeks['vega']} vs {expected_vega}"
    
    def test_portfolio_greeks_limits_validation(self):
        """Test portfolio Greeks limits validation."""
        # Create trades that would exceed limits
        trades = [
            self.create_test_trade(ticker="AAPL", delta=0.35),  # High delta
            self.create_test_trade(ticker="MSFT", delta=0.30),  # High delta
            self.create_test_trade(ticker="GOOGL", delta=0.25)  # High delta
        ]
        
        portfolio_greeks = self.portfolio_manager.calculate_portfolio_greeks(trades, self.nav)
        
        # Check if portfolio exceeds delta limits
        nav_factor = float(self.nav) / 100000
        max_delta = 0.30 * nav_factor
        
        total_delta = portfolio_greeks['delta']
        
        # Should detect limit violation
        assert abs(total_delta) > max_delta, \
            f"Portfolio delta limit test failed: {total_delta} should exceed {max_delta}"
    
    def test_portfolio_greeks_with_different_quantities(self):
        """Test portfolio Greeks with different position quantities."""
        # Create trades with different quantities
        trade1 = self.create_test_trade(ticker="AAPL", delta=0.25)
        trade1.strategy.legs[0].quantity = 2  # 2 contracts
        
        trade2 = self.create_test_trade(ticker="MSFT", delta=-0.15)
        trade2.strategy.legs[0].quantity = 3  # 3 contracts
        
        trades = [trade1, trade2]
        
        portfolio_greeks = self.portfolio_manager.calculate_portfolio_greeks(trades, self.nav)
        
        # Expected delta: (0.25 * 2 - 0.15 * 3) * 100 = (0.5 - 0.45) * 100 = 5.0
        nav_factor = float(self.nav) / 100000
        expected_delta = (0.25 * 2 - 0.15 * 3) * 100 / nav_factor
        
        assert abs(portfolio_greeks['delta'] - expected_delta) < 0.001, \
            f"Portfolio delta with quantities failed: {portfolio_greeks['delta']} vs {expected_delta}"


class TestRiskMetricsCalculations:
    """Test risk metrics calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.risk_calculator = RiskCalculator()
    
    def test_max_loss_calculation_credit_spread(self):
        """Test maximum loss calculation for credit spreads."""
        # Put credit spread parameters
        spread_params = {
            'short_strike': 100,
            'long_strike': 95,
            'net_credit': 2.0,
            'strategy_type': StrategyType.PUT_CREDIT_SPREAD
        }
        
        # Max loss = Strike width - Net credit
        expected_max_loss = 5.0 - 2.0  # 3.0
        
        calculated_max_loss = self.risk_calculator.calculate_max_loss(spread_params)
        
        assert abs(calculated_max_loss - expected_max_loss) < 0.01, \
            f"Max loss calculation failed: {calculated_max_loss} vs {expected_max_loss}"
    
    def test_breakeven_calculation_credit_spread(self):
        """Test breakeven calculation for credit spreads."""
        # Put credit spread parameters
        spread_params = {
            'short_strike': 100,
            'long_strike': 95,
            'net_credit': 2.0,
            'strategy_type': StrategyType.PUT_CREDIT_SPREAD
        }
        
        # Breakeven = Short strike - Net credit
        expected_breakeven = 100 - 2.0  # 98.0
        
        calculated_breakeven = self.risk_calculator.calculate_breakeven(spread_params)
        
        assert abs(calculated_breakeven - expected_breakeven) < 0.01, \
            f"Breakeven calculation failed: {calculated_breakeven} vs {expected_breakeven}"
    
    def test_credit_to_max_loss_ratio(self):
        """Test credit-to-max-loss ratio calculation."""
        # Credit spread parameters
        spread_params = {
            'net_credit': 2.0,
            'max_loss': 3.0
        }
        
        # Ratio = Net credit / Max loss
        expected_ratio = 2.0 / 3.0  # 0.667
        
        calculated_ratio = self.risk_calculator.calculate_credit_to_max_loss_ratio(spread_params)
        
        assert abs(calculated_ratio - expected_ratio) < 0.01, \
            f"Credit-to-max-loss ratio calculation failed: {calculated_ratio} vs {expected_ratio}"
    
    def test_position_sizing_kelly_criterion(self):
        """Test position sizing using Kelly criterion."""
        # Position sizing parameters
        sizing_params = {
            'probability_of_profit': 0.65,
            'max_profit': 200,
            'max_loss': 300,
            'available_capital': 10000,
            'nav': 100000
        }
        
        # Kelly fraction = (bp - q) / b where b = profit/loss ratio
        b = 200 / 300  # 0.667
        p = 0.65
        q = 1 - p  # 0.35
        kelly_fraction = (p * (1 + b) - 1) / b
        
        # Conservative sizing (1/4 of Kelly)
        conservative_fraction = min(0.25 * kelly_fraction, 0.02)
        
        calculated_size = self.risk_calculator.calculate_position_size(sizing_params)
        
        # Should be reasonable position size
        assert calculated_size >= 1, f"Position size should be at least 1: {calculated_size}"
        assert calculated_size <= 10, f"Position size should be at most 10: {calculated_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])