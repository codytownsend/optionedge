"""
Constraint validation tests for all hard constraints.
Tests POP constraints, credit ratios, sector diversification, and portfolio limits.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from src.data.models.options import OptionQuote, OptionType, Greeks
from src.data.models.trades import (
    StrategyDefinition, TradeCandidate, TradeLeg, StrategyType, TradeDirection
)
from src.domain.services.constraint_engine import (
    HardConstraintValidator, ConstraintValidationResult, ConstraintViolation,
    GICS_SECTORS
)
from src.domain.services.portfolio_risk_controller import PortfolioRiskController
from src.domain.services.trade_selector import TradeSelector


class TestConstraintValidation:
    """Test all hard constraint validation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = HardConstraintValidator()
        self.nav = Decimal('100000')
        self.available_capital = Decimal('10000')
        self.current_trades = []
    
    def create_test_trade(
        self, 
        ticker="AAPL", 
        pop=0.70, 
        net_credit=2.0, 
        max_loss=3.0, 
        quote_age_minutes=5,
        strategy_type=StrategyType.PUT_CREDIT_SPREAD,
        open_interest=100,
        volume=50,
        bid_ask_spread_pct=0.03
    ):
        """Create a test trade with specified parameters."""
        # Create option with quote timestamp
        quote_timestamp = datetime.utcnow() - timedelta(minutes=quote_age_minutes)
        
        bid = Decimal('2.50')
        ask = bid * (1 + Decimal(str(bid_ask_spread_pct)))
        
        option = OptionQuote(
            symbol=f"{ticker}250117P00150000",
            strike=Decimal('150'),
            expiration=date(2025, 1, 17),
            option_type=OptionType.PUT,
            bid=bid,
            ask=ask,
            last=bid + (ask - bid) / 2,
            volume=volume,
            open_interest=open_interest,
            quote_timestamp=quote_timestamp,
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
            days_to_expiration=30
        )
        
        return TradeCandidate(strategy=strategy)
    
    def test_quote_freshness_validation(self):
        """Test quote freshness validation constraint."""
        # Test fresh quote (should pass)
        fresh_trade = self.create_test_trade(quote_age_minutes=5)
        result = self.validator.validate_quote_freshness(fresh_trade, max_age_minutes=10)
        assert result.is_valid
        
        # Test stale quote (should fail)
        stale_trade = self.create_test_trade(quote_age_minutes=15)
        result = self.validator.validate_quote_freshness(stale_trade, max_age_minutes=10)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "stale quote" in result.violations[0].message.lower()
    
    def test_pop_constraint_validation(self):
        """Test probability of profit constraint validation."""
        # Test high POP (should pass)
        high_pop_trade = self.create_test_trade(pop=0.70)
        result = self.validator.validate_pop_constraint(high_pop_trade, min_pop=0.65)
        assert result.is_valid
        
        # Test low POP (should fail)
        low_pop_trade = self.create_test_trade(pop=0.60)
        result = self.validator.validate_pop_constraint(low_pop_trade, min_pop=0.65)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "probability of profit" in result.violations[0].message.lower()
    
    def test_credit_ratio_constraint_validation(self):
        """Test credit-to-max-loss ratio constraint validation."""
        # Test high ratio (should pass)
        high_ratio_trade = self.create_test_trade(net_credit=2.0, max_loss=5.0)  # 0.40 ratio
        result = self.validator.validate_credit_ratio(high_ratio_trade, min_ratio=0.33)
        assert result.is_valid
        
        # Test low ratio (should fail)
        low_ratio_trade = self.create_test_trade(net_credit=1.0, max_loss=4.0)  # 0.25 ratio
        result = self.validator.validate_credit_ratio(low_ratio_trade, min_ratio=0.33)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "credit" in result.violations[0].message.lower()
        assert "ratio" in result.violations[0].message.lower()
    
    def test_max_loss_constraint_validation(self):
        """Test maximum loss per trade constraint validation."""
        # Test acceptable loss (should pass)
        low_loss_trade = self.create_test_trade(max_loss=400)
        result = self.validator.validate_max_loss(low_loss_trade, self.nav, max_loss_dollars=500)
        assert result.is_valid
        
        # Test high loss (should fail)
        high_loss_trade = self.create_test_trade(max_loss=600)
        result = self.validator.validate_max_loss(high_loss_trade, self.nav, max_loss_dollars=500)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "maximum loss" in result.violations[0].message.lower()
    
    def test_capital_requirement_validation(self):
        """Test capital availability constraint validation."""
        # Test sufficient capital (should pass)
        low_capital_trade = self.create_test_trade(max_loss=1000)
        result = self.validator.validate_capital_requirement(
            low_capital_trade, Decimal('5000')
        )
        assert result.is_valid
        
        # Test insufficient capital (should fail)
        high_capital_trade = self.create_test_trade(max_loss=6000)
        result = self.validator.validate_capital_requirement(
            high_capital_trade, Decimal('5000')
        )
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "capital" in result.violations[0].message.lower()
    
    def test_liquidity_validation(self):
        """Test liquidity constraint validation."""
        # Test good liquidity (should pass)
        liquid_trade = self.create_test_trade(
            open_interest=100,
            volume=50,
            bid_ask_spread_pct=0.03
        )
        result = self.validator.validate_liquidity(liquid_trade)
        assert result.is_valid
        
        # Test poor liquidity - low open interest (should fail)
        illiquid_trade = self.create_test_trade(
            open_interest=10,  # Below minimum
            volume=50,
            bid_ask_spread_pct=0.03
        )
        result = self.validator.validate_liquidity(illiquid_trade, min_open_interest=50)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "open interest" in result.violations[0].message.lower()
        
        # Test poor liquidity - wide spread (should fail)
        wide_spread_trade = self.create_test_trade(
            open_interest=100,
            volume=50,
            bid_ask_spread_pct=0.10  # 10% spread
        )
        result = self.validator.validate_liquidity(wide_spread_trade, max_bid_ask_spread_pct=0.05)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "spread" in result.violations[0].message.lower()
    
    def test_sector_diversification_validation(self):
        """Test sector diversification constraint validation."""
        # Setup existing trades in tech sector
        existing_tech_trades = [
            self.create_test_trade(ticker="AAPL"),  # Tech
            self.create_test_trade(ticker="MSFT"),  # Tech
        ]
        
        # Test adding another tech trade (should fail)
        new_tech_trade = self.create_test_trade(ticker="GOOGL")  # Tech
        result = self.validator.validate_sector_diversification(
            new_tech_trade, existing_tech_trades, max_trades_per_sector=2
        )
        assert not result.is_valid
        assert len(result.violations) == 1
        assert "sector" in result.violations[0].message.lower()
        
        # Test adding health care trade (should pass)
        new_health_trade = self.create_test_trade(ticker="JNJ")  # Health Care
        result = self.validator.validate_sector_diversification(
            new_health_trade, existing_tech_trades, max_trades_per_sector=2
        )
        assert result.is_valid
    
    def test_all_constraints_validation(self):
        """Test master validation function for all constraints."""
        # Test trade that passes all constraints
        good_trade = self.create_test_trade(
            ticker="AAPL",
            pop=0.70,
            net_credit=2.0,
            max_loss=300,
            quote_age_minutes=5,
            open_interest=100,
            volume=50,
            bid_ask_spread_pct=0.03
        )
        
        result = self.validator.validate_trade_constraints(
            good_trade, 
            self.current_trades, 
            self.nav, 
            self.available_capital
        )
        assert result.is_valid
        assert len(result.violations) == 0
        
        # Test trade that fails multiple constraints
        bad_trade = self.create_test_trade(
            ticker="AAPL",
            pop=0.60,  # Too low
            net_credit=1.0,
            max_loss=4.0,  # Low credit ratio
            quote_age_minutes=15,  # Stale quote
            open_interest=10,  # Low liquidity
            volume=5,
            bid_ask_spread_pct=0.08  # Wide spread
        )
        
        result = self.validator.validate_trade_constraints(
            bad_trade, 
            self.current_trades, 
            self.nav, 
            self.available_capital
        )
        assert not result.is_valid
        assert len(result.violations) >= 3  # Multiple violations


class TestPortfolioConstraints:
    """Test portfolio-level constraint validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.portfolio_controller = PortfolioRiskController()
        self.nav = Decimal('100000')
    
    def create_test_trade_with_greeks(self, ticker="AAPL", delta=0.25, vega=0.15):
        """Create a test trade with specific Greeks."""
        option = OptionQuote(
            symbol=f"{ticker}250117C00150000",
            strike=Decimal('150'),
            expiration=date(2025, 1, 17),
            option_type=OptionType.CALL,
            bid=Decimal('2.50'),
            ask=Decimal('2.55'),
            last=Decimal('2.52'),
            volume=100,
            open_interest=500,
            greeks=Greeks(
                delta=delta,
                gamma=0.05,
                theta=-0.08,
                vega=vega,
                rho=0.02
            )
        )
        
        leg = TradeLeg(
            option=option,
            quantity=1,
            direction=TradeDirection.BUY
        )
        
        strategy = StrategyDefinition(
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            underlying=ticker,
            legs=[leg],
            probability_of_profit=0.70,
            net_credit=Decimal('2.0'),
            max_loss=Decimal('3.0')
        )
        
        return TradeCandidate(strategy=strategy)
    
    def test_portfolio_delta_limits(self):
        """Test portfolio delta limit constraints."""
        # Create existing trades with high delta
        existing_trades = [
            self.create_test_trade_with_greeks(ticker="AAPL", delta=0.30),
            self.create_test_trade_with_greeks(ticker="MSFT", delta=0.25),
            self.create_test_trade_with_greeks(ticker="GOOGL", delta=0.20)
        ]
        
        # Test adding trade that would exceed delta limits
        new_trade = self.create_test_trade_with_greeks(ticker="TSLA", delta=0.35)
        
        violations = self.portfolio_controller.check_portfolio_limits_compliance(
            new_trade, existing_trades, self.nav
        )
        
        # Should have delta limit violation
        assert len(violations) > 0
        assert any("delta" in v.message.lower() for v in violations)
    
    def test_portfolio_vega_limits(self):
        """Test portfolio vega limit constraints."""
        # Create existing trades with negative vega
        existing_trades = [
            self.create_test_trade_with_greeks(ticker="AAPL", vega=-0.02),
            self.create_test_trade_with_greeks(ticker="MSFT", vega=-0.02),
            self.create_test_trade_with_greeks(ticker="GOOGL", vega=-0.02)
        ]
        
        # Test adding trade that would exceed vega limits
        new_trade = self.create_test_trade_with_greeks(ticker="TSLA", vega=-0.03)
        
        violations = self.portfolio_controller.check_portfolio_limits_compliance(
            new_trade, existing_trades, self.nav
        )
        
        # Should have vega limit violation
        assert len(violations) > 0
        assert any("vega" in v.message.lower() for v in violations)
    
    def test_portfolio_within_limits(self):
        """Test portfolio that stays within all limits."""
        # Create existing trades within limits
        existing_trades = [
            self.create_test_trade_with_greeks(ticker="AAPL", delta=0.10, vega=0.05),
            self.create_test_trade_with_greeks(ticker="MSFT", delta=0.15, vega=0.08)
        ]
        
        # Test adding trade that stays within limits
        new_trade = self.create_test_trade_with_greeks(ticker="JNJ", delta=0.05, vega=0.03)
        
        violations = self.portfolio_controller.check_portfolio_limits_compliance(
            new_trade, existing_trades, self.nav
        )
        
        # Should have no violations
        assert len(violations) == 0


class TestTradeSelectionConstraints:
    """Test trade selection process constraint validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.selector = TradeSelector(nav=Decimal('100000'))
    
    def create_scored_trade(self, ticker="AAPL", model_score=75.0, pop=0.70):
        """Create a scored trade candidate."""
        from src.domain.services.scoring_engine import ScoredTradeCandidate, ComponentScores
        
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
        
        leg = TradeLeg(
            option=option,
            quantity=1,
            direction=TradeDirection.SELL
        )
        
        strategy = StrategyDefinition(
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            underlying=ticker,
            legs=[leg],
            probability_of_profit=pop,
            net_credit=Decimal('2.0'),
            max_loss=Decimal('300')
        )
        
        trade_candidate = TradeCandidate(strategy=strategy)
        
        # Create component scores
        component_scores = ComponentScores(
            model_score=model_score,
            pop_score=pop * 100,
            iv_rank_score=60.0,
            momentum_z=0.5,
            flow_z=0.3,
            risk_reward_score=70.0,
            liquidity_score=80.0
        )
        
        return ScoredTradeCandidate(
            trade_candidate=trade_candidate,
            component_scores=component_scores
        )
    
    def test_minimum_trades_requirement(self):
        """Test minimum 5 trades requirement."""
        # Create only 3 valid trades
        candidates = [
            self.create_scored_trade(ticker="AAPL", model_score=85.0),
            self.create_scored_trade(ticker="MSFT", model_score=80.0),
            self.create_scored_trade(ticker="GOOGL", model_score=75.0)
        ]
        
        result = self.selector.select_final_trades(
            candidates, 
            current_trades=[], 
            available_capital=Decimal('10000')
        )
        
        # Should not be execution ready
        assert not result.execution_ready
        assert "Fewer than 5 trades" in result.selection_summary['message']
    
    def test_sector_diversification_enforcement(self):
        """Test sector diversification enforcement during selection."""
        # Create 6 tech trades (only 2 should be selected)
        tech_candidates = [
            self.create_scored_trade(ticker="AAPL", model_score=90.0),
            self.create_scored_trade(ticker="MSFT", model_score=85.0),
            self.create_scored_trade(ticker="GOOGL", model_score=80.0),
            self.create_scored_trade(ticker="META", model_score=75.0),
            self.create_scored_trade(ticker="NFLX", model_score=70.0),
            self.create_scored_trade(ticker="NVDA", model_score=65.0)
        ]
        
        result = self.selector.select_final_trades(
            tech_candidates, 
            current_trades=[], 
            available_capital=Decimal('10000')
        )
        
        # Should reject some trades due to sector limits
        tech_selected = [t for t in result.selected_trades 
                        if GICS_SECTORS.get(t.trade_candidate.strategy.underlying) == "Information Technology"]
        
        assert len(tech_selected) <= 2  # Max 2 per sector
        
        # Should have rejections for sector limits
        sector_rejections = [r for r in result.rejected_trades 
                           if "sector" in r[1].lower()]
        assert len(sector_rejections) > 0
    
    def test_constraint_validation_integration(self):
        """Test integration of all constraint validation."""
        # Create mix of good and bad trades
        candidates = [
            self.create_scored_trade(ticker="AAPL", model_score=85.0, pop=0.75),  # Good
            self.create_scored_trade(ticker="MSFT", model_score=80.0, pop=0.60),  # Bad POP
            self.create_scored_trade(ticker="GOOGL", model_score=75.0, pop=0.70), # Good
            self.create_scored_trade(ticker="JNJ", model_score=70.0, pop=0.68),   # Good
            self.create_scored_trade(ticker="PG", model_score=65.0, pop=0.72),    # Good
            self.create_scored_trade(ticker="KO", model_score=60.0, pop=0.69)     # Good
        ]
        
        result = self.selector.select_final_trades(
            candidates, 
            current_trades=[], 
            available_capital=Decimal('10000')
        )
        
        # Should have exactly 5 trades (excluding the bad POP trade)
        assert len(result.selected_trades) == 5
        assert result.execution_ready
        
        # Should have constraint rejections
        constraint_rejections = [r for r in result.rejected_trades 
                               if "constraint" in r[1].lower()]
        assert len(constraint_rejections) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])