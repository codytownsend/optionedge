"""Unit tests for options models."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from src.data.models.options import (
    OptionType, OptionStyle, OptionSymbol, Greeks, OptionQuote,
    OptionsChain, OptionContract
)


class TestOptionSymbol:
    """Test OptionSymbol class."""
    
    def test_create_option_symbol(self):
        """Test creating an option symbol."""
        symbol = OptionSymbol(
            underlying="AAPL",
            expiration=date(2024, 1, 19),
            strike=Decimal("150.00"),
            option_type=OptionType.CALL
        )
        
        assert symbol.underlying == "AAPL"
        assert symbol.expiration == date(2024, 1, 19)
        assert symbol.strike == Decimal("150.00")
        assert symbol.option_type == OptionType.CALL
    
    def test_option_symbol_string_representation(self):
        """Test option symbol string representation."""
        symbol = OptionSymbol(
            underlying="AAPL",
            expiration=date(2024, 1, 19),
            strike=Decimal("150.00"),
            option_type=OptionType.CALL
        )
        
        symbol_str = str(symbol)
        assert "AAPL" in symbol_str
        assert "C" in symbol_str
    
    def test_option_symbol_from_string(self):
        """Test parsing option symbol from string."""
        # This would need proper implementation
        pass
    
    @pytest.mark.parametrize("option_type,expected_char", [
        (OptionType.CALL, "C"),
        (OptionType.PUT, "P"),
    ])
    def test_option_type_string_representation(self, option_type, expected_char):
        """Test option type string representation."""
        symbol = OptionSymbol(
            underlying="AAPL",
            expiration=date(2024, 1, 19),
            strike=Decimal("150.00"),
            option_type=option_type
        )
        
        assert expected_char in str(symbol)


class TestGreeks:
    """Test Greeks class."""
    
    def test_create_greeks(self):
        """Test creating Greeks."""
        greeks = Greeks(
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.01
        )
        
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.1
        assert greeks.theta == -0.05
        assert greeks.vega == 0.2
        assert greeks.rho == 0.01
    
    def test_greeks_validation(self):
        """Test Greeks validation."""
        # Delta should be between -1 and 1
        with pytest.raises(ValueError):
            Greeks(delta=2.0)
        
        with pytest.raises(ValueError):
            Greeks(delta=-2.0)
        
        # Gamma should be non-negative
        with pytest.raises(ValueError):
            Greeks(gamma=-0.1)
    
    def test_greeks_optional_fields(self):
        """Test Greeks with optional fields."""
        greeks = Greeks(delta=0.5)
        
        assert greeks.delta == 0.5
        assert greeks.gamma is None
        assert greeks.theta is None
        assert greeks.vega is None
        assert greeks.rho is None


class TestOptionQuote:
    """Test OptionQuote class."""
    
    def test_create_option_quote(self):
        """Test creating an option quote."""
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20"),
            last=Decimal("5.10"),
            volume=100,
            open_interest=1000,
            implied_volatility=0.25
        )
        
        assert quote.symbol == "AAPL240119C00150000"
        assert quote.underlying == "AAPL"
        assert quote.strike == Decimal("150.00")
        assert quote.option_type == OptionType.CALL
        assert quote.bid == Decimal("5.00")
        assert quote.ask == Decimal("5.20")
    
    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20")
        )
        
        assert quote.mid_price == Decimal("5.10")
    
    def test_bid_ask_spread(self):
        """Test bid-ask spread calculation."""
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20")
        )
        
        assert quote.bid_ask_spread == Decimal("0.20")
        assert abs(quote.bid_ask_spread_pct - 0.039216) < 0.0001
    
    def test_days_to_expiration(self):
        """Test days to expiration calculation."""
        # This would need to be mocked for consistent testing
        pass
    
    def test_liquidity_check(self):
        """Test liquidity check."""
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20"),
            volume=100,
            open_interest=1000
        )
        
        assert quote.is_liquid(min_volume=10, min_oi=100, max_spread_pct=0.5)
        assert not quote.is_liquid(min_volume=200, min_oi=100, max_spread_pct=0.5)
    
    def test_quote_freshness(self):
        """Test quote freshness check."""
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            quote_time=datetime.utcnow()
        )
        
        assert quote.is_quote_fresh(max_age_minutes=10)


class TestOptionsChain:
    """Test OptionsChain class."""
    
    def test_create_options_chain(self):
        """Test creating an options chain."""
        chain = OptionsChain(
            underlying="AAPL",
            underlying_price=Decimal("150.00")
        )
        
        assert chain.underlying == "AAPL"
        assert chain.underlying_price == Decimal("150.00")
        assert len(chain.options) == 0
    
    def test_add_option_to_chain(self):
        """Test adding option to chain."""
        chain = OptionsChain(underlying="AAPL")
        
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20")
        )
        
        chain.add_option(quote)
        
        retrieved_quote = chain.get_option(
            date(2024, 1, 19), 
            Decimal("150.00"), 
            OptionType.CALL
        )
        
        assert retrieved_quote == quote
    
    def test_get_expirations(self):
        """Test getting expirations."""
        chain = OptionsChain(underlying="AAPL")
        
        # Add options with different expirations
        exp1 = date(2024, 1, 19)
        exp2 = date(2024, 2, 16)
        
        quote1 = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=exp1,
            option_type=OptionType.CALL
        )
        
        quote2 = OptionQuote(
            symbol="AAPL240216C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=exp2,
            option_type=OptionType.CALL
        )
        
        chain.add_option(quote1)
        chain.add_option(quote2)
        
        expirations = chain.get_expirations()
        assert exp1 in expirations
        assert exp2 in expirations
        assert expirations == sorted(expirations)
    
    def test_get_atm_strike(self):
        """Test getting ATM strike."""
        chain = OptionsChain(
            underlying="AAPL",
            underlying_price=Decimal("152.50")
        )
        
        exp = date(2024, 1, 19)
        
        # Add options with different strikes
        for strike in [Decimal("150.00"), Decimal("155.00"), Decimal("160.00")]:
            quote = OptionQuote(
                symbol=f"AAPL240119C00{int(strike*100):05d}000",
                underlying="AAPL",
                strike=strike,
                expiration=exp,
                option_type=OptionType.CALL
            )
            chain.add_option(quote)
        
        atm_strike = chain.get_atm_strike(exp)
        assert atm_strike == Decimal("155.00")  # Closest to 152.50
    
    def test_filter_liquid_options(self):
        """Test filtering liquid options."""
        chain = OptionsChain(underlying="AAPL")
        
        # Add liquid option
        liquid_quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20"),
            volume=100,
            open_interest=1000
        )
        
        # Add illiquid option
        illiquid_quote = OptionQuote(
            symbol="AAPL240119C00160000",
            underlying="AAPL",
            strike=Decimal("160.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("1.00"),
            ask=Decimal("3.00"),  # Wide spread
            volume=5,  # Low volume
            open_interest=50  # Low OI
        )
        
        chain.add_option(liquid_quote)
        chain.add_option(illiquid_quote)
        
        filtered_chain = chain.filter_liquid_options(
            min_volume=10,
            min_oi=100,
            max_spread_pct=0.5
        )
        
        # Should only contain liquid option
        assert len(filtered_chain.options) == 1
        assert filtered_chain.get_option(
            date(2024, 1, 19), 
            Decimal("150.00"), 
            OptionType.CALL
        ) is not None
        assert filtered_chain.get_option(
            date(2024, 1, 19), 
            Decimal("160.00"), 
            OptionType.CALL
        ) is None


class TestOptionContract:
    """Test OptionContract class."""
    
    def test_create_option_contract(self):
        """Test creating an option contract."""
        symbol = OptionSymbol(
            underlying="AAPL",
            expiration=date(2024, 1, 19),
            strike=Decimal("150.00"),
            option_type=OptionType.CALL
        )
        
        contract = OptionContract(
            symbol=symbol,
            contract_size=100,
            exercise_style=OptionStyle.AMERICAN
        )
        
        assert contract.symbol == symbol
        assert contract.contract_size == 100
        assert contract.exercise_style == OptionStyle.AMERICAN
    
    def test_option_contract_with_quote(self):
        """Test option contract with quote."""
        symbol = OptionSymbol(
            underlying="AAPL",
            expiration=date(2024, 1, 19),
            strike=Decimal("150.00"),
            option_type=OptionType.CALL
        )
        
        quote = OptionQuote(
            symbol="AAPL240119C00150000",
            underlying="AAPL",
            strike=Decimal("150.00"),
            expiration=date(2024, 1, 19),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.20")
        )
        
        contract = OptionContract(symbol=symbol, quote=quote)
        
        assert contract.quote == quote