"""
Unit tests for options data models.
"""

import pytest
from decimal import Decimal
from datetime import date, datetime
from unittest.mock import Mock, patch

from src.data.models.options import OptionContract, OptionChain, OptionData, OptionType
from src.domain.value_objects.greeks import Greeks
from src.infrastructure.error_handling import ValidationError


class TestOptionContract:
    """Test OptionContract model."""
    
    def test_option_contract_creation(self):
        """Test creating an option contract."""
        greeks = Greeks(
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1
        )
        
        contract = OptionContract(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type="call",
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            greeks=greeks,
            timestamp=datetime.now()
        )
        
        assert contract.symbol == "AAPL210716C00150000"
        assert contract.underlying_symbol == "AAPL"
        assert contract.option_type == "call"
        assert contract.strike_price == 150.0
        assert contract.greeks.delta == 0.5
        assert contract.bid_ask_spread == 0.05
        assert contract.volume == 1000
        assert contract.open_interest == 5000
    
    def test_option_contract_validation(self):
        """Test option contract validation."""
        greeks = Greeks(
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1
        )
        
        # Test invalid strike price
        with pytest.raises(ValidationError, match="Strike price must be positive"):
            OptionContract(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=-150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                greeks=greeks,
                timestamp=datetime.now()
            )
        
        # Test invalid bid/ask spread
        with pytest.raises(ValidationError, match="Ask must be greater than or equal to bid"):
            OptionContract(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.55,
                ask=2.50,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                greeks=greeks,
                timestamp=datetime.now()
            )
    
    def test_option_contract_properties(self):
        """Test computed properties."""
        greeks = Greeks(
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1
        )
        
        contract = OptionContract(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type="call",
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            greeks=greeks,
            timestamp=datetime.now()
        )
        
        assert contract.bid_ask_spread == 0.05
        assert contract.mid_price == 2.525
        assert contract.volume_open_interest_ratio == 0.2
        assert contract.is_liquid(min_volume=500, min_open_interest=1000)
        assert not contract.is_liquid(min_volume=2000, min_open_interest=1000)
    
    def test_option_contract_serialization(self):
        """Test serialization to/from dict."""
        greeks = Greeks(
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1
        )
        
        contract = OptionContract(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type="call",
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            greeks=greeks,
            timestamp=datetime.now()
        )
        
        # Test to_dict
        contract_dict = contract.to_dict()
        assert contract_dict['symbol'] == "AAPL210716C00150000"
        assert contract_dict['strike_price'] == 150.0
        assert contract_dict['greeks']['delta'] == 0.5
        
        # Test from_dict
        reconstructed = OptionContract.from_dict(contract_dict)
        assert reconstructed.symbol == contract.symbol
        assert reconstructed.strike_price == contract.strike_price
        assert reconstructed.greeks.delta == contract.greeks.delta


class TestOptionChain:
    """Test OptionChain model."""
    
    def test_option_chain_creation(self):
        """Test creating an option chain."""
        # Create sample contracts
        calls = [
            OptionContract(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                greeks=Greeks(0.5, 0.1, -0.05, 0.2, 0.1),
                timestamp=datetime.now()
            )
        ]
        
        puts = [
            OptionContract(
                symbol="AAPL210716P00150000",
                underlying_symbol="AAPL",
                option_type="put",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.45,
                ask=2.50,
                last_price=2.47,
                volume=800,
                open_interest=4000,
                implied_volatility=0.28,
                greeks=Greeks(-0.5, 0.1, -0.05, 0.2, -0.1),
                timestamp=datetime.now()
            )
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiration_date=date(2021, 7, 16),
            calls=calls,
            puts=puts,
            timestamp=datetime.now()
        )
        
        assert chain.symbol == "AAPL"
        assert len(chain.calls) == 1
        assert len(chain.puts) == 1
        assert chain.total_contracts == 2
    
    def test_option_chain_filtering(self):
        """Test filtering options in chain."""
        # Create sample contracts with different strikes
        calls = [
            OptionContract(
                symbol="AAPL210716C00145000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=145.0,
                expiration_date=date(2021, 7, 16),
                bid=5.50,
                ask=5.55,
                last_price=5.52,
                volume=2000,
                open_interest=8000,
                implied_volatility=0.23,
                greeks=Greeks(0.7, 0.08, -0.06, 0.18, 0.12),
                timestamp=datetime.now()
            ),
            OptionContract(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                greeks=Greeks(0.5, 0.1, -0.05, 0.2, 0.1),
                timestamp=datetime.now()
            ),
            OptionContract(
                symbol="AAPL210716C00155000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=155.0,
                expiration_date=date(2021, 7, 16),
                bid=0.75,
                ask=0.80,
                last_price=0.77,
                volume=500,
                open_interest=2000,
                implied_volatility=0.27,
                greeks=Greeks(0.3, 0.12, -0.04, 0.22, 0.08),
                timestamp=datetime.now()
            )
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiration_date=date(2021, 7, 16),
            calls=calls,
            puts=[],
            timestamp=datetime.now()
        )
        
        # Test filtering by strike range
        filtered = chain.filter_by_strike_range(min_strike=146.0, max_strike=154.0)
        assert len(filtered.calls) == 1
        assert filtered.calls[0].strike_price == 150.0
        
        # Test filtering by delta range
        filtered = chain.filter_by_delta_range(min_delta=0.4, max_delta=0.6)
        assert len(filtered.calls) == 1
        assert filtered.calls[0].greeks.delta == 0.5
        
        # Test filtering by volume
        filtered = chain.filter_by_volume(min_volume=1500)
        assert len(filtered.calls) == 1
        assert filtered.calls[0].volume == 2000
    
    def test_option_chain_analytics(self):
        """Test chain analytics calculations."""
        calls = [
            OptionContract(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                greeks=Greeks(0.5, 0.1, -0.05, 0.2, 0.1),
                timestamp=datetime.now()
            )
        ]
        
        puts = [
            OptionContract(
                symbol="AAPL210716P00150000",
                underlying_symbol="AAPL",
                option_type="put",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.45,
                ask=2.50,
                last_price=2.47,
                volume=800,
                open_interest=4000,
                implied_volatility=0.28,
                greeks=Greeks(-0.5, 0.1, -0.05, 0.2, -0.1),
                timestamp=datetime.now()
            )
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiration_date=date(2021, 7, 16),
            calls=calls,
            puts=puts,
            timestamp=datetime.now()
        )
        
        # Test put-call ratio
        assert chain.put_call_ratio == 0.8  # 800 / 1000
        
        # Test total volume
        assert chain.total_volume == 1800
        
        # Test total open interest
        assert chain.total_open_interest == 9000
        
        # Test average IV
        assert chain.average_iv == 0.265  # (0.25 + 0.28) / 2
    
    def test_option_chain_serialization(self):
        """Test chain serialization."""
        calls = [
            OptionContract(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="call",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                greeks=Greeks(0.5, 0.1, -0.05, 0.2, 0.1),
                timestamp=datetime.now()
            )
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiration_date=date(2021, 7, 16),
            calls=calls,
            puts=[],
            timestamp=datetime.now()
        )
        
        # Test to_dict
        chain_dict = chain.to_dict()
        assert chain_dict['symbol'] == "AAPL"
        assert len(chain_dict['calls']) == 1
        assert len(chain_dict['puts']) == 0
        
        # Test from_dict
        reconstructed = OptionChain.from_dict(chain_dict)
        assert reconstructed.symbol == chain.symbol
        assert len(reconstructed.calls) == len(chain.calls)
        assert reconstructed.calls[0].symbol == chain.calls[0].symbol


class TestOptionData:
    """Test OptionData model."""
    
    def test_option_data_creation(self):
        """Test creating option data."""
        data = OptionData(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1,
            timestamp=datetime.now()
        )
        
        assert data.symbol == "AAPL210716C00150000"
        assert data.option_type == OptionType.CALL
        assert data.strike_price == 150.0
        assert data.delta == 0.5
    
    def test_option_data_validation(self):
        """Test option data validation."""
        # Test invalid option type
        with pytest.raises(ValidationError):
            OptionData(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type="invalid",
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                delta=0.5,
                gamma=0.1,
                theta=-0.05,
                vega=0.2,
                rho=0.1,
                timestamp=datetime.now()
            )
        
        # Test invalid volume
        with pytest.raises(ValidationError):
            OptionData(
                symbol="AAPL210716C00150000",
                underlying_symbol="AAPL",
                option_type=OptionType.CALL,
                strike_price=150.0,
                expiration_date=date(2021, 7, 16),
                bid=2.50,
                ask=2.55,
                last_price=2.52,
                volume=-1000,
                open_interest=5000,
                implied_volatility=0.25,
                delta=0.5,
                gamma=0.1,
                theta=-0.05,
                vega=0.2,
                rho=0.1,
                timestamp=datetime.now()
            )
    
    def test_option_data_properties(self):
        """Test computed properties."""
        data = OptionData(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1,
            timestamp=datetime.now()
        )
        
        assert data.bid_ask_spread == 0.05
        assert data.mid_price == 2.525
        assert data.is_in_the_money(underlying_price=155.0)
        assert not data.is_in_the_money(underlying_price=145.0)
        assert data.moneyness(underlying_price=150.0) == 1.0
        assert data.time_to_expiration.days > 0
    
    def test_option_data_greeks_object(self):
        """Test Greeks object creation."""
        data = OptionData(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1,
            timestamp=datetime.now()
        )
        
        greeks = data.get_greeks()
        assert isinstance(greeks, Greeks)
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.1
        assert greeks.theta == -0.05
        assert greeks.vega == 0.2
        assert greeks.rho == 0.1
    
    def test_option_data_serialization(self):
        """Test serialization to/from dict."""
        data = OptionData(
            symbol="AAPL210716C00150000",
            underlying_symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=150.0,
            expiration_date=date(2021, 7, 16),
            bid=2.50,
            ask=2.55,
            last_price=2.52,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.1,
            theta=-0.05,
            vega=0.2,
            rho=0.1,
            timestamp=datetime.now()
        )
        
        # Test to_dict
        data_dict = data.to_dict()
        assert data_dict['symbol'] == "AAPL210716C00150000"
        assert data_dict['option_type'] == "call"
        assert data_dict['strike_price'] == 150.0
        assert data_dict['delta'] == 0.5
        
        # Test from_dict
        reconstructed = OptionData.from_dict(data_dict)
        assert reconstructed.symbol == data.symbol
        assert reconstructed.option_type == data.option_type
        assert reconstructed.strike_price == data.strike_price
        assert reconstructed.delta == data.delta