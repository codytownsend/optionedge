"""
Configuration management for the options trading engine.
Handles user settings, API configurations, and system parameters.
"""

import os
from decimal import Decimal
from typing import Dict, List, Optional, Set
from pathlib import Path

from pydantic import BaseSettings, Field, validator
from pydantic.types import PositiveFloat, PositiveInt


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    # Tradier API
    tradier_api_key: str = Field(..., env="TRADIER_API_KEY")
    tradier_api_url: str = Field("https://api.tradier.com", env="TRADIER_API_URL")
    tradier_rate_limit: int = Field(120, env="TRADIER_RATE_LIMIT")  # per minute
    
    # Yahoo Finance via RapidAPI
    yahoo_rapid_api_key: str = Field(..., env="YAHOO_RAPID_API_KEY")
    yahoo_api_url: str = Field("https://yahoo-finance15.p.rapidapi.com", env="YAHOO_API_URL")
    
    # FRED API
    fred_api_key: str = Field(..., env="FRED_API_KEY")
    fred_api_url: str = Field("https://api.stlouisfed.org/fred", env="FRED_API_URL")
    
    # QuiverQuant API
    quiver_api_key: Optional[str] = Field(None, env="QUIVER_API_KEY")
    quiver_api_url: str = Field("https://api.quiverquant.com", env="QUIVER_API_URL")
    
    # General API settings
    api_timeout: int = Field(30, env="API_TIMEOUT")
    api_retry_attempts: int = Field(3, env="API_RETRY_ATTEMPTS")
    api_retry_delay: float = Field(1.0, env="API_RETRY_DELAY_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class PortfolioConfig(BaseSettings):
    """User portfolio configuration."""
    
    # Portfolio parameters
    nav: PositiveFloat = Field(100000.0, description="Net Asset Value")
    capital_available: PositiveFloat = Field(10000.0, description="Available capital for new trades")
    
    # Risk parameters
    max_trades: PositiveInt = Field(5, description="Maximum number of concurrent trades")
    max_loss_per_trade: PositiveFloat = Field(500.0, description="Maximum loss per trade")
    min_pop: float = Field(0.65, description="Minimum Probability of Profit")
    min_credit_to_max_loss: float = Field(0.33, description="Minimum credit-to-max-loss ratio")
    
    # Greeks limits (as multipliers of NAV/100k)
    max_delta_multiplier: float = Field(0.30, description="Max absolute delta multiplier")
    min_vega_multiplier: float = Field(-0.05, description="Min vega multiplier")
    max_gamma_exposure: float = Field(1000.0, description="Max gamma exposure")
    max_theta_exposure: float = Field(-100.0, description="Max theta exposure")
    
    # Diversification
    max_trades_per_sector: int = Field(2, description="Max trades per GICS sector")
    
    @validator('capital_available')
    def capital_must_be_less_than_nav(cls, v, values):
        if 'nav' in values and v > values['nav']:
            raise ValueError('capital_available cannot exceed nav')
        return v
    
    @property
    def max_delta_exposure(self) -> float:
        """Calculate maximum delta exposure based on NAV."""
        return self.max_delta_multiplier * (self.nav / 100000)
    
    @property
    def min_vega_exposure(self) -> float:
        """Calculate minimum vega exposure based on NAV."""
        return self.min_vega_multiplier * (self.nav / 100000)


class MarketScanConfig(BaseSettings):
    """Market scanning configuration."""
    
    # Universe selection
    scan_universe: str = Field("SP500", description="Universe to scan: SP500, NASDAQ100, or custom")
    custom_tickers: Optional[List[str]] = Field(None, description="Custom ticker list")
    exclude_tickers: Set[str] = Field(set(), description="Tickers to exclude")
    
    # Time constraints
    max_days_to_expiration: int = Field(45, description="Maximum days to expiration")
    min_days_to_expiration: int = Field(7, description="Minimum days to expiration")
    max_quote_age_minutes: int = Field(10, description="Maximum quote age in minutes")
    
    # Liquidity filters
    min_option_volume: int = Field(10, description="Minimum option volume")
    min_open_interest: int = Field(100, description="Minimum open interest")
    max_bid_ask_spread_pct: float = Field(0.50, description="Maximum bid-ask spread as % of mid")
    
    # Strategy preferences
    allowed_strategies: Set[str] = Field(
        {"put_credit_spread", "call_credit_spread", "iron_condor", "covered_call"},
        description="Allowed strategy types"
    )


class SystemConfig(BaseSettings):
    """System-level configuration."""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    
    # Caching
    cache_ttl_seconds: int = Field(300, env="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(1000, env="CACHE_MAX_SIZE")
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    
    # Performance
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    
    # Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(8080, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"


class AppSettings:
    """Main application settings container."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize settings from environment and config files."""
        self.api = APIConfig()
        self.portfolio = PortfolioConfig()
        self.market_scan = MarketScanConfig()
        self.system = SystemConfig()
        
        # Load additional config from YAML if provided
        if config_path and config_path.exists():
            self._load_yaml_config(config_path)
    
    def _load_yaml_config(self, config_path: Path):
        """Load additional configuration from YAML file."""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configurations with YAML data
            if 'portfolio' in config_data:
                for key, value in config_data['portfolio'].items():
                    if hasattr(self.portfolio, key):
                        setattr(self.portfolio, key, value)
            
            if 'market_scan' in config_data:
                for key, value in config_data['market_scan'].items():
                    if hasattr(self.market_scan, key):
                        setattr(self.market_scan, key, value)
                        
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")
    
    def validate_all(self) -> bool:
        """Validate all configuration sections."""
        try:
            # Validate that required API keys are present
            if not self.api.tradier_api_key:
                raise ValueError("Tradier API key is required")
            
            # Validate portfolio settings
            if self.portfolio.capital_available > self.portfolio.nav:
                raise ValueError("Available capital cannot exceed NAV")
            
            if self.portfolio.min_pop <= 0 or self.portfolio.min_pop >= 1:
                raise ValueError("Minimum POP must be between 0 and 1")
            
            # Validate market scan settings
            if self.market_scan.max_days_to_expiration <= self.market_scan.min_days_to_expiration:
                raise ValueError("Max days to expiration must be greater than min days")
            
            return True
            
        except ValueError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Convert all settings to dictionary format."""
        return {
            'api': self.api.dict(),
            'portfolio': self.portfolio.dict(),
            'market_scan': self.market_scan.dict(),
            'system': self.system.dict()
        }


# Global settings instance
def get_settings(config_path: Optional[str] = None) -> AppSettings:
    """Get application settings instance."""
    config_file = Path(config_path) if config_path else Path("config/settings.yaml")
    return AppSettings(config_file if config_file.exists() else None)