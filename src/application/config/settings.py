"""
Configuration management system for the Options Trading Engine.
Handles loading, validation, and environment variable overrides.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import jsonschema
from dataclasses import dataclass, field
from copy import deepcopy
import re

from ...infrastructure.error_handling import ConfigurationError, ValidationError, handle_errors


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    user: str
    password: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class CacheConfig:
    """Cache configuration."""
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: Optional[str] = None
    default_ttl: int = 300
    max_memory: str = "256mb"
    eviction_policy: str = "allkeys-lru"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@dataclass
class APIClientConfig:
    """API client configuration."""
    base_url: str
    rate_limit: int
    timeout: int
    retries: int = 3
    backoff_factor: float = 2.0
    api_key: Optional[str] = None


@dataclass
class TradingConfig:
    """Trading configuration."""
    nav: float
    available_capital: float
    max_portfolio_allocation: float
    strategies: list
    watchlist: list
    
    def __post_init__(self):
        """Validate trading configuration."""
        if self.available_capital > self.nav:
            raise ValidationError("Available capital cannot exceed NAV")
        
        if not 0.01 <= self.max_portfolio_allocation <= 0.10:
            raise ValidationError("Max portfolio allocation must be between 1% and 10%")


@dataclass
class ConstraintsConfig:
    """Trading constraints configuration."""
    min_pop: float
    min_credit_ratio: float
    min_dte: int
    max_dte: int
    min_volume: int
    min_open_interest: int
    max_bid_ask_spread: float
    max_sector_allocation: float
    max_total_theta: float
    max_total_vega: float
    max_total_delta: float
    max_net_liquidity: float
    max_delta_long: float
    max_delta_short: float
    min_iv_rank: float
    max_iv_rank: float
    min_days_to_earnings: int
    max_rsi: float
    min_rsi: float
    momentum_z_threshold: float
    flow_z_threshold: float


@dataclass
class ScoringConfig:
    """Scoring configuration."""
    weights: Dict[str, float]
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate scoring weights."""
        if not abs(sum(self.weights.values()) - 1.0) < 0.001:
            raise ValidationError("Scoring weights must sum to 1.0")


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool
    interval: int
    metrics_port: int
    health_check_enabled: bool
    thresholds: Dict[str, float]


@dataclass
class AlertConfig:
    """Alert configuration."""
    email: Dict[str, Any]
    slack: Dict[str, Any]


class ConfigManager:
    """
    Configuration manager with validation and environment variable support.
    
    Features:
    - YAML configuration loading
    - JSON schema validation
    - Environment variable overrides
    - Configuration caching
    - Hot reloading support
    """
    
    def __init__(self, config_path: str = "config/settings.yaml", schema_path: str = "config/settings.schema.json"):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        
        # Configuration cache
        self._config_cache = None
        self._last_modified = 0
        
        # Load schema
        self._schema = self._load_schema()
        
        # Load configuration
        self._config = self._load_and_validate_config()
        
        self.logger.info(f"Configuration loaded from {config_path}")
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for validation."""
        try:
            if not self.schema_path.exists():
                raise ConfigurationError(f"Schema file not found: {self.schema_path}")
            
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load schema: {str(e)}")
    
    @handle_errors(operation_name="load_config")
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        try:
            # Check if file exists
            if not self.config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            # Load YAML configuration
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Apply environment variable overrides
            config_data = self._apply_env_overrides(config_data)
            
            # Validate against schema
            self._validate_config(config_data)
            
            return config_data
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        config_copy = deepcopy(config)
        
        # Environment variable mapping
        env_overrides = {
            # Database
            'DATABASE_HOST': ['database', 'host'],
            'DATABASE_PORT': ['database', 'port'],
            'DATABASE_NAME': ['database', 'database'],
            'DATABASE_USER': ['database', 'user'],
            'DATABASE_PASSWORD': ['database', 'password'],
            
            # Cache
            'REDIS_HOST': ['cache', 'redis_host'],
            'REDIS_PORT': ['cache', 'redis_port'],
            'REDIS_DB': ['cache', 'redis_db'],
            'REDIS_PASSWORD': ['cache', 'redis_password'],
            
            # API Keys
            'TRADIER_API_KEY': ['api', 'tradier', 'api_key'],
            'YAHOO_FINANCE_API_KEY': ['api', 'yahoo_finance', 'api_key'],
            'FRED_API_KEY': ['api', 'fred', 'api_key'],
            'QUIVER_QUANT_API_KEY': ['api', 'quiver', 'api_key'],
            
            # Alert configuration
            'SMTP_USERNAME': ['alerts', 'email', 'username'],
            'SMTP_PASSWORD': ['alerts', 'email', 'password'],
            'SLACK_WEBHOOK_URL': ['alerts', 'slack', 'webhook_url'],
            
            # Application
            'APP_ENVIRONMENT': ['application', 'environment'],
            'APP_DEBUG': ['application', 'debug'],
            'LOG_LEVEL': ['application', 'log_level'],
        }
        
        # Apply overrides
        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value, config_path)
                self._set_nested_value(config_copy, config_path, converted_value)
        
        # Handle template substitutions (${VAR_NAME})
        config_copy = self._substitute_env_templates(config_copy)
        
        return config_copy
    
    def _convert_env_value(self, value: str, config_path: list) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion for known integer fields
        integer_fields = ['port', 'redis_port', 'redis_db', 'pool_size', 'max_overflow']
        if any(field in config_path for field in integer_fields):
            try:
                return int(value)
            except ValueError:
                pass
        
        # Float conversion for known float fields
        float_fields = ['nav', 'available_capital', 'max_portfolio_allocation']
        if any(field in config_path for field in float_fields):
            try:
                return float(value)
            except ValueError:
                pass
        
        # Default to string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """Set nested dictionary value using path list."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _substitute_env_templates(self, config: Any) -> Any:
        """Substitute ${VAR_NAME} templates with environment variables."""
        if isinstance(config, dict):
            return {k: self._substitute_env_templates(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_templates(item) for item in config]
        elif isinstance(config, str):
            # Find ${VAR_NAME} patterns and substitute
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config)
            
            result = config
            for match in matches:
                env_value = os.getenv(match, '')
                result = result.replace(f'${{{match}}}', env_value)
            
            return result
        else:
            return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration against JSON schema."""
        try:
            jsonschema.validate(config, self._schema)
        except jsonschema.ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            raise ConfigurationError(f"Invalid schema: {e.message}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return deepcopy(self._config)
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration object."""
        db_config = self._config['database']
        return DatabaseConfig(**db_config)
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration object."""
        cache_config = self._config['cache']
        return CacheConfig(**cache_config)
    
    def get_api_config(self, api_name: str) -> APIClientConfig:
        """Get API client configuration."""
        if api_name not in self._config['api']:
            raise ConfigurationError(f"API configuration not found: {api_name}")
        
        api_config = self._config['api'][api_name].copy()
        
        # Add API key from environment if not in config
        if 'api_key' not in api_config or not api_config['api_key']:
            env_key_map = {
                'tradier': 'TRADIER_API_KEY',
                'yahoo_finance': 'YAHOO_FINANCE_API_KEY',
                'fred': 'FRED_API_KEY',
                'quiver': 'QUIVER_QUANT_API_KEY'
            }
            
            if api_name in env_key_map:
                api_config['api_key'] = os.getenv(env_key_map[api_name])
        
        return APIClientConfig(**api_config)
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration object."""
        trading_config = self._config['trading']
        return TradingConfig(**trading_config)
    
    def get_constraints_config(self) -> ConstraintsConfig:
        """Get constraints configuration object."""
        constraints_config = self._config['constraints']['hard']
        return ConstraintsConfig(**constraints_config)
    
    def get_scoring_config(self) -> ScoringConfig:
        """Get scoring configuration object."""
        scoring_config = self._config['scoring']
        return ScoringConfig(**scoring_config)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration object."""
        monitoring_config = self._config['monitoring']
        return MonitoringConfig(**monitoring_config)
    
    def get_alert_config(self) -> AlertConfig:
        """Get alert configuration object."""
        alert_config = self._config['alerts']
        return AlertConfig(**alert_config)
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        keys = key_path.split('.')
        current = self._config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_value(self, key_path: str, value: Any):
        """Set configuration value by dot-separated path."""
        keys = key_path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def reload_config(self):
        """Reload configuration from file."""
        try:
            # Check if file has been modified
            current_modified = self.config_path.stat().st_mtime
            
            if current_modified > self._last_modified:
                self._config = self._load_and_validate_config()
                self._last_modified = current_modified
                self.logger.info("Configuration reloaded successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {str(e)}")
            raise ConfigurationError(f"Configuration reload failed: {str(e)}")
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that all required API keys are present."""
        required_keys = {
            'TRADIER_API_KEY': 'Tradier API',
            'YAHOO_FINANCE_API_KEY': 'Yahoo Finance API',
            'FRED_API_KEY': 'FRED API',
            'QUIVER_QUANT_API_KEY': 'Quiver Quant API'
        }
        
        validation_results = {}
        
        for env_var, description in required_keys.items():
            api_key = os.getenv(env_var)
            validation_results[description] = bool(api_key and api_key.strip())
        
        return validation_results
    
    def get_watchlist(self) -> list:
        """Get trading watchlist."""
        return self._config['trading']['watchlist']
    
    def get_enabled_strategies(self) -> list:
        """Get enabled trading strategies."""
        return self._config['trading']['strategies']
    
    def is_market_hours_only(self) -> bool:
        """Check if trading should only occur during market hours."""
        return self._config['execution']['market_hours_only']
    
    def get_execution_interval(self) -> int:
        """Get execution interval in minutes."""
        return self._config['execution']['interval_minutes']
    
    def get_max_concurrent_trades(self) -> int:
        """Get maximum concurrent trades."""
        return self._config['execution']['max_concurrent_trades']
    
    def is_development_mode(self) -> bool:
        """Check if in development mode."""
        return self._config['application']['environment'] == 'development'
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self._config['application'].get('debug', False)
    
    def get_log_level(self) -> str:
        """Get logging level."""
        return self._config['application'].get('log_level', 'INFO')
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self._config.get('performance', {})
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self._config.get('error_handling', {})
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self._config.get('risk_management', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return deepcopy(self._config)
    
    def __repr__(self) -> str:
        """String representation of configuration manager."""
        return f"ConfigManager(config_path={self.config_path}, environment={self._config['application']['environment']})"


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> Dict[str, Any]:
    """Get configuration dictionary."""
    return get_config_manager().get_config()

def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config_manager().get_database_config()

def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return get_config_manager().get_cache_config()

def get_api_config(api_name: str) -> APIClientConfig:
    """Get API configuration."""
    return get_config_manager().get_api_config(api_name)

def get_trading_config() -> TradingConfig:
    """Get trading configuration."""
    return get_config_manager().get_trading_config()

def get_constraints_config() -> ConstraintsConfig:
    """Get constraints configuration."""
    return get_config_manager().get_constraints_config()

def get_scoring_config() -> ScoringConfig:
    """Get scoring configuration."""
    return get_config_manager().get_scoring_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config_manager().get_monitoring_config()

def get_alert_config() -> AlertConfig:
    """Get alert configuration."""
    return get_config_manager().get_alert_config()

def reload_config():
    """Reload configuration from file."""
    return get_config_manager().reload_config()

def validate_api_keys() -> Dict[str, bool]:
    """Validate API keys."""
    return get_config_manager().validate_api_keys()