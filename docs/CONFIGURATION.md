# Configuration Guide

This document describes how to configure the Options Trading Engine for different environments and use cases.

## Configuration Files

The engine uses multiple configuration files located in the `config/` directory:

- `settings.yaml` - Main configuration file
- `settings.schema.json` - JSON schema for validation
- `api_endpoints.yaml` - API endpoint configurations
- `.env` - Environment variables (copy from `.env.example`)

## Environment Variables

### Required Variables

```bash
# API Authentication
TRADIER_API_KEY=your_tradier_api_key_here
YAHOO_RAPIDAPI_KEY=your_rapidapi_key_here
FRED_API_KEY=your_fred_api_key_here
QUIVER_API_KEY=your_quiver_api_key_here

# Application Environment
APP_ENVIRONMENT=development|staging|production
```

### Optional Variables

```bash
# Application Configuration
APP_NAME=options-engine
APP_VERSION=0.1.0
APP_DEBUG=true

# Trading Parameters
TRADING_MIN_POP=0.65
TRADING_MIN_CREDIT_TO_MAX_LOSS=0.33
TRADING_MAX_LOSS_PER_TRADE=500
TRADING_DEFAULT_NAV=100000

# Cache Configuration
CACHE_TYPE=memory|redis|null
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=10000

# Logging
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
LOG_FILE_PATH=logs/options_engine.log
```

## Main Configuration (settings.yaml)

### Application Settings

```yaml
application:
  name: options-engine
  version: 0.1.0
  environment: development
  debug: true
```

### API Configuration

```yaml
api:
  tradier:
    base_url: https://api.tradier.com
    timeout: 30
    rate_limit:
      requests_per_minute: 120
      burst_size: 5
  
  yahoo:
    base_url: https://query1.finance.yahoo.com
    timeout: 30
  
  fred:
    base_url: https://api.stlouisfed.org
    timeout: 30
  
  quiver:
    base_url: https://api.quiverquant.com
    timeout: 30
```

### Cache Configuration

```yaml
cache:
  type: memory  # memory, redis, null
  default_ttl: 3600  # seconds
  max_size: 10000
  
  # Redis configuration (if type: redis)
  redis:
    host: localhost
    port: 6379
    db: 0
    password: ""
```

### Logging Configuration

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  file:
    enabled: true
    path: logs/options_engine.log
    max_size: 10  # MB
    backup_count: 5
  
  console:
    enabled: true
    colors: true
```

### Trading Configuration

```yaml
trading:
  constraints:
    min_probability_of_profit: 0.65
    min_credit_to_max_loss: 0.33
    max_loss_per_trade: 500
    max_quote_age_minutes: 10
    max_delta_exposure: 0.30
    min_vega_exposure: -0.05
    max_trades_per_sector: 2
    min_days_to_expiration: 7
    max_days_to_expiration: 45
  
  liquidity:
    min_option_volume: 10
    min_open_interest: 100
    max_bid_ask_spread_pct: 0.50
  
  portfolio:
    default_nav: 100000
    max_capital_per_trade: 0.05  # 5% of NAV
```

### Data Configuration

```yaml
data:
  refresh_intervals:
    options_chains: 300  # 5 minutes
    market_data: 60      # 1 minute
    fundamentals: 3600   # 1 hour
    economic_data: 86400 # 24 hours
  
  symbols:
    broad_market:
      - SPY
      - QQQ
      - IWM
      - AAPL
      - MSFT
      - GOOGL
      - AMZN
      - TSLA
      - META
      - NVDA
    
    etf_flows:
      - SPY
      - QQQ
      - IWM
      - XLF
      - XLK
      - XLE
```

### Monitoring Configuration

```yaml
monitoring:
  metrics:
    enabled: true
    collection_interval: 60
  
  alerts:
    enabled: false
    thresholds:
      api_error_rate: 0.05
      processing_time: 30
```

## Environment-Specific Configuration

### Development Environment

```yaml
application:
  environment: development
  debug: true

logging:
  level: DEBUG
  console:
    enabled: true
    colors: true

api:
  tradier:
    base_url: https://sandbox.tradier.com  # Use sandbox
```

### Production Environment

```yaml
application:
  environment: production
  debug: false

logging:
  level: INFO
  file:
    enabled: true
    path: /var/log/options_engine/app.log

monitoring:
  metrics:
    enabled: true
  alerts:
    enabled: true
```

### Testing Environment

```yaml
application:
  environment: testing
  debug: true

cache:
  type: null  # No caching in tests

api:
  # Use mock APIs or test endpoints
  tradier:
    base_url: http://localhost:8080/mock/tradier
```

## Configuration Loading

The application loads configuration in this order (later values override earlier ones):

1. Default values from schema
2. `config/settings.yaml`
3. Environment-specific file (e.g., `config/settings.production.yaml`)
4. Environment variables
5. Command-line arguments

### Loading in Code

```python
from src.application.config.settings import Settings

# Load configuration
settings = Settings.load()

# Access nested values
api_key = settings.api.tradier.api_key
min_pop = settings.trading.constraints.min_probability_of_profit
```

### Validation

Configuration is automatically validated against the JSON schema:

```python
from src.application.config.validators import ConfigValidator

validator = ConfigValidator()
errors = validator.validate(config_dict)

if errors:
    raise ConfigurationError(f"Invalid configuration: {errors}")
```

## API Endpoint Configuration

The `api_endpoints.yaml` file defines all external API endpoints:

```yaml
tradier:
  base_url: "https://api.tradier.com"
  endpoints:
    market_data:
      quotes: "/v1/markets/quotes"
      options_chains: "/v1/markets/options/chains"
    fundamentals:
      company: "/v1/markets/fundamentals/company"
  rate_limits:
    requests_per_minute: 120
    burst_size: 5
```

## Security Configuration

### API Keys

Store API keys in environment variables, not in configuration files:

```bash
# .env file
TRADIER_API_KEY=your_actual_key_here
YAHOO_RAPIDAPI_KEY=your_actual_key_here
```

### Secrets Management

For production deployments, use proper secrets management:

```yaml
# Using environment variables
api:
  tradier:
    api_key: ${TRADIER_API_KEY}
    
# Using secret files
api:
  tradier:
    api_key_file: /secrets/tradier_api_key
```

## Performance Tuning

### Cache Settings

```yaml
cache:
  type: redis  # For better performance than memory
  default_ttl: 3600
  max_size: 50000  # Increase for high-volume usage
  
  redis:
    host: redis-cluster.internal
    port: 6379
    db: 0
```

### API Rate Limiting

```yaml
api:
  tradier:
    rate_limit:
      requests_per_minute: 120
      burst_size: 10  # Allow burst traffic
      
  # Connection pooling
  connection_pool:
    max_connections: 20
    max_keepalive_connections: 5
```

### Data Refresh Intervals

```yaml
data:
  refresh_intervals:
    # Aggressive caching for development
    options_chains: 600  # 10 minutes
    market_data: 120     # 2 minutes
    
    # Or frequent updates for production
    options_chains: 60   # 1 minute
    market_data: 15      # 15 seconds
```

## Monitoring and Alerting

### Metrics Configuration

```yaml
monitoring:
  metrics:
    enabled: true
    collection_interval: 30  # seconds
    
    # Export metrics
    exporters:
      prometheus:
        enabled: true
        port: 9090
      
      cloudwatch:
        enabled: false
        region: us-east-1
```

### Alert Configuration

```yaml
monitoring:
  alerts:
    enabled: true
    
    # Alert channels
    channels:
      slack:
        webhook_url: ${SLACK_WEBHOOK_URL}
      
      email:
        smtp_host: smtp.gmail.com
        smtp_port: 587
        username: ${EMAIL_USERNAME}
        password: ${EMAIL_PASSWORD}
    
    # Alert thresholds
    thresholds:
      api_error_rate: 0.05      # 5% error rate
      processing_time: 30       # 30 seconds
      memory_usage: 0.85        # 85% memory
      disk_usage: 0.90          # 90% disk
```

## Custom Configuration

### Adding New Settings

1. Update the JSON schema (`config/settings.schema.json`)
2. Add default values to `config/settings.yaml`
3. Update the settings model if using typed configuration

### Custom Validators

```python
from src.application.config.validators import register_validator

@register_validator("trading.constraints.min_probability_of_profit")
def validate_probability(value):
    if not 0 <= value <= 1:
        raise ValueError("Probability must be between 0 and 1")
    return value
```

### Dynamic Configuration

```python
from src.application.config.settings import Settings

# Reload configuration at runtime
Settings.reload()

# Watch for configuration changes
Settings.watch_for_changes()
```

## Troubleshooting

### Common Configuration Issues

1. **Invalid API keys**: Check environment variables
2. **Rate limit errors**: Adjust rate limiting settings
3. **Cache issues**: Clear cache or change cache type
4. **Timeout errors**: Increase timeout values
5. **Memory issues**: Reduce cache size or data retention

### Validation Errors

```bash
# Validate configuration
python -m src.application.config.validators config/settings.yaml

# Check schema compliance
jsonschema -i config/settings.yaml config/settings.schema.json
```

### Debug Configuration

```python
from src.application.config.settings import Settings

# Print current configuration
settings = Settings.load()
print(settings.to_dict())

# Check configuration sources
print(settings.get_sources())
```

## Best Practices

1. **Use environment variables** for secrets and environment-specific values
2. **Validate configuration** on startup
3. **Document all settings** with descriptions and examples
4. **Use typed configuration** for better IDE support
5. **Monitor configuration changes** in production
6. **Keep configuration DRY** using includes and references
7. **Version configuration** alongside code
8. **Test configuration** in all environments