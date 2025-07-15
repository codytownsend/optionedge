# Options Engine API Documentation

This document describes the API interfaces and usage patterns for the Options Trading Engine.

## Overview

The Options Engine provides a Python API for quantitative options trading analysis. It generates trade recommendations based on configurable criteria and risk parameters.

## Core Components

### Trade Generation API

#### `TradeGenerationUseCase`

The main entry point for generating trade recommendations.

```python
from src.application.use_cases.generate_trades import TradeGenerationUseCase, TradeGenerationRequest

# Create request
request = TradeGenerationRequest(
    nav=Decimal("100000"),
    max_trades=5,
    scan_type="broad_market"
)

# Generate trades
response = await use_case.execute(request)
```

##### Request Parameters

- `nav`: Portfolio net asset value (Decimal)
- `max_trades`: Maximum number of trades to return (int, default: 5)
- `scan_type`: Type of scan ("broad_market", "portfolio", "custom")
- `symbols`: List of symbols to scan (optional)
- `custom_criteria`: Custom filter criteria (optional)

##### Response Format

```python
class TradeGenerationResponse:
    selected_trades: List[TradeCandidate]
    rejected_trades: List[TradeCandidate]
    portfolio_impact: Dict[str, Any]
    generation_stats: Dict[str, Any]
```

### Market Scanning API

#### `MarketScanUseCase`

Scans market for trade opportunities.

```python
from src.application.use_cases.scan_market import MarketScanUseCase, MarketScanRequest

# Create scan request
request = MarketScanRequest(
    scan_type="broad_market",
    strategy_types=[StrategyType.PUT_CREDIT_SPREAD, StrategyType.IRON_CONDOR]
)

# Execute scan
response = await use_case.execute(request)
```

### Data Models

#### `TradeCandidate`

Represents a potential trade opportunity.

```python
class TradeCandidate:
    strategy: StrategyDefinition
    model_score: Optional[float]
    momentum_z_score: Optional[float]
    flow_z_score: Optional[float]
    iv_rank: Optional[float]
    quote_age_minutes: Optional[float]
    liquidity_score: Optional[float]
    capital_required: Optional[Decimal]
    sector: Optional[str]
    thesis: Optional[str]
    rank: Optional[int]
    selected: bool
```

#### `StrategyDefinition`

Defines an options strategy.

```python
class StrategyDefinition:
    strategy_type: StrategyType
    underlying: str
    legs: List[TradeLeg]
    net_premium: Optional[Decimal]
    max_profit: Optional[Decimal]
    max_loss: Optional[Decimal]
    probability_of_profit: Optional[float]
    breakeven_points: List[Decimal]
    # Greeks
    net_delta: Optional[float]
    net_gamma: Optional[float]
    net_theta: Optional[float]
    net_vega: Optional[float]
    net_rho: Optional[float]
```

#### `TradeLeg`

Individual leg of an options strategy.

```python
class TradeLeg:
    option: OptionQuote
    direction: TradeDirection  # LONG or SHORT
    quantity: int
```

### Configuration API

#### Settings Management

```python
from src.application.config.settings import Settings

# Load settings
settings = Settings.load()

# Access configuration
api_key = settings.api.tradier.api_key
min_pop = settings.trading.constraints.min_probability_of_profit
```

#### Filter Criteria

```python
from src.data.models.trades import TradeFilterCriteria

criteria = TradeFilterCriteria(
    min_probability_of_profit=0.65,
    min_credit_to_max_loss=0.33,
    max_loss_per_trade=Decimal("500"),
    max_quote_age_minutes=10.0,
    max_delta_exposure=0.30,
    min_vega_exposure=-0.05,
    max_trades_per_sector=2
)
```

### Formatting API

#### Console Output

```python
from src.presentation.formatters.console_formatter import ConsoleFormatter

formatter = ConsoleFormatter(use_colors=True)

# Format trade results
output = formatter.format_trade_results(
    trades=selected_trades,
    portfolio_impact=portfolio_impact,
    stats=generation_stats
)

print(output)
```

#### Table Formatting

```python
from src.presentation.formatters.table_formatter import TradeTableFormatter

formatter = TradeTableFormatter(max_width=120)

# Format trades as table
table = formatter.format_trades(trades, title="Top Recommendations")
print(table)
```

### Infrastructure APIs

#### Caching

```python
from src.infrastructure.cache.base_cache import CacheManager

# Get cache instance
cache = CacheManager().get_cache("default")

# Cache operations
await cache.set("key", data, ttl=3600)
data = await cache.get("key")
```

#### Metrics

```python
from src.infrastructure.monitoring.metrics import app_metrics

# Record metrics
app_metrics.record_api_call("tradier", duration=1.5, success=True)
app_metrics.record_strategy_generation(count=50, duration=2.3)
```

#### API Clients

```python
from src.infrastructure.api.tradier_client import TradierClient

# Create client
client = TradierClient(api_key="your_key")

# Get options chain
chain = await client.get_options_chain("AAPL")
```

## Error Handling

All APIs use standard Python exceptions with custom exception types:

```python
from src.domain.exceptions import (
    ValidationError,
    APIError,
    ConfigurationError,
    InsufficientDataError
)

try:
    response = await use_case.execute(request)
except ValidationError as e:
    print(f"Validation error: {e}")
except APIError as e:
    print(f"API error: {e}")
```

## Rate Limiting

APIs implement rate limiting based on configuration:

```python
# Rate limits are configured per API
tradier_limits = {
    "requests_per_minute": 120,
    "burst_size": 5
}

# Clients automatically handle rate limiting
# with exponential backoff
```

## Authentication

APIs require authentication tokens:

```python
# Environment variables
TRADIER_API_KEY=your_tradier_key
YAHOO_RAPIDAPI_KEY=your_yahoo_key
FRED_API_KEY=your_fred_key
QUIVER_API_KEY=your_quiver_key
```

## Response Formats

### Success Response

```json
{
    "success": true,
    "data": {
        "selected_trades": [...],
        "portfolio_impact": {...},
        "generation_stats": {...}
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Response

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid parameter: nav must be positive",
        "details": {...}
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## Usage Examples

### Basic Trade Generation

```python
import asyncio
from decimal import Decimal
from src.application.use_cases.generate_trades import TradeGenerationUseCase, TradeGenerationRequest

async def main():
    # Initialize use case with dependencies
    use_case = TradeGenerationUseCase(
        market_scan_use_case=market_scan_use_case,
        scoring_engine=scoring_engine,
        risk_calculator=risk_calculator,
        portfolio_manager=portfolio_manager
    )
    
    # Create request
    request = TradeGenerationRequest(
        nav=Decimal("100000"),
        max_trades=5,
        scan_type="broad_market"
    )
    
    # Generate trades
    response = await use_case.execute(request)
    
    # Display results
    formatter = ConsoleFormatter()
    output = formatter.format_trade_results(
        trades=response.selected_trades,
        portfolio_impact=response.portfolio_impact,
        stats=response.generation_stats
    )
    print(output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Symbol Scanning

```python
from src.application.use_cases.scan_market import MarketScanRequest
from src.data.models.trades import TradeFilterCriteria

# Custom scan with specific symbols
request = MarketScanRequest(
    scan_type="custom",
    symbols=["AAPL", "MSFT", "GOOGL"],
    strategy_types=[StrategyType.PUT_CREDIT_SPREAD],
    filter_criteria=TradeFilterCriteria(
        min_probability_of_profit=0.70,
        max_loss_per_trade=Decimal("300")
    )
)
```

### Portfolio Risk Management

```python
from src.data.models.portfolios import Portfolio

# Create portfolio with existing positions
portfolio = Portfolio(
    total_delta=0.15,
    total_vega=-0.02,
    total_value=Decimal("95000")
)

# Generate trades considering portfolio
request = TradeGenerationRequest(
    nav=Decimal("100000"),
    portfolio=portfolio,
    max_trades=3
)
```

## Best Practices

1. **Always validate input parameters** before API calls
2. **Use appropriate timeouts** for API requests
3. **Implement proper error handling** with retries
4. **Monitor API usage** against rate limits
5. **Cache data appropriately** to reduce API calls
6. **Use structured logging** for debugging
7. **Validate market data age** before trading decisions

## Performance Considerations

- API calls are asynchronous for better performance
- Caching reduces redundant API requests
- Batch operations when possible
- Monitor memory usage with large datasets
- Use connection pooling for HTTP clients

## Security

- Never log API keys or sensitive data
- Use environment variables for configuration
- Validate all inputs to prevent injection attacks
- Implement rate limiting to prevent abuse
- Use HTTPS for all API communications