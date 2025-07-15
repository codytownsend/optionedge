# Development Guide

This guide covers setting up a development environment, coding standards, testing procedures, and contribution guidelines for the Options Trading Engine.

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)
- API keys for external services (Tradier, Yahoo Finance, FRED, QuiverQuant)

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd options_engine
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Create directories**
   ```bash
   mkdir -p logs cache temp
   ```

6. **Verify installation**
   ```bash
   python -m pytest tests/ -v
   python -c "from src import __version__; print(__version__)"
   ```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python
- Pylance
- Python Docstring Generator
- autoDocstring
- Black Formatter
- isort

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter: Settings → Project → Python Interpreter
3. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest
4. Configure code style: Settings → Editor → Code Style → Python → Black

## Project Structure

```
options_engine/
├── src/                     # Source code
│   ├── data/               # Data layer
│   ├── domain/             # Business logic
│   ├── infrastructure/     # External integrations
│   ├── application/        # Use cases and config
│   └── presentation/       # Output formatting
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── end_to_end/        # E2E tests
├── config/                # Configuration files
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── logs/                  # Log files
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specific guidelines:

- **Line length**: 88 characters (Black formatter default)
- **Import ordering**: Use isort with Black-compatible settings
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all public APIs
- **Error handling**: Use specific exception types

### Code Formatting

We use Black for code formatting:

```bash
# Format all code
black src/ tests/

# Check formatting
black --check src/ tests/
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### Import Sorting

Use isort for import organization:

```bash
# Sort imports
isort src/ tests/

# Check import order
isort --check-only src/ tests/
```

Configuration in `pyproject.toml`:
```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]
```

### Type Checking

Use mypy for static type checking:

```bash
# Type check
mypy src/

# Type check with strict mode
mypy --strict src/
```

Configuration in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true
strict_optional = true
```

### Linting

Use pylint for additional code quality checks:

```bash
# Lint code
pylint src/

# Lint with specific config
pylint --rcfile=.pylintrc src/
```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def calculate_strategy_greeks(strategy: StrategyDefinition) -> Greeks:
    """Calculate net Greeks for an options strategy.
    
    This function computes the aggregate Greeks (delta, gamma, theta, vega, rho)
    for all legs in an options strategy, taking into account the direction
    and quantity of each leg.
    
    Args:
        strategy: The options strategy containing legs with individual Greeks.
        
    Returns:
        Greeks object containing the net Greeks for the entire strategy.
        
    Raises:
        ValueError: If the strategy has no legs or invalid Greeks data.
        
    Example:
        >>> strategy = create_iron_condor("AAPL", 150, 155, 145, 160)
        >>> net_greeks = calculate_strategy_greeks(strategy)
        >>> print(f"Net delta: {net_greeks.delta}")
        Net delta: 0.05
    """
```

### API Documentation

Document all public APIs with:
- Purpose and behavior
- Parameter descriptions with types
- Return value descriptions
- Exception conditions
- Usage examples
- Performance considerations

### Code Comments

Use comments sparingly and focus on:
- **Why** the code does something (not what)
- Complex business logic explanations
- Performance optimizations
- Temporary workarounds with TODO comments

```python
# TODO: Replace with more sophisticated volatility model once we have
# sufficient historical data (target: Q2 2024)
implied_vol = self._simple_volatility_estimate(option_chain)

# Use Black-Scholes approximation for American options to avoid
# computational overhead during real-time scanning
greeks = self._bs_greeks(option, implied_vol)
```

## Testing

### Test Organization

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test component interactions and external APIs
- **End-to-end tests**: Test complete workflows from input to output

### Writing Tests

#### Unit Test Example

```python
import pytest
from decimal import Decimal
from src.data.models.options import OptionQuote, Greeks

class TestOptionQuote:
    def test_mid_price_calculation(self):
        """Test that mid price is calculated correctly from bid/ask."""
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
    
    def test_liquidity_check_with_valid_option(self):
        """Test liquidity check for a liquid option."""
        quote = OptionQuote(
            # ... setup quote with good liquidity metrics
        )
        
        assert quote.is_liquid(min_volume=10, min_oi=100, max_spread_pct=0.5)
    
    @pytest.mark.parametrize("volume,expected", [
        (5, False),    # Below minimum
        (10, True),    # At minimum
        (100, True),   # Above minimum
    ])
    def test_volume_requirements(self, volume, expected):
        """Test volume requirement validation."""
        quote = self._create_test_quote(volume=volume)
        assert quote.is_liquid(min_volume=10) == expected
```

#### Integration Test Example

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.application.use_cases.generate_trades import TradeGenerationUseCase

class TestTradeGenerationIntegration:
    @pytest.mark.asyncio
    @patch('src.infrastructure.api.tradier_client.TradierClient')
    async def test_complete_trade_generation_workflow(self, mock_client):
        """Test complete workflow from market scan to trade selection."""
        # Setup mocks
        mock_client.get_options_chain.return_value = self._mock_options_chain()
        mock_client.get_stock_quote.return_value = self._mock_stock_quote()
        
        # Create use case
        use_case = TradeGenerationUseCase(...)
        
        # Execute
        request = TradeGenerationRequest(nav=Decimal("100000"))
        response = await use_case.execute(request)
        
        # Verify
        assert len(response.selected_trades) <= 5
        assert all(trade.probability_of_profit >= 0.65 for trade in response.selected_trades)
```

### Test Configuration

Configuration in `pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
    -ra
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    asyncio: marks tests as async
```

### Test Coverage

Monitor test coverage:

```bash
# Run tests with coverage
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# Enforce minimum coverage
pytest --cov=src --cov-fail-under=80 tests/
```

Target coverage levels:
- **Unit tests**: 90%+ for core business logic
- **Integration tests**: 70%+ for API interactions
- **Overall**: 80%+ across the entire codebase

### Mocking External Dependencies

```python
from unittest.mock import AsyncMock, patch, MagicMock

# Mock API clients
@patch('src.infrastructure.api.tradier_client.TradierClient')
async def test_with_mocked_api(mock_client):
    mock_client.get_options_chain.return_value = mock_options_chain
    # ... test logic

# Mock database/cache
@patch('src.infrastructure.cache.memory_cache.MemoryCache')
async def test_with_mocked_cache(mock_cache):
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    # ... test logic

# Mock time-dependent functions
@patch('src.data.models.options.datetime')
def test_with_mocked_time(mock_datetime):
    mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)
    # ... test logic
```

## Development Workflow

### Git Workflow

We use Git Flow branching model:

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: Feature development branches
- **hotfix/***: Critical production fixes
- **release/***: Release preparation branches

#### Feature Development

```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/new-strategy-generator

# Work on feature
git add .
git commit -m "Add iron butterfly strategy generator"

# Push feature branch
git push origin feature/new-strategy-generator

# Create pull request to develop branch
```

#### Commit Messages

Use conventional commit format:

```
type(scope): short description

Longer description if needed

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

Examples:
```
feat(trading): add iron butterfly strategy generator

fix(api): handle rate limiting for Tradier API calls

docs(readme): update installation instructions

test(models): add comprehensive tests for OptionQuote model
```

### Code Review Process

#### Pull Request Checklist

- [ ] Code follows style guidelines (Black, isort, pylint)
- [ ] All tests pass
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No sensitive data in commits
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

#### Review Guidelines

**For Authors:**
- Keep PRs small and focused
- Write clear descriptions
- Include context and reasoning
- Respond promptly to feedback
- Update documentation

**For Reviewers:**
- Review within 24 hours
- Focus on logic, design, and maintainability
- Be constructive and specific
- Ask questions for clarity
- Approve when satisfied

### Pre-commit Hooks

Install pre-commit hooks to catch issues early:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Configuration in `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
```

## Debugging

### Logging

Use structured logging throughout the application:

```python
import logging
from src.infrastructure.monitoring.logger import get_logger

logger = get_logger(__name__)

def process_options_chain(symbol: str, chain: OptionsChain) -> List[StrategyDefinition]:
    logger.info("Processing options chain", extra={
        "symbol": symbol,
        "chain_size": len(chain.options),
        "timestamp": datetime.utcnow()
    })
    
    try:
        strategies = self._generate_strategies(chain)
        logger.info("Generated strategies", extra={
            "symbol": symbol,
            "strategy_count": len(strategies)
        })
        return strategies
    except Exception as e:
        logger.error("Failed to process options chain", extra={
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, exc_info=True)
        raise
```

### Debug Configuration

Development settings for debugging:

```yaml
# config/settings.yaml (development)
application:
  debug: true

logging:
  level: DEBUG
  console:
    enabled: true
    colors: true

api:
  # Use longer timeouts for debugging
  default_timeout: 60
  
  # Enable detailed request logging
  log_requests: true
  log_responses: true

cache:
  # Disable caching to see fresh data
  type: null
```

### Performance Profiling

Use built-in profiling tools:

```python
# Profile a function
import cProfile
import pstats

def profile_trade_generation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    response = await trade_generator.execute(request)
    
    profiler.disable()
    
    # Save profile data
    profiler.dump_stats('trade_generation.prof')
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)
```

Memory profiling:

```python
# Use memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass

# Run with: python -m memory_profiler script.py
```

## Performance Optimization

### General Guidelines

1. **Profile before optimizing** - measure performance bottlenecks
2. **Cache appropriately** - balance memory usage and API calls
3. **Use async/await** for I/O-bound operations
4. **Batch API requests** when possible
5. **Optimize data structures** for frequent operations
6. **Monitor memory usage** with large datasets

### Common Optimizations

#### Caching Strategy

```python
from functools import lru_cache
from src.infrastructure.cache.base_cache import cache_manager

# Memory caching for expensive calculations
@lru_cache(maxsize=1000)
def calculate_black_scholes_greeks(s, k, t, r, sigma, option_type):
    # Expensive calculation
    return greeks

# Distributed caching for API responses
async def get_options_chain_cached(symbol: str) -> OptionsChain:
    cache = cache_manager.get_cache("api_responses")
    
    cached_chain = await cache.get(f"options_chain:{symbol}")
    if cached_chain:
        return cached_chain
    
    chain = await self.api_client.get_options_chain(symbol)
    await cache.set(f"options_chain:{symbol}", chain, ttl=300)
    
    return chain
```

#### Async Batch Processing

```python
import asyncio
from typing import List

async def process_symbols_batch(symbols: List[str]) -> List[OptionsChain]:
    """Process multiple symbols concurrently."""
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    
    async def process_symbol(symbol: str) -> OptionsChain:
        async with semaphore:
            return await self.get_options_chain(symbol)
    
    tasks = [process_symbol(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    return [result for result in results if not isinstance(result, Exception)]
```

## Contributing

### Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Contribution Guidelines

- Follow the existing code style and conventions
- Write comprehensive tests for new features
- Update documentation for API changes
- Keep commits focused and atomic
- Write clear commit messages
- Respond to code review feedback promptly

### Release Process

1. **Feature freeze** on develop branch
2. **Create release branch** from develop
3. **Testing and bug fixes** on release branch
4. **Update version numbers** and changelog
5. **Merge to main** and tag release
6. **Deploy to production**
7. **Merge back to develop**

For more details, see the project's contributing guidelines and code of conduct.