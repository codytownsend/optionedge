#!/usr/bin/env python3
"""
Project initialization script for Options Trading Engine.
Creates directory structure, initializes configuration, and validates setup.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any

def create_directory_structure(base_path: Path) -> None:
    """Create the complete project directory structure."""
    
    directories = [
        # Source directories
        "src",
        "src/data",
        "src/data/models",
        "src/data/repositories", 
        "src/data/validators",
        "src/domain",
        "src/domain/entities",
        "src/domain/services",
        "src/domain/value_objects",
        "src/infrastructure",
        "src/infrastructure/api",
        "src/infrastructure/cache",
        "src/infrastructure/monitoring",
        "src/application",
        "src/application/use_cases",
        "src/application/config",
        "src/presentation",
        "src/presentation/formatters",
        "src/presentation/cli",
        
        # Test directories
        "tests",
        "tests/unit",
        "tests/unit/test_models",
        "tests/unit/test_services",
        "tests/unit/test_api",
        "tests/integration",
        "tests/integration/test_api_integration",
        "tests/end_to_end",
        
        # Configuration and docs
        "config",
        "docs",
        "scripts",
        
        # Runtime directories
        "logs",
        "cache",
        "data",
        "temp",
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/") or directory.startswith("tests/"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print(f"✓ Created {len(directories)} directories")


def create_init_files(base_path: Path) -> None:
    """Create __init__.py files with proper imports."""
    
    init_files = {
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/data/models/__init__.py": '''"""Data models for the options trading engine."""

from .options import OptionQuote, OptionsChain, OptionType, Greeks
from .market_data import StockQuote, TechnicalIndicators, FundamentalData, EconomicIndicator
from .trades import StrategyDefinition, TradeCandidate, Portfolio

__all__ = [
    "OptionQuote", "OptionsChain", "OptionType", "Greeks",
    "StockQuote", "TechnicalIndicators", "FundamentalData", "EconomicIndicator", 
    "StrategyDefinition", "TradeCandidate", "Portfolio"
]
''',
        "src/infrastructure/__init__.py": "",
        "src/infrastructure/api/__init__.py": '''"""API clients for external data sources."""

from .tradier_client import TradierClient
from .yahoo_client import YahooFinanceClient
from .fred_client import FREDClient

__all__ = ["TradierClient", "YahooFinanceClient", "FREDClient"]
''',
        "src/application/__init__.py": "",
        "src/application/config/__init__.py": '''"""Configuration management."""

from .settings import get_settings, AppSettings

__all__ = ["get_settings", "AppSettings"]
''',
        "tests/__init__.py": "",
    }
    
    print("Creating __init__.py files...")
    for file_path, content in init_files.items():
        full_path = base_path / file_path
        full_path.write_text(content)
    
    print(f"✓ Created {len(init_files)} __init__.py files")


def create_env_file(base_path: Path) -> None:
    """Create .env file from .env.example if it doesn't exist."""
    
    env_example = base_path / ".env.example"
    env_file = base_path / ".env"
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from .env.example")
        print("  → Please edit .env with your actual API keys")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        print("⚠ .env.example not found - create it manually")


def validate_python_version() -> bool:
    """Validate Python version meets requirements."""
    
    required_version = (3, 9)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        print(f"✓ Python {'.'.join(map(str, current_version))} meets requirements")
        return True
    else:
        print(f"✗ Python {'.'.join(map(str, current_version))} < {'.'.join(map(str, required_version))}")
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    
    required_packages = [
        "requests", "pandas", "numpy", "scipy", "pydantic", 
        "python-dotenv", "tenacity", "structlog", "colorama"
    ]
    
    optional_packages = [
        "yfinance", "fredapi", "quantlib"
    ]
    
    results = {}
    
    print("Checking required dependencies...")
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            results[package] = True
            print(f"✓ {package}")
        except ImportError:
            results[package] = False
            print(f"✗ {package} - please install")
    
    print("\nChecking optional dependencies...")
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            results[package] = True
            print(f"✓ {package}")
        except ImportError:
            results[package] = False
            print(f"- {package} - optional but recommended")
    
    return results


def create_sample_config(base_path: Path) -> None:
    """Create sample configuration files."""
    
    sample_settings = '''# Sample configuration for Options Trading Engine
# Copy this file to config/settings.yaml and customize

portfolio:
  nav: 100000.0              # Net Asset Value
  capital_available: 10000.0  # Available capital for new trades
  max_trades: 5              # Maximum concurrent trades
  max_loss_per_trade: 500.0  # Maximum loss per trade
  min_pop: 0.65             # Minimum Probability of Profit
  min_credit_to_max_loss: 0.33  # Minimum credit-to-max-loss ratio

market_scan:
  scan_universe: "SP500"     # Universe to scan: SP500, NASDAQ100, or custom
  max_days_to_expiration: 45 # Maximum days to expiration
  min_days_to_expiration: 7  # Minimum days to expiration
  max_quote_age_minutes: 10  # Maximum quote age in minutes

system:
  log_level: "INFO"          # Logging level
  cache_ttl_seconds: 300     # Cache TTL in seconds
  enable_cache: true         # Enable caching
'''
    
    config_file = base_path / "config" / "settings.yaml"
    if not config_file.exists():
        config_file.write_text(sample_settings)
        print("✓ Created sample config/settings.yaml")
    else:
        print("✓ config/settings.yaml already exists")


def create_readme(base_path: Path) -> None:
    """Create a basic README file."""
    
    readme_content = '''# Options Trading Engine

A Python-based options trade discovery engine that simulates the research workflow of a quantitative options desk.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the engine:**
   ```bash
   python -m src.presentation.cli.main
   ```

## Project Structure

- `src/` - Main source code
  - `data/` - Data models and repositories
  - `domain/` - Business logic and entities
  - `infrastructure/` - External integrations (APIs, cache)
  - `application/` - Use cases and configuration
  - `presentation/` - Output formatting and CLI
- `tests/` - Test suite
- `config/` - Configuration files
- `docs/` - Documentation

## Configuration

Edit `config/settings.yaml` to customize:
- Portfolio parameters (NAV, capital, risk limits)
- Market scanning preferences
- System settings (logging, caching)

## API Keys Required

- **Tradier API**: For options chain data
- **Yahoo Finance/RapidAPI**: For fundamental data
- **FRED API**: For economic indicators
- **QuiverQuant** (optional): For sentiment data

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src tests
isort src tests
```

Type checking:
```bash
mypy src
```

## License

MIT License - see LICENSE file for details.
'''
    
    readme_file = base_path / "README.md"
    if not readme_file.exists():
        readme_file.write_text(readme_content)
        print("✓ Created README.md")
    else:
        print("✓ README.md already exists")


def main():
    """Main initialization function."""
    
    print("🚀 Initializing Options Trading Engine Project")
    print("=" * 50)
    
    # Get project root
    base_path = Path.cwd()
    print(f"Project path: {base_path}")
    
    # Validate Python version
    if not validate_python_version():
        print("\n❌ Python version check failed")
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure(base_path)
    
    # Create init files
    create_init_files(base_path)
    
    # Create environment file
    create_env_file(base_path)
    
    # Create sample configuration
    create_sample_config(base_path)
    
    # Create README
    create_readme(base_path)
    
    # Check dependencies
    print("\n" + "=" * 50)
    dependency_results = check_dependencies()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Project initialization complete!")
    
    missing_deps = [pkg for pkg, available in dependency_results.items() if not available]
    if missing_deps:
        print(f"\n⚠ Install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    print("\n📋 Next steps:")
    print("1. Edit .env with your API keys")
    print("2. Customize config/settings.yaml")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run tests: pytest")
    print("5. Start developing! 🚀")


if __name__ == "__main__":
    main()