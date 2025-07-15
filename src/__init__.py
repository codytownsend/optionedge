"""
Options Trading Engine - A quantitative options trade discovery system.

This package contains all the core functionality for the options trading engine,
organized into clean architectural layers:

- data: Data models, repositories, and validation
- domain: Business logic, entities, and services  
- infrastructure: External integrations (APIs, cache, monitoring)
- application: Use cases and configuration management
- presentation: Output formatting and user interfaces
"""

__version__ = "0.1.0"
__author__ = "Options Trading Team"
__email__ = "team@optionstrading.com"

# Package metadata
__title__ = "options-trading-engine"
__description__ = "A Python-based options trade discovery engine for quantitative trading"
__url__ = "https://github.com/your-org/options-trading-engine"
__license__ = "MIT"

# Version info tuple
VERSION = tuple(map(int, __version__.split('.')))

# Public API exports (will be populated as we build out the system)
__all__ = [
    "__version__",
    "VERSION",
]