#!/usr/bin/env python3
"""
Main CLI entry point for the Options Trading Engine.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from application.config.settings import get_settings
from infrastructure.monitoring.logger import setup_logging, get_logger
from infrastructure.api.tradier_client import TradierClient
from infrastructure.api.yahoo_client import YahooFinanceClient
from infrastructure.api.fred_client import FREDClient


def setup_environment():
    """Setup logging and configuration."""
    # Get settings
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        log_level=settings.system.log_level,
        enable_console=True,
        enable_file=True,
        enable_json=False
    )
    
    return settings


def test_api_connections(settings):
    """Test connections to all configured APIs."""
    logger = get_logger("cli.api_test")
    
    results = {}
    
    # Test Tradier API
    logger.info("Testing Tradier API connection...")
    try:
        tradier = TradierClient(settings.api.tradier_api_key)
        results['tradier'] = tradier.test_connection()
        logger.info(f"Tradier API: {'✓ Connected' if results['tradier'] else '✗ Failed'}")
    except Exception as e:
        results['tradier'] = False
        logger.error(f"Tradier API error: {e}")
    
    # Test Yahoo Finance API
    logger.info("Testing Yahoo Finance API connection...")
    try:
        yahoo = YahooFinanceClient(settings.api.yahoo_rapid_api_key)
        results['yahoo'] = yahoo.test_connection()
        logger.info(f"Yahoo Finance API: {'✓ Connected' if results['yahoo'] else '✗ Failed'}")
    except Exception as e:
        results['yahoo'] = False
        logger.error(f"Yahoo Finance API error: {e}")
    
    # Test FRED API
    logger.info("Testing FRED API connection...")
    try:
        fred = FREDClient(settings.api.fred_api_key)
        results['fred'] = fred.test_connection()
        logger.info(f"FRED API: {'✓ Connected' if results['fred'] else '✗ Failed'}")
    except Exception as e:
        results['fred'] = False
        logger.error(f"FRED API error: {e}")
    
    return results


def validate_configuration(settings):
    """Validate configuration settings."""
    logger = get_logger("cli.config_validation")
    
    logger.info("Validating configuration...")
    
    try:
        is_valid = settings.validate_all()
        if is_valid:
            logger.info("✓ Configuration validation passed")
        else:
            logger.error("✗ Configuration validation failed")
        return is_valid
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return False


def show_configuration(settings):
    """Display current configuration."""
    logger = get_logger("cli.config_display")
    
    print("\n" + "=" * 60)
    print("📊 CURRENT CONFIGURATION")
    print("=" * 60)
    
    # Portfolio settings
    print(f"\n💼 Portfolio Settings:")
    print(f"   NAV: ${settings.portfolio.nav:,.2f}")
    print(f"   Available Capital: ${settings.portfolio.capital_available:,.2f}")
    print(f"   Max Trades: {settings.portfolio.max_trades}")
    print(f"   Max Loss per Trade: ${settings.portfolio.max_loss_per_trade:,.2f}")
    print(f"   Min POP: {settings.portfolio.min_pop:.1%}")
    
    # Market scan settings
    print(f"\n🔍 Market Scan Settings:")
    print(f"   Universe: {settings.market_scan.scan_universe}")
    print(f"   Max Days to Expiration: {settings.market_scan.max_days_to_expiration}")
    print(f"   Min Days to Expiration: {settings.market_scan.min_days_to_expiration}")
    print(f"   Max Quote Age: {settings.market_scan.max_quote_age_minutes} minutes")
    
    # System settings
    print(f"\n⚙️ System Settings:")
    print(f"   Environment: {settings.system.environment}")
    print(f"   Log Level: {settings.system.log_level}")
    print(f"   Cache Enabled: {settings.system.enable_cache}")
    print(f"   Cache TTL: {settings.system.cache_ttl_seconds}s")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Options Trading Engine - Quantitative Options Trade Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test-apis          Test API connections
  %(prog)s --validate-config    Validate configuration
  %(prog)s --show-config        Show current configuration
  %(prog)s --scan               Run market scan (Phase 2+)
        """
    )
    
    parser.add_argument(
        "--test-apis",
        action="store_true",
        help="Test connections to all configured APIs"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true", 
        help="Validate configuration settings"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display current configuration"
    )
    
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run market scan (not implemented in Phase 1)"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Show header
    print("🚀 Options Trading Engine v0.1.0")
    print("Quantitative Options Trade Discovery System")
    print("=" * 60)
    
    try:
        # Setup environment
        settings = setup_environment()
        logger = get_logger("cli.main")
        
        logger.info("Options Trading Engine started")
        
        # Override log level if specified
        if args.log_level:
            import logging
            logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Execute requested actions
        if args.show_config:
            show_configuration(settings)
        
        if args.validate_config:
            is_valid = validate_configuration(settings)
            if not is_valid:
                sys.exit(1)
        
        if args.test_apis:
            print(f"\n🔗 Testing API Connections...")
            print("-" * 40)
            results = test_api_connections(settings)
            
            all_connected = all(results.values())
            print(f"\n{'✓ All APIs connected' if all_connected else '⚠ Some APIs failed'}")
            
            if not all_connected:
                print("\n💡 Tips:")
                print("- Check your API keys in .env file")
                print("- Verify network connectivity")
                print("- Check API rate limits")
        
        if args.scan:
            print(f"\n❌ Market scanning not implemented yet")
            print("This feature will be available in Phase 2")
            print("Current Phase 1 includes:")
            print("- ✓ Configuration management")
            print("- ✓ API integration framework")
            print("- ✓ Data models")
            print("- ✓ Logging and monitoring")
            print("- ✓ Caching system")
        
        # If no specific action, show help
        if not any([args.test_apis, args.validate_config, args.show_config, args.scan]):
            parser.print_help()
            print(f"\n💡 Try: {parser.prog} --test-apis --show-config")
        
        logger.info("Options Trading Engine completed")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()