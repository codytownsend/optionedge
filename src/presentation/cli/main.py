"""
Main CLI interface for the Options Trading Engine.
"""

import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from ...application.config.settings import ConfigManager
from ...application.use_cases.generate_trades import GenerateTradesUseCase
from ...application.use_cases.scan_market import ScanMarketUseCase
from ...infrastructure.api import TradierClient, YahooClient, FredClient, QuiverClient
from ...infrastructure.cache import CacheManager
from ...infrastructure.error_handling import OptionsEngineError
from ...data.repositories.market_repo import MarketDataRepository
from ...data.repositories.options_repo import OptionsRepository
from ...domain.services.scoring_engine import ScoringEngine
from ...domain.services.risk_calculator import RiskCalculator
from ...domain.services.strategy_generation_service import StrategyGenerationService
from ...domain.services.constraint_engine import ConstraintEngine
from ...domain.services.trade_selector import TradeSelector
from ..formatters.trade_formatter import TradeFormatter
from ..formatters.console_formatter import ConsoleFormatter
from ..formatters.table_formatter import TableFormatter

logger = logging.getLogger(__name__)


class OptionsEngineCLI:
    """
    Main CLI interface for the Options Trading Engine.
    
    Provides command-line interface for all engine functionality including
    trade generation, market scanning, and system management.
    """
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.trade_formatter = TradeFormatter()
        self.console_formatter = ConsoleFormatter()
        self.table_formatter = TableFormatter()
        
        # Initialize API clients
        self._init_api_clients()
        
        # Initialize repositories
        self._init_repositories()
        
        # Initialize domain services
        self._init_domain_services()
        
        # Initialize use cases
        self._init_use_cases()
    
    def _init_api_clients(self):
        """Initialize API clients."""
        config = self.config_manager.get_config()
        
        self.tradier_client = TradierClient(
            api_key=config.api_keys.tradier_api_key,
            base_url=config.api_endpoints.tradier_base_url
        )
        
        self.yahoo_client = YahooClient(
            api_key=config.api_keys.yahoo_finance_api_key,
            base_url=config.api_endpoints.yahoo_base_url
        )
        
        self.fred_client = FredClient(
            api_key=config.api_keys.fred_api_key,
            base_url=config.api_endpoints.fred_base_url
        )
        
        self.quiver_client = QuiverClient(
            api_key=config.api_keys.quiver_quant_api_key,
            base_url=config.api_endpoints.quiver_base_url
        )
    
    def _init_repositories(self):
        """Initialize data repositories."""
        self.market_repo = MarketDataRepository(
            tradier_client=self.tradier_client,
            yahoo_client=self.yahoo_client,
            fred_client=self.fred_client,
            quiver_client=self.quiver_client,
            cache_manager=self.cache_manager
        )
        
        self.options_repo = OptionsRepository(
            tradier_client=self.tradier_client,
            yahoo_client=self.yahoo_client,
            cache_manager=self.cache_manager
        )
    
    def _init_domain_services(self):
        """Initialize domain services."""
        self.scoring_engine = ScoringEngine()
        self.risk_calculator = RiskCalculator()
        self.strategy_service = StrategyGenerationService()
        self.constraint_engine = ConstraintEngine()
        self.trade_selector = TradeSelector()
    
    def _init_use_cases(self):
        """Initialize use cases."""
        self.generate_trades_uc = GenerateTradesUseCase(
            market_repo=self.market_repo,
            options_repo=self.options_repo,
            strategy_service=self.strategy_service,
            scoring_engine=self.scoring_engine,
            constraint_engine=self.constraint_engine,
            trade_selector=self.trade_selector,
            config_manager=self.config_manager
        )
        
        self.scan_market_uc = ScanMarketUseCase(
            market_repo=self.market_repo,
            options_repo=self.options_repo,
            config_manager=self.config_manager
        )
    
    def run(self):
        """Main entry point for CLI."""
        parser = self._create_parser()
        args = parser.parse_args()
        
        # Configure logging
        self._configure_logging(args.log_level)
        
        try:
            # Execute command
            result = self._execute_command(args)
            
            # Output result
            if result:
                self._output_result(result, args.output_format)
                
            return 0
            
        except OptionsEngineError as e:
            self.console_formatter.print_error(f"Engine Error: {str(e)}")
            return 1
        except Exception as e:
            self.console_formatter.print_error(f"Unexpected Error: {str(e)}")
            logger.exception("Unexpected error in CLI")
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Options Trading Engine CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global options
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level'
        )
        
        parser.add_argument(
            '--output-format',
            choices=['json', 'table', 'csv'],
            default='table',
            help='Output format'
        )
        
        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Generate trades command
        self._add_generate_trades_parser(subparsers)
        
        # Scan market command
        self._add_scan_market_parser(subparsers)
        
        # Get quotes command
        self._add_get_quotes_parser(subparsers)
        
        # System commands
        self._add_system_commands(subparsers)
        
        return parser
    
    def _add_generate_trades_parser(self, subparsers):
        """Add generate trades command parser."""
        parser = subparsers.add_parser(
            'generate-trades',
            help='Generate trade recommendations'
        )
        
        parser.add_argument(
            'symbols',
            nargs='+',
            help='Stock symbols to analyze'
        )
        
        parser.add_argument(
            '--strategies',
            nargs='+',
            choices=['covered_call', 'cash_secured_put', 'iron_condor', 'butterfly', 'straddle', 'strangle'],
            default=['covered_call', 'cash_secured_put'],
            help='Strategies to generate'
        )
        
        parser.add_argument(
            '--max-trades',
            type=int,
            default=10,
            help='Maximum number of trades to generate'
        )
        
        parser.add_argument(
            '--min-score',
            type=float,
            default=0.7,
            help='Minimum trade score (0-1)'
        )
        
        parser.add_argument(
            '--max-risk',
            type=float,
            default=0.05,
            help='Maximum risk per trade (as percentage of portfolio)'
        )
        
        parser.add_argument(
            '--min-dte',
            type=int,
            default=7,
            help='Minimum days to expiration'
        )
        
        parser.add_argument(
            '--max-dte',
            type=int,
            default=60,
            help='Maximum days to expiration'
        )
    
    def _add_scan_market_parser(self, subparsers):
        """Add scan market command parser."""
        parser = subparsers.add_parser(
            'scan-market',
            help='Scan market for opportunities'
        )
        
        parser.add_argument(
            '--scan-type',
            choices=['momentum', 'mean_reversion', 'volatility', 'earnings', 'technical'],
            default='momentum',
            help='Type of scan to perform'
        )
        
        parser.add_argument(
            '--universe',
            choices=['sp500', 'nasdaq100', 'russell2000', 'custom'],
            default='sp500',
            help='Universe to scan'
        )
        
        parser.add_argument(
            '--symbols',
            nargs='*',
            help='Custom symbols to scan (if universe=custom)'
        )
        
        parser.add_argument(
            '--min-volume',
            type=int,
            default=1000000,
            help='Minimum average daily volume'
        )
        
        parser.add_argument(
            '--min-price',
            type=float,
            default=10.0,
            help='Minimum stock price'
        )
        
        parser.add_argument(
            '--max-price',
            type=float,
            default=1000.0,
            help='Maximum stock price'
        )
        
        parser.add_argument(
            '--limit',
            type=int,
            default=50,
            help='Maximum number of results'
        )
    
    def _add_get_quotes_parser(self, subparsers):
        """Add get quotes command parser."""
        parser = subparsers.add_parser(
            'get-quotes',
            help='Get quotes for symbols'
        )
        
        parser.add_argument(
            'symbols',
            nargs='+',
            help='Stock or option symbols'
        )
        
        parser.add_argument(
            '--type',
            choices=['stock', 'option'],
            default='stock',
            help='Type of quotes to get'
        )
        
        parser.add_argument(
            '--include-greeks',
            action='store_true',
            help='Include Greeks for options'
        )
    
    def _add_system_commands(self, subparsers):
        """Add system management commands."""
        # Status command
        status_parser = subparsers.add_parser(
            'status',
            help='Show system status'
        )
        
        # Cache management
        cache_parser = subparsers.add_parser(
            'cache',
            help='Cache management'
        )
        cache_subparsers = cache_parser.add_subparsers(dest='cache_action')
        
        cache_subparsers.add_parser('clear', help='Clear cache')
        cache_subparsers.add_parser('stats', help='Show cache statistics')
        
        # Config management
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management'
        )
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        
        config_subparsers.add_parser('show', help='Show configuration')
        config_subparsers.add_parser('validate', help='Validate configuration')
    
    def _configure_logging(self, log_level: str):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/options_engine.log')
            ]
        )
    
    def _execute_command(self, args) -> Optional[Dict[str, Any]]:
        """Execute CLI command."""
        if args.command == 'generate-trades':
            return self._execute_generate_trades(args)
        elif args.command == 'scan-market':
            return self._execute_scan_market(args)
        elif args.command == 'get-quotes':
            return self._execute_get_quotes(args)
        elif args.command == 'status':
            return self._execute_status(args)
        elif args.command == 'cache':
            return self._execute_cache_command(args)
        elif args.command == 'config':
            return self._execute_config_command(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    
    def _execute_generate_trades(self, args) -> Dict[str, Any]:
        """Execute generate trades command."""
        self.console_formatter.print_info(f"Generating trades for {len(args.symbols)} symbols...")
        
        # Create request
        request = {
            'symbols': args.symbols,
            'strategies': args.strategies,
            'max_trades': args.max_trades,
            'filters': {
                'min_score': args.min_score,
                'max_risk': args.max_risk,
                'min_dte': args.min_dte,
                'max_dte': args.max_dte
            }
        }
        
        # Execute use case
        result = self.generate_trades_uc.execute(request)
        
        # Format result
        return {
            'command': 'generate-trades',
            'request': request,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_scan_market(self, args) -> Dict[str, Any]:
        """Execute scan market command."""
        self.console_formatter.print_info(f"Scanning market with {args.scan_type} strategy...")
        
        # Determine symbols to scan
        symbols = args.symbols or []
        if args.universe != 'custom':
            symbols = self._get_universe_symbols(args.universe)
        
        # Create request
        request = {
            'scan_type': args.scan_type,
            'symbols': symbols,
            'filters': {
                'min_volume': args.min_volume,
                'min_price': args.min_price,
                'max_price': args.max_price
            },
            'limit': args.limit
        }
        
        # Execute use case
        result = self.scan_market_uc.execute(request)
        
        # Format result
        return {
            'command': 'scan-market',
            'request': request,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_get_quotes(self, args) -> Dict[str, Any]:
        """Execute get quotes command."""
        self.console_formatter.print_info(f"Getting quotes for {len(args.symbols)} symbols...")
        
        if args.type == 'stock':
            quotes = {}
            for symbol in args.symbols:
                market_data = self.market_repo.get_market_data(symbol)
                if market_data:
                    quotes[symbol] = market_data.to_dict()
        else:
            quotes = self.options_repo.get_option_quotes(args.symbols)
            quotes = {k: v.to_dict() for k, v in quotes.items()}
        
        return {
            'command': 'get-quotes',
            'type': args.type,
            'symbols': args.symbols,
            'quotes': quotes,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_status(self, args) -> Dict[str, Any]:
        """Execute status command."""
        # Get system status
        status = {
            'engine_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'api_clients': {},
            'repositories': {},
            'cache': self.cache_manager.get_stats(),
            'config': {
                'loaded': True,
                'valid': True
            }
        }
        
        # Check API clients
        for name, client in [
            ('tradier', self.tradier_client),
            ('yahoo', self.yahoo_client),
            ('fred', self.fred_client),
            ('quiver', self.quiver_client)
        ]:
            try:
                health = client.health_check()
                status['api_clients'][name] = health
            except Exception as e:
                status['api_clients'][name] = {'status': 'error', 'error': str(e)}
        
        # Check repositories
        for name, repo in [
            ('market', self.market_repo),
            ('options', self.options_repo)
        ]:
            try:
                health = repo.health_check()
                status['repositories'][name] = health
            except Exception as e:
                status['repositories'][name] = {'status': 'error', 'error': str(e)}
        
        return status
    
    def _execute_cache_command(self, args) -> Dict[str, Any]:
        """Execute cache management command."""
        if args.cache_action == 'clear':
            self.cache_manager.clear()
            return {'action': 'clear', 'status': 'success', 'message': 'Cache cleared'}
        
        elif args.cache_action == 'stats':
            stats = self.cache_manager.get_stats()
            return {'action': 'stats', 'stats': stats}
        
        else:
            raise ValueError(f"Unknown cache action: {args.cache_action}")
    
    def _execute_config_command(self, args) -> Dict[str, Any]:
        """Execute config management command."""
        if args.config_action == 'show':
            config = self.config_manager.get_config()
            return {'action': 'show', 'config': config.to_dict()}
        
        elif args.config_action == 'validate':
            try:
                self.config_manager.validate_config()
                return {'action': 'validate', 'status': 'valid'}
            except Exception as e:
                return {'action': 'validate', 'status': 'invalid', 'error': str(e)}
        
        else:
            raise ValueError(f"Unknown config action: {args.config_action}")
    
    def _get_universe_symbols(self, universe: str) -> List[str]:
        """Get symbols for a universe."""
        # This would typically load from a data source
        # For now, return sample symbols
        if universe == 'sp500':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        elif universe == 'nasdaq100':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CSCO']
        elif universe == 'russell2000':
            return ['AMC', 'GME', 'SNDL', 'CLOV', 'WISH', 'PLTR', 'BB', 'NOK', 'SPCE', 'TLRY']
        else:
            return []
    
    def _output_result(self, result: Dict[str, Any], format_type: str):
        """Output result in specified format."""
        if format_type == 'json':
            print(json.dumps(result, indent=2, default=str))
        
        elif format_type == 'table':
            if result['command'] == 'generate-trades':
                self._output_trades_table(result['result'])
            elif result['command'] == 'scan-market':
                self._output_scan_table(result['result'])
            elif result['command'] == 'get-quotes':
                self._output_quotes_table(result['quotes'])
            elif result['command'] == 'status':
                self._output_status_table(result)
            else:
                # Fallback to JSON for unknown commands
                print(json.dumps(result, indent=2, default=str))
        
        elif format_type == 'csv':
            # Convert to CSV format
            self._output_csv(result)
        
        else:
            raise ValueError(f"Unknown output format: {format_type}")
    
    def _output_trades_table(self, trades: List[Dict[str, Any]]):
        """Output trades in table format."""
        if not trades:
            self.console_formatter.print_warning("No trades found")
            return
        
        headers = ['Symbol', 'Strategy', 'Score', 'Max Profit', 'Max Loss', 'POP', 'DTE', 'Capital']
        rows = []
        
        for trade in trades:
            rows.append([
                trade.get('symbol', ''),
                trade.get('strategy', ''),
                f"{trade.get('score', 0):.2f}",
                f"${trade.get('max_profit', 0):.2f}",
                f"${trade.get('max_loss', 0):.2f}",
                f"{trade.get('probability_of_profit', 0):.1%}",
                trade.get('days_to_expiration', 0),
                f"${trade.get('capital_required', 0):.2f}"
            ])
        
        self.table_formatter.print_table(headers, rows, title="Trade Recommendations")
    
    def _output_scan_table(self, scan_results: List[Dict[str, Any]]):
        """Output scan results in table format."""
        if not scan_results:
            self.console_formatter.print_warning("No scan results found")
            return
        
        headers = ['Symbol', 'Price', 'Change', 'Volume', 'Score', 'Reason']
        rows = []
        
        for result in scan_results:
            rows.append([
                result.get('symbol', ''),
                f"${result.get('price', 0):.2f}",
                f"{result.get('change_percentage', 0):.2f}%",
                f"{result.get('volume', 0):,}",
                f"{result.get('score', 0):.2f}",
                result.get('reason', '')
            ])
        
        self.table_formatter.print_table(headers, rows, title="Market Scan Results")
    
    def _output_quotes_table(self, quotes: Dict[str, Any]):
        """Output quotes in table format."""
        if not quotes:
            self.console_formatter.print_warning("No quotes found")
            return
        
        headers = ['Symbol', 'Price', 'Change', 'Volume', 'Bid', 'Ask']
        rows = []
        
        for symbol, quote in quotes.items():
            rows.append([
                symbol,
                f"${quote.get('price', 0):.2f}",
                f"{quote.get('change_percentage', 0):.2f}%",
                f"{quote.get('volume', 0):,}",
                f"${quote.get('bid', 0):.2f}",
                f"${quote.get('ask', 0):.2f}"
            ])
        
        self.table_formatter.print_table(headers, rows, title="Quotes")
    
    def _output_status_table(self, status: Dict[str, Any]):
        """Output status in table format."""
        self.console_formatter.print_success(f"Engine Status: {status['engine_status']}")
        self.console_formatter.print_info(f"Timestamp: {status['timestamp']}")
        
        # API Clients status
        print("\nAPI Clients:")
        for name, client_status in status['api_clients'].items():
            status_str = client_status.get('status', 'unknown')
            if status_str == 'healthy':
                self.console_formatter.print_success(f"  {name}: {status_str}")
            else:
                self.console_formatter.print_error(f"  {name}: {status_str}")
        
        # Repositories status
        print("\nRepositories:")
        for name, repo_status in status['repositories'].items():
            status_str = repo_status.get('status', 'unknown')
            if status_str == 'healthy':
                self.console_formatter.print_success(f"  {name}: {status_str}")
            else:
                self.console_formatter.print_error(f"  {name}: {status_str}")
        
        # Cache stats
        cache_stats = status.get('cache', {})
        print(f"\nCache Statistics:")
        print(f"  Hits: {cache_stats.get('hits', 0)}")
        print(f"  Misses: {cache_stats.get('misses', 0)}")
        print(f"  Size: {cache_stats.get('size', 0)}")
    
    def _output_csv(self, result: Dict[str, Any]):
        """Output result in CSV format."""
        # Simple CSV implementation
        import csv
        import io
        
        output = io.StringIO()
        
        if result['command'] == 'generate-trades':
            writer = csv.DictWriter(output, fieldnames=['symbol', 'strategy', 'score', 'max_profit', 'max_loss'])
            writer.writeheader()
            for trade in result['result']:
                writer.writerow(trade)
        
        elif result['command'] == 'get-quotes':
            if result['quotes']:
                first_quote = next(iter(result['quotes'].values()))
                writer = csv.DictWriter(output, fieldnames=first_quote.keys())
                writer.writeheader()
                for symbol, quote in result['quotes'].items():
                    quote['symbol'] = symbol
                    writer.writerow(quote)
        
        print(output.getvalue())


def main():
    """Main entry point."""
    cli = OptionsEngineCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())