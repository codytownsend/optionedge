#!/usr/bin/env python3
"""
Production readiness validation script for the Options Trading Engine.
Performs comprehensive validation of all production readiness requirements.
"""

import os
import sys
import logging
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import yaml
import psutil
import requests
from dataclasses import dataclass
import sqlite3
import redis
import psycopg2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.infrastructure.monitoring.monitoring_system import get_monitoring_system
from src.infrastructure.performance.performance_optimizer import get_performance_optimizer
from src.infrastructure.error_handling.recovery_system import get_recovery_system


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'message': self.message,
            'severity': self.severity,
            'details': self.details or {}
        }


class ProductionReadinessValidator:
    """
    Comprehensive production readiness validator.
    
    Validates all aspects of the system for production deployment including:
    - Environment setup and dependencies
    - Configuration management
    - Database setup and connectivity
    - Security measures
    - Performance requirements
    - Monitoring and alerting
    - Error handling and recovery
    - Documentation completeness
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        self.validation_results = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _add_result(self, name: str, passed: bool, message: str, severity: str = "error", details: Dict[str, Any] = None):
        """Add validation result."""
        result = ValidationResult(name, passed, message, severity, details)
        self.validation_results.append(result)
        
        # Log result
        log_level = logging.INFO if passed else logging.ERROR
        if severity == "warning":
            log_level = logging.WARNING
        
        self.logger.log(log_level, f"{name}: {message}")
    
    def validate_environment_setup(self) -> bool:
        """Validate environment setup and dependencies."""
        self.logger.info("Validating environment setup...")
        
        all_passed = True
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 9):
            self._add_result("Python Version", True, f"Python {python_version.major}.{python_version.minor} is supported", "info")
        else:
            self._add_result("Python Version", False, f"Python {python_version.major}.{python_version.minor} is too old. Requires Python 3.9+")
            all_passed = False
        
        # Check system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 8:
            self._add_result("Memory", True, f"System has {memory_gb:.1f}GB RAM", "info")
        else:
            self._add_result("Memory", False, f"Insufficient memory: {memory_gb:.1f}GB (requires 8GB+)", "warning")
            all_passed = False
        
        cpu_count = psutil.cpu_count()
        if cpu_count >= 4:
            self._add_result("CPU Cores", True, f"System has {cpu_count} CPU cores", "info")
        else:
            self._add_result("CPU Cores", False, f"Insufficient CPU cores: {cpu_count} (requires 4+)", "warning")
            all_passed = False
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        if free_gb >= 100:
            self._add_result("Disk Space", True, f"System has {free_gb:.1f}GB free space", "info")
        else:
            self._add_result("Disk Space", False, f"Insufficient disk space: {free_gb:.1f}GB (requires 100GB+)", "warning")
            all_passed = False
        
        # Check required dependencies
        required_packages = [
            'requests', 'pandas', 'numpy', 'scipy', 'pyyaml', 
            'pytest', 'psutil', 'psycopg2', 'redis'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self._add_result(f"Package {package}", True, f"Package {package} is installed", "info")
            except ImportError:
                self._add_result(f"Package {package}", False, f"Package {package} is not installed")
                all_passed = False
        
        # Check virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self._add_result("Virtual Environment", True, "Running in virtual environment", "info")
        else:
            self._add_result("Virtual Environment", False, "Not running in virtual environment", "warning")
        
        return all_passed
    
    def validate_configuration(self) -> bool:
        """Validate configuration management."""
        self.logger.info("Validating configuration...")
        
        all_passed = True
        
        # Check configuration file
        if os.path.exists(self.config_path):
            self._add_result("Configuration File", True, f"Configuration file exists: {self.config_path}", "info")
        else:
            self._add_result("Configuration File", False, f"Configuration file missing: {self.config_path}")
            all_passed = False
        
        # Check environment variables
        required_env_vars = [
            'TRADIER_API_KEY', 'YAHOO_FINANCE_API_KEY', 'FRED_API_KEY', 
            'QUIVER_QUANT_API_KEY', 'DATABASE_URL', 'REDIS_URL'
        ]
        
        for env_var in required_env_vars:
            if os.getenv(env_var):
                self._add_result(f"Environment Variable {env_var}", True, f"Environment variable {env_var} is set", "info")
            else:
                self._add_result(f"Environment Variable {env_var}", False, f"Environment variable {env_var} is not set")
                all_passed = False
        
        # Check .env file
        env_file = Path('.env')
        if env_file.exists():
            self._add_result("Environment File", True, ".env file exists", "info")
        else:
            self._add_result("Environment File", False, ".env file missing", "warning")
        
        # Check logging configuration
        logging_config = Path('config/logging.yaml')
        if logging_config.exists():
            self._add_result("Logging Configuration", True, "Logging configuration exists", "info")
        else:
            self._add_result("Logging Configuration", False, "Logging configuration missing", "warning")
        
        return all_passed
    
    def validate_database_setup(self) -> bool:
        """Validate database setup and connectivity."""
        self.logger.info("Validating database setup...")
        
        all_passed = True
        
        # Check database connectivity
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            try:
                conn = psycopg2.connect(database_url)
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                self._add_result("Database Connectivity", True, f"Database connected: {version}", "info")
            except Exception as e:
                self._add_result("Database Connectivity", False, f"Database connection failed: {e}")
                all_passed = False
        else:
            self._add_result("Database Connectivity", False, "DATABASE_URL not configured")
            all_passed = False
        
        # Check Redis connectivity
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            try:
                r = redis.from_url(redis_url)
                r.ping()
                info = r.info()
                r.close()
                
                self._add_result("Redis Connectivity", True, f"Redis connected: {info['redis_version']}", "info")
            except Exception as e:
                self._add_result("Redis Connectivity", False, f"Redis connection failed: {e}")
                all_passed = False
        else:
            self._add_result("Redis Connectivity", False, "REDIS_URL not configured")
            all_passed = False
        
        return all_passed
    
    def validate_security_measures(self) -> bool:
        """Validate security measures."""
        self.logger.info("Validating security measures...")
        
        all_passed = True
        
        # Check file permissions
        sensitive_files = ['.env', 'config/settings.yaml']
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                file_stat = os.stat(file_path)
                file_mode = oct(file_stat.st_mode)[-3:]
                if file_mode in ['600', '400']:
                    self._add_result(f"File Permissions {file_path}", True, f"File permissions secure: {file_mode}", "info")
                else:
                    self._add_result(f"File Permissions {file_path}", False, f"File permissions insecure: {file_mode} (should be 600 or 400)", "warning")
                    all_passed = False
        
        # Check for secrets in code
        secret_patterns = ['password', 'secret', 'key', 'token']
        python_files = list(Path('src').glob('**/*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    for pattern in secret_patterns:
                        if f'{pattern}=' in content or f'{pattern}:' in content:
                            self._add_result(f"Secret Exposure {file_path}", False, f"Potential secret exposure in {file_path}", "warning")
                            all_passed = False
                            break
            except Exception:
                continue
        
        # Check SSL/TLS configuration
        ssl_enabled = self.config.get('ssl', {}).get('enabled', False)
        if ssl_enabled:
            self._add_result("SSL/TLS", True, "SSL/TLS is enabled", "info")
        else:
            self._add_result("SSL/TLS", False, "SSL/TLS is not enabled", "warning")
        
        return all_passed
    
    def validate_performance_requirements(self) -> bool:
        """Validate performance requirements."""
        self.logger.info("Validating performance requirements...")
        
        all_passed = True
        
        # Test import performance
        start_time = time.time()
        try:
            import src.main
            import_time = time.time() - start_time
            
            if import_time < 5.0:
                self._add_result("Import Performance", True, f"Module import time: {import_time:.2f}s", "info")
            else:
                self._add_result("Import Performance", False, f"Slow module import: {import_time:.2f}s (should be <5s)", "warning")
                all_passed = False
        except Exception as e:
            self._add_result("Import Performance", False, f"Failed to import main module: {e}")
            all_passed = False
        
        # Check performance optimizer
        try:
            perf_optimizer = get_performance_optimizer()
            self._add_result("Performance Optimizer", True, "Performance optimizer initialized", "info")
        except Exception as e:
            self._add_result("Performance Optimizer", False, f"Performance optimizer failed: {e}")
            all_passed = False
        
        # Check thread pool configuration
        thread_pool_size = self.config.get('performance', {}).get('thread_pool_size', 8)
        if thread_pool_size >= 4:
            self._add_result("Thread Pool Size", True, f"Thread pool size: {thread_pool_size}", "info")
        else:
            self._add_result("Thread Pool Size", False, f"Thread pool size too small: {thread_pool_size}", "warning")
            all_passed = False
        
        return all_passed
    
    def validate_monitoring_alerting(self) -> bool:
        """Validate monitoring and alerting systems."""
        self.logger.info("Validating monitoring and alerting...")
        
        all_passed = True
        
        # Check monitoring system
        try:
            monitoring_system = get_monitoring_system()
            self._add_result("Monitoring System", True, "Monitoring system initialized", "info")
        except Exception as e:
            self._add_result("Monitoring System", False, f"Monitoring system failed: {e}")
            all_passed = False
        
        # Check alert configuration
        alert_config = self.config.get('alerts', {})
        if alert_config:
            self._add_result("Alert Configuration", True, "Alert configuration present", "info")
        else:
            self._add_result("Alert Configuration", False, "Alert configuration missing", "warning")
        
        # Check logging directory
        log_dir = Path('logs')
        if log_dir.exists() and log_dir.is_dir():
            self._add_result("Log Directory", True, "Log directory exists", "info")
        else:
            self._add_result("Log Directory", False, "Log directory missing", "warning")
            all_passed = False
        
        return all_passed
    
    def validate_error_handling(self) -> bool:
        """Validate error handling and recovery."""
        self.logger.info("Validating error handling...")
        
        all_passed = True
        
        # Check error recovery system
        try:
            recovery_system = get_recovery_system()
            self._add_result("Error Recovery System", True, "Error recovery system initialized", "info")
        except Exception as e:
            self._add_result("Error Recovery System", False, f"Error recovery system failed: {e}")
            all_passed = False
        
        # Check error handling configuration
        error_config = self.config.get('error_handling', {})
        if error_config:
            self._add_result("Error Handling Configuration", True, "Error handling configuration present", "info")
        else:
            self._add_result("Error Handling Configuration", False, "Error handling configuration missing", "warning")
        
        return all_passed
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        self.logger.info("Validating documentation...")
        
        all_passed = True
        
        # Check required documentation files
        required_docs = [
            'README.md',
            'docs/DEPLOYMENT.md',
            'docs/OPERATIONS.md',
            'docs/PRODUCTION_READINESS_CHECKLIST.md',
            'docs/API.md',
            'docs/CONFIGURATION.md'
        ]
        
        for doc_file in required_docs:
            if os.path.exists(doc_file):
                self._add_result(f"Documentation {doc_file}", True, f"Documentation exists: {doc_file}", "info")
            else:
                self._add_result(f"Documentation {doc_file}", False, f"Documentation missing: {doc_file}", "warning")
                all_passed = False
        
        return all_passed
    
    def validate_tests(self) -> bool:
        """Validate test suite."""
        self.logger.info("Validating tests...")
        
        all_passed = True
        
        # Check test directories
        test_dirs = ['tests/unit', 'tests/integration', 'tests/end_to_end']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                self._add_result(f"Test Directory {test_dir}", True, f"Test directory exists: {test_dir}", "info")
            else:
                self._add_result(f"Test Directory {test_dir}", False, f"Test directory missing: {test_dir}", "warning")
                all_passed = False
        
        # Run tests if pytest is available
        try:
            result = subprocess.run(['python', '-m', 'pytest', 'tests/', '--tb=short'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self._add_result("Test Suite", True, "All tests passed", "info")
            else:
                self._add_result("Test Suite", False, f"Tests failed: {result.stdout}", "warning")
                all_passed = False
        except subprocess.TimeoutExpired:
            self._add_result("Test Suite", False, "Test suite timed out", "warning")
            all_passed = False
        except Exception as e:
            self._add_result("Test Suite", False, f"Failed to run tests: {e}", "warning")
            all_passed = False
        
        return all_passed
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run full production readiness validation."""
        self.logger.info("Starting full production readiness validation...")
        
        validation_start = datetime.now()
        
        # Run all validations
        validations = [
            ("Environment Setup", self.validate_environment_setup),
            ("Configuration", self.validate_configuration),
            ("Database Setup", self.validate_database_setup),
            ("Security Measures", self.validate_security_measures),
            ("Performance Requirements", self.validate_performance_requirements),
            ("Monitoring & Alerting", self.validate_monitoring_alerting),
            ("Error Handling", self.validate_error_handling),
            ("Documentation", self.validate_documentation),
            ("Tests", self.validate_tests)
        ]
        
        validation_summary = {}
        overall_passed = True
        
        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                validation_summary[validation_name] = result
                if not result:
                    overall_passed = False
            except Exception as e:
                self.logger.error(f"Validation {validation_name} failed with exception: {e}")
                validation_summary[validation_name] = False
                overall_passed = False
        
        validation_end = datetime.now()
        validation_duration = (validation_end - validation_start).total_seconds()
        
        # Generate summary report
        summary = {
            'overall_passed': overall_passed,
            'validation_start': validation_start.isoformat(),
            'validation_end': validation_end.isoformat(),
            'validation_duration': validation_duration,
            'validation_summary': validation_summary,
            'total_checks': len(self.validation_results),
            'passed_checks': len([r for r in self.validation_results if r.passed]),
            'failed_checks': len([r for r in self.validation_results if not r.passed]),
            'results': [r.to_dict() for r in self.validation_results]
        }
        
        # Save results to file
        results_file = f'validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Validation completed in {validation_duration:.2f} seconds")
        self.logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        self._print_validation_summary(summary)
        
        return summary
    
    def _print_validation_summary(self, summary: Dict[str, Any]):
        """Print validation summary to console."""
        print("\n" + "="*80)
        print("PRODUCTION READINESS VALIDATION SUMMARY")
        print("="*80)
        
        print(f"Overall Status: {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed_checks']}")
        print(f"Failed: {summary['failed_checks']}")
        print(f"Duration: {summary['validation_duration']:.2f} seconds")
        
        print("\nValidation Categories:")
        for category, result in summary['validation_summary'].items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {category}: {status}")
        
        # Print failed checks
        failed_results = [r for r in self.validation_results if not r.passed]
        if failed_results:
            print(f"\nFailed Checks ({len(failed_results)}):")
            for result in failed_results:
                print(f"  ❌ {result.name}: {result.message}")
        
        # Print warnings
        warning_results = [r for r in self.validation_results if r.severity == "warning"]
        if warning_results:
            print(f"\nWarnings ({len(warning_results)}):")
            for result in warning_results:
                print(f"  ⚠️  {result.name}: {result.message}")
        
        print("\n" + "="*80)
        
        if summary['overall_passed']:
            print("🎉 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
        else:
            print("❌ SYSTEM IS NOT READY FOR PRODUCTION DEPLOYMENT")
            print("   Please address the failed checks before deploying to production.")
        
        print("="*80)


def main():
    """Main function to run production readiness validation."""
    parser = argparse.ArgumentParser(description='Validate production readiness of Options Trading Engine')
    parser.add_argument('--config', default='config/settings.yaml', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = ProductionReadinessValidator(args.config)
    summary = validator.run_full_validation()
    
    # Exit with appropriate code
    sys.exit(0 if summary['overall_passed'] else 1)


if __name__ == "__main__":
    import argparse
    main()