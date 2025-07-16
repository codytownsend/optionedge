#!/usr/bin/env python3
"""
Test runner script for the options trading engine.
Provides convenient test execution with different configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ Command completed successfully")
        return True


def run_unit_tests():
    """Run unit tests only."""
    cmd = "python -m pytest tests/unit/ -v -m 'not slow' --tb=short"
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests only."""
    cmd = "python -m pytest tests/integration/ -v --tb=short"
    return run_command(cmd, "Integration Tests")


def run_stress_tests():
    """Run stress tests only."""
    cmd = "python -m pytest tests/stress_testing/ -v --tb=short -x"
    return run_command(cmd, "Stress Tests")


def run_backtest_tests():
    """Run backtesting tests only."""
    cmd = "python -m pytest tests/backtesting/ -v --tb=short"
    return run_command(cmd, "Backtesting Tests")


def run_all_tests():
    """Run all tests."""
    cmd = "python -m pytest tests/ -v --tb=short"
    return run_command(cmd, "All Tests")


def run_fast_tests():
    """Run fast tests only (excluding slow tests)."""
    cmd = "python -m pytest tests/ -v -m 'not slow' --tb=short"
    return run_command(cmd, "Fast Tests")


def run_coverage_report():
    """Generate coverage report."""
    cmd = "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing"
    return run_command(cmd, "Coverage Report")


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    cmd = f"python -m pytest {test_path} -v --tb=short"
    return run_command(cmd, f"Specific Test: {test_path}")


def run_mathematical_tests():
    """Run mathematical validation tests."""
    cmd = "python -m pytest tests/unit/test_mathematical_validation.py -v --tb=short"
    return run_command(cmd, "Mathematical Validation Tests")


def run_constraint_tests():
    """Run constraint validation tests."""
    cmd = "python -m pytest tests/unit/test_constraint_validation.py -v --tb=short"
    return run_command(cmd, "Constraint Validation Tests")


def run_api_tests():
    """Run API integration tests."""
    cmd = "python -m pytest tests/unit/test_api_integration.py -v --tb=short"
    return run_command(cmd, "API Integration Tests")


def run_performance_tests():
    """Run performance validation tests."""
    cmd = "python -m pytest tests/ -v -m 'performance' --tb=short"
    return run_command(cmd, "Performance Tests")


def run_with_profiling(test_path="tests/"):
    """Run tests with profiling."""
    cmd = f"python -m pytest {test_path} -v --tb=short --profile"
    return run_command(cmd, f"Profiling Tests: {test_path}")


def run_parallel_tests():
    """Run tests in parallel."""
    cmd = "python -m pytest tests/ -v --tb=short -n auto"
    return run_command(cmd, "Parallel Tests")


def check_test_environment():
    """Check if test environment is properly set up."""
    print("\n" + "="*60)
    print("Checking Test Environment")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    # Check required packages
    required_packages = [
        'pytest',
        'pytest-cov',
        'pytest-mock',
        'pandas',
        'numpy',
        'requests',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check project structure
    required_dirs = [
        'src',
        'tests',
        'tests/unit',
        'tests/integration',
        'tests/stress_testing',
        'tests/backtesting'
    ]
    
    project_root = Path(__file__).parent.parent
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            return False
    
    print("\n✅ Test environment is properly configured")
    return True


def clean_test_artifacts():
    """Clean up test artifacts."""
    print("\n" + "="*60)
    print("Cleaning Test Artifacts")
    print("="*60)
    
    artifacts_to_clean = [
        '.pytest_cache',
        '__pycache__',
        '*.pyc',
        '.coverage',
        'htmlcov',
        'test-results.xml',
        'coverage.xml'
    ]
    
    for artifact in artifacts_to_clean:
        if '*' in artifact:
            cmd = f"find . -name '{artifact}' -delete"
        else:
            cmd = f"rm -rf {artifact}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            print(f"✅ Cleaned {artifact}")
        else:
            print(f"ℹ️  {artifact} not found or already clean")
    
    print("\n✅ Test artifacts cleaned")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for the options trading engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py --unit                # Run unit tests only
    python run_tests.py --integration         # Run integration tests only
    python run_tests.py --stress              # Run stress tests only
    python run_tests.py --fast                # Run fast tests only
    python run_tests.py --coverage            # Generate coverage report
    python run_tests.py --check-env           # Check test environment
    python run_tests.py --clean               # Clean test artifacts
    python run_tests.py --specific tests/unit/test_mathematical_validation.py
        """
    )
    
    # Test type options
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--stress', action='store_true', help='Run stress tests only')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting tests only')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only (exclude slow tests)')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--mathematical', action='store_true', help='Run mathematical validation tests')
    parser.add_argument('--constraint', action='store_true', help='Run constraint validation tests')
    parser.add_argument('--api', action='store_true', help='Run API integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    
    # Test execution options
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--profile', action='store_true', help='Run with profiling')
    parser.add_argument('--specific', type=str, help='Run specific test file or function')
    
    # Utility options
    parser.add_argument('--check-env', action='store_true', help='Check test environment')
    parser.add_argument('--clean', action='store_true', help='Clean test artifacts')
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Add src to Python path
    sys.path.insert(0, str(project_root / 'src'))
    
    success = True
    
    # Handle utility options first
    if args.check_env:
        success = check_test_environment()
        if not success:
            sys.exit(1)
    
    if args.clean:
        clean_test_artifacts()
        return
    
    # Run specific test type
    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.stress:
        success = run_stress_tests()
    elif args.backtest:
        success = run_backtest_tests()
    elif args.fast:
        success = run_fast_tests()
    elif args.mathematical:
        success = run_mathematical_tests()
    elif args.constraint:
        success = run_constraint_tests()
    elif args.api:
        success = run_api_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.coverage:
        success = run_coverage_report()
    elif args.parallel:
        success = run_parallel_tests()
    elif args.profile:
        success = run_with_profiling()
    elif args.specific:
        success = run_specific_test(args.specific)
    elif args.all:
        success = run_all_tests()
    else:
        # Default: run fast tests
        success = run_fast_tests()
    
    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()