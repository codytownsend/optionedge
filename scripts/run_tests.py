#!/usr/bin/env python3
"""
Test runner script for the options trading engine.

This script provides a convenient way to run different types of tests
with various configurations and reporting options.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_unit_tests(
    verbose: bool = False,
    coverage: bool = False,
    fail_under: Optional[int] = None,
    parallel: bool = False
) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if fail_under:
            cmd.append(f"--cov-fail-under={fail_under}")
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    return run_command(cmd)


def run_integration_tests(verbose: bool = False, slow: bool = False) -> int:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd)


def run_end_to_end_tests(verbose: bool = False) -> int:
    """Run end-to-end tests."""
    cmd = ["python", "-m", "pytest", "tests/end_to_end/"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def run_all_tests(
    verbose: bool = False,
    coverage: bool = False,
    fail_under: Optional[int] = None,
    parallel: bool = False,
    slow: bool = False
) -> int:
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
        if fail_under:
            cmd.append(f"--cov-fail-under={fail_under}")
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd)


def run_linting() -> int:
    """Run code linting checks."""
    print("Running code quality checks...")
    
    # Black formatting check
    result = run_command(["black", "--check", "src/", "tests/"])
    if result != 0:
        print("❌ Black formatting check failed")
        return result
    
    # isort import sorting check
    result = run_command(["isort", "--check-only", "src/", "tests/"])
    if result != 0:
        print("❌ isort import sorting check failed")
        return result
    
    # flake8 linting
    result = run_command(["flake8", "src/", "tests/"])
    if result != 0:
        print("❌ flake8 linting failed")
        return result
    
    # pylint
    result = run_command(["pylint", "src/"])
    if result != 0:
        print("❌ pylint check failed")
        return result
    
    # mypy type checking
    result = run_command(["mypy", "src/"])
    if result != 0:
        print("❌ mypy type checking failed")
        return result
    
    print("✅ All code quality checks passed")
    return 0


def run_security_checks() -> int:
    """Run security checks."""
    print("Running security checks...")
    
    # bandit security linting
    result = run_command(["bandit", "-r", "src/"])
    if result != 0:
        print("❌ bandit security check failed")
        return result
    
    # safety dependency check
    result = run_command(["safety", "check"])
    if result != 0:
        print("❌ safety dependency check failed")
        return result
    
    print("✅ All security checks passed")
    return 0


def generate_coverage_report(format_type: str = "html") -> int:
    """Generate coverage report."""
    cmd = ["python", "-m", "pytest", "tests/", "--cov=src"]
    
    if format_type == "html":
        cmd.append("--cov-report=html")
    elif format_type == "xml":
        cmd.append("--cov-report=xml")
    elif format_type == "json":
        cmd.append("--cov-report=json")
    else:
        cmd.append("--cov-report=term-missing")
    
    return run_command(cmd)


def clean_cache() -> None:
    """Clean test and build cache."""
    print("Cleaning cache directories...")
    
    cache_dirs = [
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        ".coverage",
        "htmlcov",
        "coverage.xml",
        "coverage.json",
        ".tox",
        "build",
        "dist",
        "*.egg-info"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            if os.path.isdir(cache_dir):
                subprocess.run(["rm", "-rf", cache_dir])
            else:
                os.remove(cache_dir)
    
    # Remove __pycache__ recursively
    subprocess.run(["find", ".", "-type", "d", "-name", "__pycache__", "-delete"])
    subprocess.run(["find", ".", "-name", "*.pyc", "-delete"])
    subprocess.run(["find", ".", "-name", "*.pyo", "-delete"])
    
    print("Cache cleaned")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for options trading engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s unit                    # Run unit tests
  %(prog)s integration --slow      # Run integration tests including slow ones
  %(prog)s all --coverage --fail-under 80  # Run all tests with coverage
  %(prog)s lint                    # Run linting checks
  %(prog)s security                # Run security checks
  %(prog)s coverage --format html  # Generate HTML coverage report
  %(prog)s clean                   # Clean cache directories
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "e2e", "all", "lint", "security", "coverage", "clean"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--fail-under",
        type=int,
        help="Fail if coverage is under this percentage"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    
    parser.add_argument(
        "--format",
        choices=["html", "xml", "json", "term"],
        default="html",
        help="Coverage report format"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    exit_code = 0
    
    if args.test_type == "unit":
        exit_code = run_unit_tests(
            verbose=args.verbose,
            coverage=args.coverage,
            fail_under=args.fail_under,
            parallel=args.parallel
        )
    elif args.test_type == "integration":
        exit_code = run_integration_tests(
            verbose=args.verbose,
            slow=args.slow
        )
    elif args.test_type == "e2e":
        exit_code = run_end_to_end_tests(verbose=args.verbose)
    elif args.test_type == "all":
        exit_code = run_all_tests(
            verbose=args.verbose,
            coverage=args.coverage,
            fail_under=args.fail_under,
            parallel=args.parallel,
            slow=args.slow
        )
    elif args.test_type == "lint":
        exit_code = run_linting()
    elif args.test_type == "security":
        exit_code = run_security_checks()
    elif args.test_type == "coverage":
        exit_code = generate_coverage_report(format_type=args.format)
    elif args.test_type == "clean":
        clean_cache()
    
    if exit_code == 0:
        print("\n✅ All checks passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()