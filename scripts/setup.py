#!/usr/bin/env python3
"""
Project setup and initialization script for the options trading engine.

This script helps with initial project setup, environment configuration,
and dependency installation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import shutil
import urllib.request
import json


class SetupError(Exception):
    """Exception raised for setup errors."""
    pass


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            raise SetupError(f"Command failed: {' '.join(cmd)}")
        return e


def check_python_version() -> None:
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 9):
        raise SetupError(
            f"Python 3.9 or higher is required. Current version: {sys.version}"
        )
    
    print(f"✅ Python version {sys.version.split()[0]} is compatible")


def check_git_available() -> bool:
    """Check if git is available."""
    try:
        run_command(["git", "--version"])
        return True
    except SetupError:
        return False


def create_virtual_environment(venv_path: Path) -> None:
    """Create virtual environment."""
    print(f"Creating virtual environment at {venv_path}...")
    
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return
    
    run_command([sys.executable, "-m", "venv", str(venv_path)])
    print(f"✅ Virtual environment created at {venv_path}")


def get_venv_python(venv_path: Path) -> Path:
    """Get path to Python executable in virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies(venv_path: Path, dev: bool = False) -> None:
    """Install project dependencies."""
    print("Installing dependencies...")
    
    python_exe = get_venv_python(venv_path)
    
    # Upgrade pip
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    requirements_file = "requirements.txt"
    if requirements_file and Path(requirements_file).exists():
        run_command([str(python_exe), "-m", "pip", "install", "-r", requirements_file])
    
    # Install development dependencies
    if dev:
        dev_requirements = [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pylint>=2.15.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
            "pre-commit>=2.20.0"
        ]
        run_command([str(python_exe), "-m", "pip", "install"] + dev_requirements)
    
    # Install project in development mode
    run_command([str(python_exe), "-m", "pip", "install", "-e", "."])
    
    print("✅ Dependencies installed successfully")


def create_directories() -> None:
    """Create necessary directories."""
    print("Creating project directories...")
    
    directories = [
        "logs",
        "cache",
        "temp",
        "data",
        "reports",
        "backups"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    print("✅ Project directories created")


def setup_environment_file() -> None:
    """Setup environment configuration file."""
    print("Setting up environment configuration...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print(f"✅ Created .env file from .env.example")
        print("⚠️  Please edit .env file with your API keys and configuration")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  .env.example not found, creating basic .env file")
        basic_env_content = """# Basic environment configuration
APP_ENVIRONMENT=development
LOG_LEVEL=INFO

# Add your API keys here
TRADIER_API_KEY=your_tradier_api_key_here
YAHOO_RAPIDAPI_KEY=your_rapidapi_key_here
FRED_API_KEY=your_fred_api_key_here
QUIVER_API_KEY=your_quiver_api_key_here
"""
        env_file.write_text(basic_env_content)
        print("✅ Created basic .env file")


def setup_git_hooks(venv_path: Path) -> None:
    """Setup git pre-commit hooks."""
    print("Setting up git hooks...")
    
    if not check_git_available():
        print("⚠️  Git not available, skipping git hooks setup")
        return
    
    python_exe = get_venv_python(venv_path)
    
    try:
        # Install pre-commit hooks
        run_command([str(python_exe), "-m", "pre_commit", "install"])
        print("✅ Git pre-commit hooks installed")
    except SetupError:
        print("⚠️  Failed to setup git hooks (pre-commit may not be installed)")


def validate_setup(venv_path: Path) -> None:
    """Validate the setup by running basic tests."""
    print("Validating setup...")
    
    python_exe = get_venv_python(venv_path)
    
    # Test import
    try:
        run_command([str(python_exe), "-c", "import src; print('Import successful')"])
        print("✅ Package import test passed")
    except SetupError:
        print("❌ Package import test failed")
        raise
    
    # Test configuration loading
    try:
        run_command([
            str(python_exe), "-c", 
            "from src.application.config.settings import Settings; print('Config loading successful')"
        ])
        print("✅ Configuration loading test passed")
    except SetupError:
        print("⚠️  Configuration loading test failed (may need API keys)")
    
    # Run basic tests if pytest is available
    try:
        run_command([str(python_exe), "-m", "pytest", "tests/", "-v", "--tb=short"], check=False)
        print("✅ Basic tests completed")
    except SetupError:
        print("⚠️  Some tests may have failed (check test output)")


def download_sample_data() -> None:
    """Download sample data for development/testing."""
    print("Setting up sample data...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample configuration files
    sample_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    symbols_file = data_dir / "sample_symbols.json"
    
    if not symbols_file.exists():
        with open(symbols_file, 'w') as f:
            json.dump({
                "broad_market": sample_symbols,
                "tech_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "meme_stocks": ["GME", "AMC", "TSLA"]
            }, f, indent=2)
        print(f"✅ Created sample symbols file: {symbols_file}")


def print_next_steps() -> None:
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("\n2. Edit the .env file with your API keys:")
    print("   nano .env  # or your preferred editor")
    print("\n3. Verify the installation:")
    print("   python -c 'from src import __version__; print(__version__)'")
    print("\n4. Run the tests:")
    print("   python scripts/run_tests.py all")
    print("\n5. Generate your first trade recommendations:")
    print("   python -m src.presentation.cli.main")
    print("\n6. Read the documentation:")
    print("   - API documentation: docs/API.md")
    print("   - Configuration guide: docs/CONFIGURATION.md")
    print("   - Development guide: docs/DEVELOPMENT.md")
    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup script for options trading engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Basic setup
  %(prog)s --dev              # Setup with development tools
  %(prog)s --venv myenv       # Use custom virtual environment name
  %(prog)s --no-venv          # Skip virtual environment creation
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies"
    )
    
    parser.add_argument(
        "--venv",
        default="venv",
        help="Virtual environment directory name (default: venv)"
    )
    
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Skip virtual environment creation"
    )
    
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip git hooks setup"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip setup validation"
    )
    
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download sample data"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        print("🚀 Starting Options Trading Engine setup...")
        print(f"Project root: {project_root.absolute()}")
        
        # Check prerequisites
        check_python_version()
        
        # Setup virtual environment
        venv_path = None
        if not args.no_venv:
            venv_path = Path(args.venv)
            create_virtual_environment(venv_path)
        else:
            print("⚠️  Skipping virtual environment creation")
            venv_path = Path(sys.executable).parent.parent  # Fallback
        
        # Create directories
        create_directories()
        
        # Setup environment file
        setup_environment_file()
        
        # Install dependencies
        if venv_path:
            install_dependencies(venv_path, dev=args.dev)
        
        # Setup git hooks
        if not args.skip_git and venv_path:
            setup_git_hooks(venv_path)
        
        # Download sample data
        if args.download_data:
            download_sample_data()
        
        # Validate setup
        if not args.skip_validation and venv_path:
            validate_setup(venv_path)
        
        # Print next steps
        print_next_steps()
        
    except SetupError as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()