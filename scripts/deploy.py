#!/usr/bin/env python3
"""
Deployment script for the options trading engine.

This script handles deployment tasks such as building, testing,
and deploying the application to different environments.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import yaml
import json
from datetime import datetime


class DeploymentError(Exception):
    """Exception raised for deployment errors."""
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
        raise DeploymentError(f"Command failed: {' '.join(cmd)}")


def check_prerequisites() -> None:
    """Check deployment prerequisites."""
    print("Checking prerequisites...")
    
    # Check if git is clean
    try:
        result = run_command(["git", "status", "--porcelain"])
        if result.stdout.strip():
            raise DeploymentError("Git repository is not clean. Please commit or stash changes.")
    except subprocess.CalledProcessError:
        raise DeploymentError("Git is not available or not in a git repository")
    
    # Check if we're on the correct branch
    try:
        result = run_command(["git", "branch", "--show-current"])
        current_branch = result.stdout.strip()
        if current_branch != "main":
            print(f"Warning: Deploying from branch '{current_branch}' (expected 'main')")
    except subprocess.CalledProcessError:
        pass
    
    # Check if all tests pass
    print("Running tests...")
    try:
        run_command(["python", "scripts/run_tests.py", "all", "--fail-under", "80"])
    except DeploymentError:
        raise DeploymentError("Tests failed. Please fix issues before deploying.")
    
    print("✅ Prerequisites check passed")


def build_package() -> None:
    """Build the package."""
    print("Building package...")
    
    # Clean previous builds
    run_command(["rm", "-rf", "build", "dist", "*.egg-info"])
    
    # Build wheel
    run_command(["python", "setup.py", "bdist_wheel"])
    
    # Build source distribution
    run_command(["python", "setup.py", "sdist"])
    
    print("✅ Package built successfully")


def run_security_checks() -> None:
    """Run security checks."""
    print("Running security checks...")
    
    try:
        run_command(["python", "scripts/run_tests.py", "security"])
    except DeploymentError:
        raise DeploymentError("Security checks failed. Please fix issues before deploying.")
    
    print("✅ Security checks passed")


def update_version(version: str) -> None:
    """Update version in relevant files."""
    print(f"Updating version to {version}...")
    
    # Update setup.py
    setup_py_path = Path("setup.py")
    if setup_py_path.exists():
        content = setup_py_path.read_text()
        # Simple version replacement - in production, use more robust parsing
        content = content.replace('version="0.1.0"', f'version="{version}"')
        setup_py_path.write_text(content)
    
    # Update src/__init__.py
    init_py_path = Path("src/__init__.py")
    if init_py_path.exists():
        content = init_py_path.read_text()
        content = content.replace('__version__ = "0.1.0"', f'__version__ = "{version}"')
        init_py_path.write_text(content)
    
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = content.replace('version = "0.1.0"', f'version = "{version}"')
        pyproject_path.write_text(content)
    
    print(f"✅ Version updated to {version}")


def create_git_tag(version: str) -> None:
    """Create and push git tag."""
    print(f"Creating git tag {version}...")
    
    # Create tag
    run_command(["git", "tag", "-a", f"v{version}", "-m", f"Release version {version}"])
    
    # Push tag
    run_command(["git", "push", "origin", f"v{version}"])
    
    print(f"✅ Git tag v{version} created and pushed")


def deploy_to_environment(environment: str, dry_run: bool = False) -> None:
    """Deploy to specific environment."""
    print(f"Deploying to {environment} environment...")
    
    if dry_run:
        print("🔍 DRY RUN - No actual deployment will be performed")
    
    # Load environment-specific configuration
    config_path = Path(f"config/settings.{environment}.yaml")
    if not config_path.exists():
        raise DeploymentError(f"Configuration file for {environment} not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Environment-specific deployment logic
    if environment == "development":
        deploy_to_development(config, dry_run)
    elif environment == "staging":
        deploy_to_staging(config, dry_run)
    elif environment == "production":
        deploy_to_production(config, dry_run)
    else:
        raise DeploymentError(f"Unknown environment: {environment}")
    
    print(f"✅ Deployment to {environment} completed")


def deploy_to_development(config: dict, dry_run: bool = False) -> None:
    """Deploy to development environment."""
    print("Deploying to development environment...")
    
    if dry_run:
        print("Would deploy to development server")
        return
    
    # Install in development mode
    run_command(["pip", "install", "-e", "."])
    
    # Run database migrations if needed
    # run_command(["alembic", "upgrade", "head"])
    
    # Restart development server
    # run_command(["systemctl", "restart", "options-engine-dev"])
    
    print("Development deployment completed")


def deploy_to_staging(config: dict, dry_run: bool = False) -> None:
    """Deploy to staging environment."""
    print("Deploying to staging environment...")
    
    if dry_run:
        print("Would deploy to staging server")
        return
    
    # Build and push Docker image
    # run_command(["docker", "build", "-t", "options-engine:staging", "."])
    # run_command(["docker", "push", "options-engine:staging"])
    
    # Deploy to staging server
    # run_command(["kubectl", "apply", "-f", "k8s/staging/"])
    
    print("Staging deployment completed")


def deploy_to_production(config: dict, dry_run: bool = False) -> None:
    """Deploy to production environment."""
    print("Deploying to production environment...")
    
    if dry_run:
        print("Would deploy to production server")
        return
    
    # Additional production checks
    if not confirm_production_deployment():
        raise DeploymentError("Production deployment cancelled")
    
    # Build and push Docker image
    # run_command(["docker", "build", "-t", "options-engine:latest", "."])
    # run_command(["docker", "push", "options-engine:latest"])
    
    # Deploy to production with blue-green deployment
    # run_command(["kubectl", "apply", "-f", "k8s/production/"])
    
    # Run smoke tests
    # run_smoke_tests()
    
    print("Production deployment completed")


def confirm_production_deployment() -> bool:
    """Confirm production deployment with user."""
    response = input("Are you sure you want to deploy to PRODUCTION? (yes/no): ")
    return response.lower() == "yes"


def run_smoke_tests() -> None:
    """Run smoke tests after deployment."""
    print("Running smoke tests...")
    
    # Run basic health checks
    try:
        run_command(["python", "-c", "from src import __version__; print(__version__)"])
        
        # Add more smoke tests here
        # - Check API endpoints
        # - Verify database connections
        # - Test critical functionality
        
        print("✅ Smoke tests passed")
    except Exception as e:
        raise DeploymentError(f"Smoke tests failed: {e}")


def generate_deployment_report(environment: str, version: str) -> None:
    """Generate deployment report."""
    print("Generating deployment report...")
    
    report = {
        "deployment": {
            "environment": environment,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "deployed_by": os.environ.get("USER", "unknown"),
            "git_commit": get_git_commit_hash(),
            "git_branch": get_git_branch()
        }
    }
    
    # Save report
    report_path = Path(f"deployment_report_{environment}_{version}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Deployment report saved to {report_path}")


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = run_command(["git", "rev-parse", "HEAD"])
        return result.stdout.strip()
    except:
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch."""
    try:
        result = run_command(["git", "branch", "--show-current"])
        return result.stdout.strip()
    except:
        return "unknown"


def rollback_deployment(environment: str, version: str) -> None:
    """Rollback to previous deployment."""
    print(f"Rolling back {environment} deployment to version {version}...")
    
    if environment == "production":
        if not confirm_production_deployment():
            raise DeploymentError("Production rollback cancelled")
    
    # Environment-specific rollback logic
    if environment == "development":
        # Rollback development deployment
        run_command(["git", "checkout", f"v{version}"])
        run_command(["pip", "install", "-e", "."])
    elif environment == "staging":
        # Rollback staging deployment
        # run_command(["kubectl", "rollout", "undo", "deployment/options-engine", "-n", "staging"])
        pass
    elif environment == "production":
        # Rollback production deployment
        # run_command(["kubectl", "rollout", "undo", "deployment/options-engine", "-n", "production"])
        pass
    
    print(f"✅ Rollback to version {version} completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deployment script for options trading engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s deploy development           # Deploy to development
  %(prog)s deploy staging --dry-run     # Dry run staging deployment
  %(prog)s deploy production --version 1.0.0  # Deploy version 1.0.0 to production
  %(prog)s rollback production 0.9.0    # Rollback production to version 0.9.0
  %(prog)s build                        # Build package only
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to environment")
    deploy_parser.add_argument(
        "environment",
        choices=["development", "staging", "production"],
        help="Target environment"
    )
    deploy_parser.add_argument(
        "--version",
        help="Version to deploy (default: current version)"
    )
    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual deployment"
    )
    deploy_parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (not recommended)"
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback deployment")
    rollback_parser.add_argument(
        "environment",
        choices=["development", "staging", "production"],
        help="Target environment"
    )
    rollback_parser.add_argument(
        "version",
        help="Version to rollback to"
    )
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build package")
    build_parser.add_argument(
        "--version",
        help="Version to build (updates version files)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        if args.command == "deploy":
            if not args.skip_tests:
                check_prerequisites()
            
            if args.version:
                update_version(args.version)
            
            build_package()
            run_security_checks()
            deploy_to_environment(args.environment, args.dry_run)
            
            if args.version and not args.dry_run:
                create_git_tag(args.version)
                generate_deployment_report(args.environment, args.version)
        
        elif args.command == "rollback":
            rollback_deployment(args.environment, args.version)
        
        elif args.command == "build":
            if args.version:
                update_version(args.version)
            build_package()
        
        print("\n✅ Deployment script completed successfully!")
        
    except DeploymentError as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()