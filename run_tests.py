#!/usr/bin/env python3
"""
Test runner for Virtual Context MCP Server

This script provides various test execution options including unit tests,
integration tests, performance benchmarks, and coverage reports.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for Virtual Context MCP Server")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (skip slow tests)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--pattern", "-k", type=str, help="Run tests matching pattern")
    
    args = parser.parse_args()
    
    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    # Add test selection
    if args.unit:
        pytest_cmd.append("tests/unit/")
    elif args.integration:
        pytest_cmd.append("tests/integration/")
    elif args.performance:
        pytest_cmd.extend(["-m", "performance"])
    else:
        pytest_cmd.append("tests/")
    
    # Add markers
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Add pattern matching
    if args.pattern:
        pytest_cmd.extend(["-k", args.pattern])
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src/virtual_context_mcp",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Run tests
    success = run_command(pytest_cmd, "Test Execution")
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")
    
    # Summary
    if success:
        print(f"\n🎉 All tests completed successfully!")
        return 0
    else:
        print(f"\n💥 Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())