#!/usr/bin/env python3
"""Test runner script for Harmony Guard test suite."""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Harmony Guard test suite")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific category is selected
    if not any([args.unit, args.integration, args.performance, args.security]):
        args.all = True
    
    results = []
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    if args.verbose:
        base_cmd.append("-v")
    
    if args.coverage:
        base_cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Run unit tests
    if args.unit or args.all:
        cmd = base_cmd + ["-m", "unit", "tests/"]
        success = run_command(cmd, "Unit Tests")
        results.append(("Unit Tests", success))
    
    # Run integration tests
    if args.integration or args.all:
        cmd = base_cmd + ["-m", "integration", "tests/"]
        success = run_command(cmd, "Integration Tests")
        results.append(("Integration Tests", success))
    
    # Run performance tests
    if args.performance or args.all:
        cmd = base_cmd + ["-m", "performance", "tests/"]
        success = run_command(cmd, "Performance Tests")
        results.append(("Performance Tests", success))
    
    # Run security tests
    if args.security or args.all:
        cmd = base_cmd + ["-m", "security", "tests/"]
        success = run_command(cmd, "Security Tests")
        results.append(("Security Tests", success))
    
    # If running all tests without markers, run everything
    if args.all and not any([args.unit, args.integration, args.performance, args.security]):
        cmd = base_cmd + ["tests/"]
        success = run_command(cmd, "All Tests")
        results.append(("All Tests", success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 All test suites passed!")
        return 0
    else:
        print("💥 Some test suites failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())