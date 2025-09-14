#!/usr/bin/env python3
"""
Test Runner for RAG System
This script provides a comprehensive test runner with different test configurations.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
from typing import List, Dict, Any

def run_tests(test_type: str = "all", verbose: bool = True, coverage: bool = False) -> Dict[str, Any]:
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run (unit, integration, performance, all)
        verbose: Whether to run in verbose mode
        coverage: Whether to generate coverage report
        
    Returns:
        Dictionary with test results
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test type markers
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    # For "all", don't add any markers
    
    # Add additional options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    # Parse results
    test_results = {
        'test_type': test_type,
        'return_code': result.returncode,
        'duration': end_time - start_time,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }
    
    return test_results

def run_specific_tests(test_files: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Run specific test files.
    
    Args:
        test_files: List of test files to run
        verbose: Whether to run in verbose mode
        
    Returns:
        Dictionary with test results
    """
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(test_files)
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"Running specific tests: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    test_results = {
        'test_files': test_files,
        'return_code': result.returncode,
        'duration': end_time - start_time,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }
    
    return test_results

def run_performance_tests() -> Dict[str, Any]:
    """Run performance tests with detailed metrics."""
    print("Running Performance Tests")
    print("=" * 40)
    
    # Performance test configuration
    perf_cmd = [
        "python", "-m", "pytest",
        "tests/test_rag_system.py::TestPerformance",
        "-v",
        "--tb=short",
        "-m", "performance"
    ]
    
    start_time = time.time()
    result = subprocess.run(perf_cmd, capture_output=True, text=True)
    end_time = time.time()
    
    return {
        'test_type': 'performance',
        'return_code': result.returncode,
        'duration': end_time - start_time,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def run_integration_tests() -> Dict[str, Any]:
    """Run integration tests."""
    print("Running Integration Tests")
    print("=" * 40)
    
    integration_cmd = [
        "python", "-m", "pytest",
        "tests/test_rag_system.py::TestIntegration",
        "-v",
        "--tb=short",
        "-m", "integration"
    ]
    
    start_time = time.time()
    result = subprocess.run(integration_cmd, capture_output=True, text=True)
    end_time = time.time()
    
    return {
        'test_type': 'integration',
        'return_code': result.returncode,
        'duration': end_time - start_time,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def generate_test_report(results: List[Dict[str, Any]], output_file: Path = None):
    """Generate a comprehensive test report."""
    if output_file is None:
        output_file = Path("test_report.json")
    
    report = {
        'timestamp': time.time(),
        'total_tests': len(results),
        'successful_tests': sum(1 for r in results if r['success']),
        'failed_tests': sum(1 for r in results if not r['success']),
        'total_duration': sum(r['duration'] for r in results),
        'test_results': results
    }
    
    # Calculate success rate
    report['success_rate'] = report['successful_tests'] / report['total_tests'] if report['total_tests'] > 0 else 0
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Test report saved to: {output_file}")
    return report

def print_test_summary(results: List[Dict[str, Any]]):
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_tests - successful
    total_duration = sum(r['duration'] for r in results)
    
    print(f"Total test suites: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total_tests*100:.1f}%")
    print(f"Total duration: {total_duration:.2f} seconds")
    
    print("\nDetailed Results:")
    print("-" * 40)
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{result.get('test_type', 'unknown').upper():<15} {status:<5} {result['duration']:.2f}s")
        
        if not result['success'] and result['stderr']:
            print(f"  Error: {result['stderr'][:100]}...")

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "performance", "slow"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific test files to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Run in verbose mode"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output file for test report"
    )
    
    args = parser.parse_args()
    
    results = []
    
    try:
        if args.files:
            # Run specific test files
            result = run_specific_tests(args.files, args.verbose)
            results.append(result)
        else:
            # Run tests by type
            if args.type == "all":
                # Run all test types
                test_types = ["unit", "integration", "performance"]
                for test_type in test_types:
                    result = run_tests(test_type, args.verbose, args.coverage)
                    results.append(result)
            else:
                # Run specific test type
                result = run_tests(args.type, args.verbose, args.coverage)
                results.append(result)
        
        # Print summary
        print_test_summary(results)
        
        # Generate report if requested
        if args.report:
            report = generate_test_report(results, Path(args.report))
        else:
            report = generate_test_report(results)
        
        # Exit with appropriate code
        all_successful = all(r['success'] for r in results)
        sys.exit(0 if all_successful else 1)
        
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
