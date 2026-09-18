"""
tests/run_all_tests.py - Unified Automated Test Suite Runner.

Executes all unit and integration test cases across Database, Computer Vision Recognition,
Service decision pipelines, and Bulk Photo Import. Outputs an audit report table.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.create_test_assets import setup_test_assets


def run_all_tests():
    """Run all test suites and present consolidated execution summary."""
    print("=" * 80)
    print("BIOMETRIC-BASED EXAM AUTHENTICATION SYSTEM — AUTOMATED TEST SUITE")
    print("=" * 80)

    # Ensure test assets exist
    setup_test_assets()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover and add test modules
    suite.addTests(loader.discover(start_dir=str(BASE_DIR / "tests"), pattern="test_*.py"))

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time

    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed:          {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:        {len(result.failures)}")
    print(f"Errors:          {len(result.errors)}")
    print(f"Duration:        {duration:.2f} seconds")
    print("=" * 80)

    if result.wasSuccessful():
        print(">>> ALL TESTS PASSED SUCCESSFULLY! <<<\n")
        return 0
    else:
        print(">>> SOME TESTS FAILED! <<<\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
