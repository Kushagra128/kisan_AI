#!/usr/bin/env python
"""
Test Runner Script for Kisan AI Chatbot
Runs all test suites and generates a comprehensive report
"""

import sys
import os
import django
from io import StringIO
import time

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'kisan_project.settings')
django.setup()

from django.test.runner import DiscoverRunner
from django.test.utils import get_runner
from django.conf import settings


def run_all_tests():
    """Run all test suites and print summary"""
    print("=" * 80)
    print("KISAN AI CHATBOT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test suites to run
    test_suites = [
        ('chatbot.test_comprehensive', 'Comprehensive Tests (Unit + Integration)'),
        ('chatbot.test_edge_cases', 'Edge Cases and Boundary Conditions'),
        ('chatbot.test_accuracy', 'Accuracy Evaluation Tests'),
    ]
    
    results = {}
    total_start = time.time()
    
    for test_module, description in test_suites:
        print(f"\n{'─' * 80}")
        print(f"Running: {description}")
        print(f"Module: {test_module}")
        print(f"{'─' * 80}\n")
        
        start_time = time.time()
        
        # Run tests
        TestRunner = get_runner(settings)
        test_runner = TestRunner(verbosity=2, interactive=False, keepdb=True)
        
        try:
            failures = test_runner.run_tests([test_module])
            elapsed = time.time() - start_time
            
            results[description] = {
                'failures': failures,
                'elapsed': elapsed,
                'status': 'PASSED' if failures == 0 else 'FAILED'
            }
        except Exception as e:
            elapsed = time.time() - start_time
            results[description] = {
                'failures': -1,
                'elapsed': elapsed,
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"\n❌ ERROR running {test_module}: {e}\n")
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_passed = 0
    total_failed = 0
    
    for description, result in results.items():
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"\n{status_icon} {description}")
        print(f"   Status: {result['status']}")
        print(f"   Time: {result['elapsed']:.2f}s")
        
        if result['status'] == 'PASSED':
            total_passed += 1
            print(f"   Result: All tests passed")
        elif result['status'] == 'FAILED':
            total_failed += 1
            print(f"   Result: {result['failures']} test(s) failed")
        else:
            total_failed += 1
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'─' * 80}")
    print(f"Total Test Suites: {len(results)}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"{'─' * 80}\n")
    
    # Exit with appropriate code
    if total_failed > 0:
        print("❌ Some tests failed. Please review the output above.")
        sys.exit(1)
    else:
        print("✅ All test suites passed successfully!")
        sys.exit(0)


def run_specific_test(test_path):
    """Run a specific test module or test case"""
    print(f"Running specific test: {test_path}\n")
    
    TestRunner = get_runner(settings)
    test_runner = TestRunner(verbosity=2, interactive=False)
    failures = test_runner.run_tests([test_path])
    
    if failures == 0:
        print("\n✅ Test passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {failures} test(s) failed")
        sys.exit(1)


def run_quick_tests():
    """Run only fast unit tests (skip integration and slow tests)"""
    print("Running quick tests (unit tests only)...\n")
    
    quick_tests = [
        'chatbot.test_comprehensive.DatasetLoaderTests',
        'chatbot.test_comprehensive.AIEngineGuardrailTests',
        'chatbot.test_comprehensive.AIEngineTextProcessingTests',
    ]
    
    TestRunner = get_runner(settings)
    test_runner = TestRunner(verbosity=2, interactive=False, keepdb=True)
    failures = test_runner.run_tests(quick_tests)
    
    if failures == 0:
        print("\n✅ Quick tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {failures} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--quick':
            run_quick_tests()
        elif command == '--help':
            print("Kisan AI Chatbot Test Runner")
            print("\nUsage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py --quick      # Run only quick unit tests")
            print("  python run_tests.py <test_path>  # Run specific test")
            print("\nExamples:")
            print("  python run_tests.py chatbot.test_comprehensive.DatasetLoaderTests")
            print("  python run_tests.py chatbot.test_edge_cases.ExtremeInputTests.test_empty_string_query")
            sys.exit(0)
        else:
            # Run specific test
            run_specific_test(command)
    else:
        # Run all tests
        run_all_tests()
