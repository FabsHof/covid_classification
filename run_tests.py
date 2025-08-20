#!/usr/bin/env python3
"""
Simple test runner script to verify the test infrastructure works.
Run this to check if all tests are passing.
"""
import subprocess
import sys
from pathlib import Path


def run_tests(with_coverage=False):
    """Run the unit tests and return the result."""
    project_root = Path(__file__).parent

    print("🧪 Running unit tests for util modules...")
    print(f"📁 Project root: {project_root}")

    # Build command
    cmd = ["uv", "run", "pytest", "tests/", "-v"]
    if with_coverage:
        cmd.extend(
            ["--cov=src", "--cov-report=html", "--cov-report=term-missing"]
        )

    try:
        # Run tests using uv
        result = subprocess.run(
            cmd, cwd=project_root, capture_output=True, text=True, timeout=120
        )

        print("\n📊 Test Results:")
        print("=" * 50)
        print(result.stdout)

        if result.stderr:
            print("\n⚠️  Errors/Warnings:")
            print(result.stderr)

        if result.returncode == 0:
            print("\n✅ All tests passed!")
            if with_coverage:
                print("📈 Coverage report generated in htmlcov/")
            return True
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("\n⏰ Tests timed out after 120 seconds")
        return False
    except FileNotFoundError:
        print("\n❌ uv command not found. Please install uv package manager.")
        return False
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False


def main():
    """Main function."""
    print("🚀 COVID Classification - Test Runner")
    print("=" * 50)

    # Check if coverage flag is provided
    with_coverage = "--coverage" in sys.argv

    if with_coverage:
        print("📈 Running tests with coverage analysis...")

    success = run_tests(with_coverage)

    if success:
        print("\n🎉 Test infrastructure is working correctly!")
        print("\nUsage:")
        print("  python run_tests.py           # Run tests without coverage")
        print("  python run_tests.py --coverage # Run tests with coverage")
        sys.exit(0)
    else:
        print("\n💥 Test infrastructure needs attention.")
        print("Check the output above for details on failing tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
