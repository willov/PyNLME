#!/usr/bin/env python3
"""
Verification script for PyNLME installation from GitHub releases.
Run this after installing PyNLME to verify everything works correctly.
"""

import sys
import traceback


def test_import():
    """Test that PyNLME can be imported successfully."""
    print("🔍 Testing PyNLME import...")
    try:
        import pynlme

        print(f"✅ PyNLME imported successfully (version: {pynlme.__version__})")
        return True
    except ImportError as e:
        print(f"❌ Failed to import PyNLME: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with a simple example."""
    print("\n🧪 Testing basic functionality...")
    try:
        import numpy as np

        from pynlme import nlmefit

        # Simple test data
        def simple_model(phi, x, v=None):
            return phi[0] + phi[1] * x.ravel()

        x = np.array([[1], [2], [3], [4]])
        y = np.array([2.1, 4.0, 5.9, 8.1])
        group = np.array([0, 0, 1, 1])
        beta0 = np.array([1.0, 2.0])

        # This should work without errors
        beta, psi, stats, b = nlmefit(x, y, group, None, simple_model, beta0)
        print("✅ Basic NLME fitting works correctly")
        print(f"   Fixed effects: {beta}")
        print(f"   Log-likelihood: {stats.logl:.2f}")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("🔧 PyNLME Installation Verification")
    print("=" * 40)

    tests = [
        ("Import Test", test_import),
        ("Basic Functionality Test", test_basic_functionality),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 40)
    print("📋 Verification Summary:")

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! PyNLME is ready to use.")
        print("\n📚 Next steps:")
        print(
            "  - Check out examples: https://github.com/willov/PyNLME/tree/main/examples"
        )
        print(
            "  - Read documentation: https://github.com/willov/PyNLME/blob/main/docs/README.md"
        )
        return 0
    else:
        print("\n⚠️  Some tests failed. PyNLME may not work correctly.")
        print("   Please check the installation or report an issue at:")
        print("   https://github.com/willov/PyNLME/issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
