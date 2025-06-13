# PyNLME - Nonlinear Mixed-Effects Models for Python

[![Tests](https://img.shields.io/badge/tests-48%2F48%20passing-brightgreen)]()
[![Rust Backend](https://img.shields.io/badge/rust%20backend-enabled-orange)]()
[![MATLAB Compatible](https://img.shields.io/badge/MATLAB-compatible-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

A high-performance Python implementation of nonlinear mixed-effects (NLME) models with MATLAB-compatible API and Rust backend for optimization.

## AI Disclaimer

I (willov) asked Claude Sonnet 4 in GitHub Copilot agent mode to "create a Python NLME implementation with a rust backend with a similar call syntax to MATLAB's nlmefitsa". Please use the package with caution, it is very likely that the AI have hallucinated quirks not tested. If you find it useful, please assist in creating tests to test the reliability of the package.

At the current state, this should be seen as a proof-of-concept rather than production ready.

## ⚠️ Current Status

**Proof of Concept**: The package is functional with 48/48 tests passing, but requires thorough validation before production use. Both Rust and Python backends work, with automatic fallback capabilities. All core algorithms (MLE and SAEM) are implemented and appear to work correctly in test scenarios.

## 🚀 Quick Start

```python
import numpy as np
from pynlme import nlmefit

# Define your model
def exponential_decay(phi, t, v=None):
    return phi[0] * np.exp(-phi[1] * t.ravel())

# Fit the model (uses optimized Rust backend with Python fallback)
beta, psi, stats, b = nlmefit(t, y, group, None, exponential_decay, [10.0, 0.5])
print(f"Fixed effects: {beta}, Log-likelihood: {stats.logl}")
```

## 📦 Installation

### Option 1: Install from GitHub Release (Recommended)

Pre-built wheels are available for easy installation without requiring Rust:

```bash
# Latest release (v0.2.0+):
# Choose the appropriate wheel for your platform:

# Linux (x86_64):
uv add https://github.com/willov/PyNLME/releases/download/v0.2.0/pynlme-0.2.0-cp311-abi3-linux_x86_64.whl

# Windows (x86_64):
uv add https://github.com/willov/PyNLME/releases/download/v0.2.0/pynlme-0.2.0-cp311-abi3-win_amd64.whl

# macOS (Intel):
uv add https://github.com/willov/PyNLME/releases/download/v0.2.0/pynlme-0.2.0-cp311-abi3-macosx_10_12_x86_64.whl

# macOS (Apple Silicon):
uv add https://github.com/willov/PyNLME/releases/download/v0.2.0/pynlme-0.2.0-cp311-abi3-macosx_11_0_arm64.whl

# Or using pip instead of uv:
pip install https://github.com/willov/PyNLME/releases/download/v0.2.0/pynlme-0.2.0-cp311-abi3-linux_x86_64.whl

# Or browse releases to find the exact wheel for your platform:
# https://github.com/willov/PyNLME/releases/latest
```

**Platform compatibility:**
- **Python**: 3.11, 3.12, 3.13+ (single wheel works for all)
- **Linux**: x86_64 (Intel/AMD 64-bit)
- **Windows**: x86_64 (Intel/AMD 64-bit)  
- **macOS**: Intel x86_64 and Apple Silicon ARM64

### Option 2: Install from source (for developers)

```bash
# Requires Rust toolchain for building
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

git clone https://github.com/willov/PyNLME
cd PyNLME
uv sync
uv run maturin develop  # Build Rust extension
```

> **Note**: The `cp311-abi3` wheels use Python's stable ABI and work with Python 3.11+ including future versions. You don't need separate wheels for each Python version. Option 2 is for developers who want to build from source or contribute to the project.

## 🎯 Features

- **🔥 Fast**: Rust backend with automatic fallback to Python
- **🔄 MATLAB Compatible**: Drop-in replacement for `nlmefit`/`nlmefitsa`
- **🧪 Robust**: 48/48 tests passing, comprehensive error handling
- **📊 Complete**: Full diagnostics, residuals, and model statistics

## 📚 Documentation

- **[Complete Documentation](docs/README.md)** - Full feature overview and examples
- **[API Reference](docs/api_reference.md)** - Detailed function documentation  
- **[Implementation Details](docs/implementation.md)** - Technical architecture
- **[Examples](examples/)** - Usage examples and tutorials
- **[Changelog](CHANGELOG.md)** - Version history and release notes

## 🧪 Quick Test

```bash
# Run the full test suite (for developers)
uv run pytest tests/ -v  # All 48 tests should pass

# Verify installation works (for end users)
python -c "
import pynlme
print(f'PyNLME {pynlme.__version__} installed successfully!')
"

# Or use the comprehensive verification script
curl -sSL https://raw.githubusercontent.com/willov/PyNLME/main/scripts/verify-installation.py | python3
```

## 💡 Important Note

When using UV for package management, run Python scripts with `uv run python script.py` to ensure proper access to installed packages.

## 📄 License

MIT License - see LICENSE file for details.

## ✅ Current Features (Working)

- **MATLAB-Compatible API**: Identical interface to MATLAB's `nlmefit` and `nlmefitsa`
- **High-Performance Rust Backend**: Core algorithms implemented in Rust with automatic Python fallback
- **Multiple Algorithms**: Both MLE and SAEM (Stochastic Approximation EM) algorithms with full Rust backend support  
- **Robust Implementation**: Proper convergence checking and error handling
- **Comprehensive Statistics**: AIC, BIC, log-likelihood, residuals, and more
- **Type Safety**: Full type annotations and data validation
- **Real-World Tested**: Validated with pharmacokinetic modeling scenarios

## 🚧 In Development

- **Advanced Error Models**: Proportional, combined, and exponential error models
- **Parameter Transformations**: Log, probit, and logit transformations
- **Enhanced Diagnostics**: Additional model validation tools

## 📖 References

This implementation is based on the following papers:

1. Lindstrom, M. J., and D. M. Bates. "Nonlinear mixed-effects models for repeated measures data." Biometrics. Vol. 46, 1990, pp. 673–687.
2. Davidian, M., and D. M. Giltinan. Nonlinear Models for Repeated Measurements Data. New York: Chapman & Hall, 1995.
3. Pinheiro, J. C., and D. M. Bates. "Approximations to the log-likelihood function in the nonlinear mixed-effects model." Journal of Computational and Graphical Statistics. Vol. 4, 1995, pp. 12–35.
4. Demidenko, E. Mixed Models: Theory and Applications. Hoboken, NJ: John Wiley & Sons, Inc., 2004.
