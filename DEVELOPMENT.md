# Developer Quick Start Guide

Welcome to PyNLME development! This guide will help you get set up quickly.

## 🛠️ Prerequisites

- **Python ≥3.11**
- **Rust ≥1.65** ([Install Rust](https://rustup.rs/))
- **UV package manager** ([Install UV](https://docs.astral.sh/uv/getting-started/installation/))
- **Git**

## 🚀 Quick Setup

```bash
# 1. Clone and navigate
git clone https://github.com/willov/PyNLME
cd PyNLME

# 2. Install dependencies and build
uv sync --dev
uv run maturin develop

# 3. Run tests to verify setup
uv run pytest tests/ -v

# 4. Optional: Set up pre-commit hooks
uv run pre-commit install
```

## 🧪 Development Commands

```bash
# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=pynlme --cov-report=html

# Type checking
uv run mypy src/pynlme/

# Format code
uv run ruff format src/ tests/ examples/

# Lint code  
uv run ruff check src/ tests/ examples/

# Build Rust extension
uv run maturin develop

# Build distribution packages
uv run maturin build --release

# Run examples
cd examples/ && python basic_usage.py
```

## 📁 Project Structure

```
PyNLME/
├── src/
│   ├── _core/          # Rust implementation
│   └── pynlme/         # Python package
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/               # Documentation
├── scripts/            # Utility scripts (e.g., verify-installation.py)
└── .github/            # CI/CD workflows
```

## 🔧 Development Workflow

1. **Create feature branch**: `git checkout -b feature-name`
2. **Make changes** to Python/Rust code
3. **Add tests** for new functionality
4. **Run test suite**: `uv run pytest`
5. **Format code**: `uv run ruff format .`
6. **Update documentation** if needed
7. **Commit and push**: `git commit -am "Description"`
8. **Create Pull Request**

## 🧩 Working with the Rust Backend

The Rust code is in `src/_core/`. Key files:
- `lib.rs` - Python bindings
- `nlme.rs` - Core NLME algorithms
- `mle.rs` - Maximum likelihood estimation
- `saem.rs` - SAEM algorithm

After Rust changes:
```bash
# Rebuild extension
uv run maturin develop

# Run tests to ensure compatibility
uv run pytest tests/test_algorithms.py -v
```

## 🐛 Debugging Tips

### Python Debugging
```bash
# Run specific test with verbose output
uv run pytest tests/test_basic.py::test_nlmefit -v -s

# Debug with IPython
uv add ipython
uv run python -c "import pynlme; pynlme.nlmefit(...)"
```

### Rust Debugging
```bash
# Check Rust compilation
cargo check

# Run Rust tests
cargo test

# Format Rust code
cargo fmt
```

## 📊 Performance Testing

```bash
# Run benchmark example
cd examples/
python performance_benchmark.py

# Profile Python code
uv add cProfile
python -m cProfile -o profile.stats your_script.py
```

## 📝 Documentation Updates

- **API changes**: Update `docs/api_reference.md`
- **New features**: Add to `docs/README.md`
- **Breaking changes**: Update `CHANGELOG.md`
- **Examples**: Add to `examples/` with descriptive README

## 🚢 Release Process

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit and push to main

# GitHub Actions automatically:
# - Detects version change
# - Runs comprehensive tests
# - Creates release with wheels if tests pass
```

## 🤝 Contribution Guidelines

- **Follow PEP 8** for Python code style
- **Add docstrings** to all public functions
- **Include type hints** for function signatures
- **Write tests** for new functionality
- **Update documentation** for user-facing changes
- **Keep commits atomic** and well-described

## 🆘 Getting Help

- **Check documentation**: `docs/` folder
- **Run examples**: `examples/` folder
- **Ask questions**: Open a GitHub issue
- **Check tests**: Look at `tests/` for usage patterns

## 🎯 Common Tasks

### Adding a New Algorithm
1. Implement in `src/_core/` (Rust)
2. Add Python wrapper in `src/pynlme/algorithms.py`
3. Add tests in `tests/test_algorithms.py`
4. Update API documentation
5. Add example usage

### Adding a New Error Model
1. Update `ErrorModel` enum in `src/pynlme/data_types.py`
2. Implement in Rust backend
3. Add tests in `tests/test_data_types.py`
4. Update documentation

### Optimizing Performance
1. Profile with `cProfile` or `py-spy`
2. Implement bottlenecks in Rust
3. Benchmark with `examples/performance_benchmark.py`
4. Ensure backward compatibility

## Wheel Distribution Strategy

PyNLME uses pre-built wheels to eliminate the need for users to have Rust installed. This provides several benefits:

### **Benefits of Wheel Distribution:**
- ✅ **Easy installation**: Users can install from GitHub releases 
- ✅ **No Rust required**: Pre-compiled binaries included
- ✅ **Fast installation**: No compilation step
- ✅ **Cross-platform**: Wheels for Linux, macOS, and Windows
- ✅ **Multiple Python versions**: Support for Python 3.11+

### **CI/CD Pipeline:**
The project uses GitHub Actions with `cibuildwheel` to:
1. **Build wheels** on every release tag for multiple platforms
2. **Test wheels** to ensure they work correctly  
3. **Upload to PyPI** automatically on tagged releases
4. **Create GitHub releases** with wheel artifacts

### **Local Wheel Building:**
```bash
# Build wheels locally for testing
uv run cibuildwheel --output-dir wheelhouse

# Test a local wheel
uv add wheelhouse/pynlme-*.whl
```

### **Release Process:**

PyNLME uses **manual version control** with automatic release creation:

#### **Manual Version Release (Recommended):**

1. **Update version manually** in `pyproject.toml`:

   ```toml
   [project]
   version = "0.1.1"  # Change from 0.1.0 to 0.1.1
   ```

2. **Update changelog** in `CHANGELOG.md`:

   ```markdown
   ## [0.1.1] - 2025-06-11
   ### Added
   - New feature X
   ### Fixed  
   - Bug fix Y
   ```

3. **Commit and push**:

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release v0.1.1: Add feature X, fix bug Y"
   git push origin main
   ```

4. **🤖 GitHub Actions automatically**:
   - Detects version change in `pyproject.toml`
   - Runs tests and validation
   - Creates git tag `v0.1.1`
   - Builds wheels for all platforms
   - Creates GitHub release with artifacts

#### **What Happens:**

- Users can install with pre-built wheels from GitHub releases (no Rust needed!)
- When ready for wider adoption, can enable PyPI publishing

Happy coding! 🎉
