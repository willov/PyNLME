# Contributing to PyNLME

Thank you for your interest in contributing to PyNLME! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python ≥3.11
- Rust ≥1.65
- UV package manager (recommended)

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/willov/PyNLME
cd PyNLME

# Install dependencies
uv sync --dev

# Build the Rust extension
uv run maturin develop

# Run tests
uv run pytest
```

## Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all public functions and classes
- Maximum line length: 88 characters

### Rust
- Follow standard Rust conventions (`cargo fmt`)
- Use `cargo clippy` for linting
- Add documentation comments for public APIs

## Testing

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pynlme

# Run specific test file
uv run pytest tests/test_fitting.py
```

### Writing Tests
- Add tests for all new functionality
- Use descriptive test names
- Include edge cases and error conditions
- Test both Python and Rust backends when applicable

## Documentation

### API Documentation
- Update `docs/api_reference.md` for new functions
- Include examples in docstrings
- Document all parameters and return values

### Implementation Documentation
- Update `docs/implementation.md` for architectural changes
- Document algorithm changes and optimizations

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** the test suite (`pytest`)
6. **Update** documentation as needed
7. **Commit** your changes (`git commit -m 'Add amazing feature'`)
8. **Push** to the branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

### Pull Request Guidelines
- Provide a clear description of the changes
- Reference any related issues
- Include test results
- Update CHANGELOG.md for notable changes

## Reporting Issues

### Bug Reports
When filing a bug report, please include:
- Python and Rust versions
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests
When requesting features, please include:
- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)

## Code of Conduct

This project follows a standard code of conduct:
- Be respectful and inclusive
- Assume good intentions
- Provide constructive feedback
- Help others learn and grow

## Getting Help

- **Documentation**: Check the `docs/` folder
- **Examples**: Look at the `examples/` folder
- **Issues**: Search existing issues first
- **Discussions**: Start a discussion for questions

## Development Tips

### Performance
- Profile code changes with realistic datasets
- Benchmark against existing implementations
- Consider both single-threaded and multi-threaded scenarios

### Algorithm Development
- Implement in Python first for correctness
- Port to Rust for performance
- Maintain feature parity between backends

### Testing Large Datasets
```bash
# Generate test data
python examples/generate_test_data.py

# Run performance tests
python examples/benchmark.py
```

Thank you for contributing to PyNLME! 🎉
