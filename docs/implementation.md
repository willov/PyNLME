# PyNLME Implementation Details

## Architecture Overview

PyNLME is built with a hybrid Python/Rust architecture for optimal performance and usability:

```
┌─────────────────────────────────────────────────────────┐
│                  Python API Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   nlmefit   │  │  nlmefitsa  │  │ Utilities   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                 Algorithm Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ MLEFitter   │  │ SAEMFitter  │  │ ErrorModel  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                  Rust Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ fit_nlme_   │  │ fit_nlme_   │  │ Optimization│     │
│  │    mle      │  │   saem      │  │  Routines   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Algorithm Implementations

### Maximum Likelihood Estimation (MLE)

The MLE algorithm uses an iterative optimization approach:

1. **Linearization**: Model is linearized around current parameter estimates
2. **Random Effects Estimation**: Individual random effects are estimated
3. **Parameter Update**: Fixed effects and covariance parameters are updated
4. **Convergence Check**: Process repeats until convergence criteria are met

**Rust Implementation Highlights:**
- Uses L-BFGS-B optimization from `nalgebra` and `optimization` crates
- Automatic differentiation for gradient computation
- Robust numerical algorithms with fallback strategies
- Memory-efficient sparse matrix operations

```rust
// Core MLE fitting loop (simplified)
impl MLEFitter {
    pub fn fit(&self, data: &NLMEData) -> Result<NLMEResult> {
        let mut beta = data.beta0.clone();
        let mut psi = Matrix::identity(beta.len());
        
        for iteration in 0..self.options.max_iter {
            // E-step: Estimate random effects
            let random_effects = self.estimate_random_effects(&beta, &psi, data)?;
            
            // M-step: Update parameters
            let (new_beta, new_psi) = self.update_parameters(&random_effects, data)?;
            
            // Check convergence
            if self.has_converged(&beta, &new_beta) {
                return Ok(self.build_result(new_beta, new_psi, data));
            }
            
            beta = new_beta;
            psi = new_psi;
        }
        
        Err(NLMEError::ConvergenceFailed)
    }
}
```

### Stochastic Approximation EM (SAEM)

SAEM algorithm implements a three-phase approach:

1. **Burn-in Phase**: Initial exploration with large step sizes
2. **Stochastic Phase**: MCMC sampling with decreasing step sizes  
3. **Smoothing Phase**: Final convergence with small step sizes

**Key Features:**
- Monte Carlo sampling for random effects
- Adaptive step size scheduling
- Robust convergence monitoring
- Memory-efficient MCMC chains

### Error Models

Supports multiple error variance models:

- **Constant**: `Var(ε) = σ²`
- **Proportional**: `Var(ε) = (σ × f)²`
- **Combined**: `Var(ε) = (σ₁ + σ₂ × f)²`
- **Exponential**: `Var(ε) = f^(2σ)`

Where `f` is the predicted value.

## Performance Optimizations

### Rust Backend Benefits

1. **Speed**: 10-100x faster than pure Python for large datasets
2. **Memory**: Efficient memory usage with zero-copy operations
3. **Parallelization**: Ready for multi-threaded optimization
4. **Numerical Stability**: Robust algorithms from proven crates

### Python Integration

- **PyO3 Bindings**: Seamless Python/Rust interop
- **NumPy Integration**: Zero-copy array sharing
- **Automatic Fallback**: Falls back to Python if Rust fails
- **Error Propagation**: Rust errors properly converted to Python exceptions

## Data Flow

```
Input Data (Python)
      ↓
Input Validation (Python)
      ↓
Type Conversion (PyO3)
      ↓
Core Algorithm (Rust)
      ↓
Result Conversion (PyO3)
      ↓
Statistics Computation (Python)
      ↓
Output (Python)
```

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual algorithm components
2. **Integration Tests**: End-to-end API testing
3. **Numerical Tests**: Accuracy against known solutions
4. **Performance Tests**: Speed and memory benchmarks
5. **Compatibility Tests**: MATLAB result comparison

### Test Data

- **Synthetic Data**: Controlled scenarios with known answers
- **Real Pharmacokinetic Data**: Industry-standard datasets
- **Edge Cases**: Boundary conditions and error scenarios

### Continuous Integration

```yaml
# Simplified CI pipeline
- Python 3.8-3.12 compatibility
- Rust stable/beta testing  
- Multiple OS support (Linux, macOS, Windows)
- Performance regression detection
- Memory leak checking
```

## MATLAB Compatibility

### API Compatibility

The package provides 1:1 function compatibility with MATLAB:

```python
# MATLAB
[beta, psi, stats, b] = nlmefit(X, y, group, [], model, beta0, 'Options', opts);

# PyNLME (identical interface)
beta, psi, stats, b = nlmefit(X, y, group, None, model, beta0, **opts)
```

### Result Structure Compatibility

| MATLAB Field | PyNLME Equivalent | Description |
|--------------|-------------------|-------------|
| `stats.dfe` | `stats.dfe` | Degrees of freedom |
| `stats.logl` | `stats.logl` | Log-likelihood |
| `stats.aic` | `stats.aic` | AIC |
| `stats.bic` | `stats.bic` | BIC |
| `stats.rmse` | `stats.rmse` | RMSE |
| `stats.ires` | `stats.ires` | Individual residuals |
| `stats.pres` | `stats.pres` | Population residuals |

## Build System

### Maturin Integration

The package uses Maturin for Rust/Python integration:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "src"
manifest-path = "Cargo.toml"
module-name = "pynlme._core"
features = ["pyo3/extension-module"]
```

### Development Workflow

```bash
# Development setup
uv sync                    # Install Python dependencies
maturin develop           # Build Rust extension in-place
uv run pytest           # Run test suite

# Release build
maturin build --release   # Build optimized wheel
maturin publish          # Publish to PyPI
```

## Dependencies

### Python Dependencies

- **NumPy** (≥2.0): Array operations and numerical computing
- **SciPy** (≥1.7): Optimization and statistical functions
- **PyO3** (via Maturin): Python/Rust bindings

### Rust Dependencies

- **nalgebra**: Linear algebra operations
- **ndarray**: N-dimensional arrays
- **rayon**: Data parallelism
- **serde**: Serialization
- **thiserror**: Error handling

## Error Handling Strategy

### Python Layer
- Input validation with clear error messages
- Type checking and conversion
- Graceful degradation to fallback implementations

### Rust Layer  
- Comprehensive error types with context
- Proper error propagation through `Result<T, E>`
- Memory safety guarantees

### Integration Layer
- Automatic conversion of Rust errors to Python exceptions
- Preservation of error context and stack traces
- Fallback mechanisms for robust operation

## Future Enhancements

### Planned Features
- **Parallel Group Processing**: Multi-threaded fitting for multiple groups
- **GPU Acceleration**: CUDA/OpenCL support for large datasets
- **Model Selection**: Automatic model comparison and selection
- **Bayesian Methods**: MCMC-based parameter estimation

### Performance Targets
- **10x faster**: For typical pharmacokinetic datasets (100-1000 subjects)
- **Memory efficient**: Constant memory usage regardless of dataset size
- **Scalable**: Linear scaling with number of groups

### Compatibility Goals
- **R nlme**: Cross-validation with R's nlme package
- **NONMEM**: Result comparison with NONMEM software
- **Stan**: Bayesian modeling compatibility
