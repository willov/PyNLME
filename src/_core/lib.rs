//! PyNLME Core - Rust implementation of nonlinear mixed-effects models
//! 
//! This module provides high-performance implementations of NLME algorithms
//! including MLE and SAEM, with Python bindings via PyO3.

use pyo3::prelude::*;

mod nlme;
mod saem;
mod mle;
mod utils;
mod errors;

use nlme::*;
use saem::*;
use mle::*;

/// Python module for PyNLME core functionality
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NLMEResult>()?;
    m.add_class::<SAEMOptions>()?;
    m.add_class::<MLEOptions>()?;
    
    m.add_function(wrap_pyfunction!(fit_nlme_mle, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nlme_saem, m)?)?;
    
    Ok(())
}
