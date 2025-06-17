//! PyNLME Core - Rust implementation of nonlinear mixed-effects models
//!
//! This module provides high-performance implementations of NLME algorithms
//! including MLE and SAEM, with Python bindings via PyO3.

#![allow(
    unused_imports,
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    non_snake_case
)]

use pyo3::prelude::*;

mod errors;
mod mle;
mod mle_batched;
mod nlme;
mod saem;
mod utils;

use mle::*;
use nlme::*;
use saem::*;

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
