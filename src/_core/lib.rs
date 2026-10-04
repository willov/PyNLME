//! PyNLME Core - Rust implementation of nonlinear mixed-effects models
//!
//! Currently only the experimental SAEM fitter lives here; nlmefit is
//! implemented in Python (src/pynlme/lindstrom_bates.py).

#![allow(
    unused_imports,
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    non_snake_case
)]

use pyo3::prelude::*;

mod errors;
mod nlme;
mod saem;
mod utils;

use nlme::*;
use saem::*;

/// Python module for PyNLME core functionality
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NLMEResult>()?;
    m.add_class::<SAEMOptions>()?;

    m.add_function(wrap_pyfunction!(fit_nlme_saem, m)?)?;

    Ok(())
}
