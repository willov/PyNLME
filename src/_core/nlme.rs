//! Nonlinear Mixed-Effects Model types and structures

#![allow(clippy::manual_clamp, dead_code, non_local_definitions)]

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

/// Result structure for NLME fitting
#[pyclass]
#[derive(Clone)]
pub struct NLMEResult {
    /// Fixed-effects parameter estimates
    #[pyo3(get, set)]
    pub beta: PyObject,

    /// Random-effects covariance matrix
    #[pyo3(get, set)]
    pub psi: PyObject,

    /// Log-likelihood
    #[pyo3(get, set)]
    pub logl: Option<f64>,

    /// AIC
    #[pyo3(get, set)]
    pub aic: Option<f64>,

    /// BIC
    #[pyo3(get, set)]
    pub bic: Option<f64>,

    /// RMSE
    #[pyo3(get, set)]
    pub rmse: Option<f64>,

    /// Standard errors of fixed effects
    #[pyo3(get, set)]
    pub se_beta: Option<PyObject>,

    /// Random effects estimates
    #[pyo3(get, set)]
    pub random_effects: Option<PyObject>,

    /// Residuals
    #[pyo3(get, set)]
    pub residuals: Option<PyObject>,
}

#[pymethods]
impl NLMEResult {
    #[new]
    pub fn new() -> Self {
        Python::with_gil(|py| Self {
            beta: PyArray1::<f64>::zeros(py, 0, false).to_object(py),
            psi: PyArray2::<f64>::zeros(py, (0, 0), false).to_object(py),
            logl: None,
            aic: None,
            bic: None,
            rmse: None,
            se_beta: None,
            random_effects: None,
            residuals: None,
        })
    }
}

impl Default for NLMEResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Error model types
#[derive(Clone, Debug)]
pub enum ErrorModel {
    Constant { sigma: f64 },
    Proportional { sigma: f64 },
    Combined { sigma1: f64, sigma2: f64 },
    Exponential { sigma: f64 },
}

impl ErrorModel {
    pub fn variance(&self, y_pred: f64) -> f64 {
        match self {
            ErrorModel::Constant { sigma } => sigma * sigma,
            ErrorModel::Proportional { sigma } => (sigma * y_pred.abs()).powi(2),
            ErrorModel::Combined { sigma1, sigma2 } => (sigma1 + sigma2 * y_pred.abs()).powi(2),
            ErrorModel::Exponential { sigma } => (sigma * y_pred.abs()).powi(2),
        }
    }
}

/// Parameter transformation types
#[derive(Clone, Debug)]
pub enum Transform {
    Identity,
    Log,
    Probit,
    Logit,
}

impl Transform {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Transform::Identity => x,
            Transform::Log => x.exp(),
            Transform::Probit => {
                // Approximate normal CDF using tanh
                0.5 * (1.0 + (x * 0.7978).tanh())
            }
            Transform::Logit => 1.0 / (1.0 + (-x).exp()),
        }
    }

    pub fn inverse(&self, x: f64) -> f64 {
        match self {
            Transform::Identity => x,
            Transform::Log => x.max(1e-12).ln(),
            Transform::Probit => {
                let x_clipped = x.clamp(1e-12, 1.0 - 1e-12);
                (2.0 * x_clipped - 1.0).atanh() / 0.7978
            }
            Transform::Logit => {
                let x_clipped = x.clamp(1e-12, 1.0 - 1e-12);
                (x_clipped / (1.0 - x_clipped)).ln()
            }
        }
    }
}
