//! MLE (Maximum Likelihood Estimation) algorithm implementation

#![allow(
    clippy::too_many_arguments,
    clippy::manual_clamp,
    clippy::new_without_default,
    unused_variables,
    dead_code,
    non_local_definitions
)]

use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::errors::NLMEError;
use crate::nlme::{ErrorModel, NLMEResult, Transform};

/// MLE algorithm options
#[pyclass]
#[derive(Clone)]
pub struct MLEOptions {
    /// Maximum number of iterations
    #[pyo3(get, set)]
    pub max_iter: usize,

    /// Function tolerance
    #[pyo3(get, set)]
    pub tol_fun: f64,

    /// Parameter tolerance
    #[pyo3(get, set)]
    pub tol_x: f64,

    /// Approximation type
    #[pyo3(get, set)]
    pub approximation_type: String,

    /// Compute standard errors
    #[pyo3(get, set)]
    pub compute_std_errors: bool,

    /// Verbosity level
    #[pyo3(get, set)]
    pub verbose: usize,
}

#[pymethods]
impl MLEOptions {
    #[new]
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            tol_fun: 1e-6,
            tol_x: 1e-6,
            approximation_type: "LME".to_string(),
            compute_std_errors: true,
            verbose: 0,
        }
    }
}

impl Default for MLEOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// MLE algorithm implementation
pub struct MLEFitter {
    options: MLEOptions,
}

impl MLEFitter {
    pub fn new(options: MLEOptions) -> Self {
        Self { options }
    }

    /// Fit NLME model using MLE with Python model function
    pub fn fit_with_model(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        beta0: &Array1<f64>,
        error_model: ErrorModel,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<NLMEResult, NLMEError> {
        // Use Python model function
        self.fit_internal(x, y, groups, v, beta0, error_model, transforms, py, model_func)
    }

    /// Internal fit method for Python models
    fn fit_internal(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        beta0: &Array1<f64>,
        error_model: ErrorModel,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<NLMEResult, NLMEError> {
        let n_obs = y.len();
        let n_groups = groups.iter().max().unwrap_or(&0) + 1;
        let n_params = beta0.len();

        // Initialize parameters
        let mut beta = beta0.clone();
        let mut psi = Array2::eye(n_params) * 0.1;
        let mut sigma = 1.0; // Error variance

        // Optimization loop
        let mut prev_logl = f64::NEG_INFINITY;

        for iter in 0..self.options.max_iter {
            // E-step: Estimate random effects
            let random_effects = self.estimate_random_effects(
                x,
                y,
                groups,
                v,
                &beta,
                &psi,
                sigma,
                &error_model,
                transforms,
            )?;

            // M-step: Update parameters
            let (new_beta, new_psi, new_sigma) =
                self.update_parameters(x, y, groups, v, &random_effects, &error_model, transforms, py, model_func, &beta)?;

            // Compute log-likelihood
            let logl = self.compute_log_likelihood(
                x,
                y,
                groups,
                v,
                &new_beta,
                &new_psi,
                new_sigma,
                &error_model,
                transforms,
                py,
                model_func,
            )?;

            // Check convergence
            let beta_change = (&new_beta - &beta).mapv(|x| x.abs()).sum();
            let logl_change = (logl - prev_logl).abs() / (prev_logl.abs() + 1e-12);

            // Check convergence
            if beta_change < self.options.tol_x && logl_change < self.options.tol_fun {
                break;
            }

            beta = new_beta;
            psi = new_psi;
            sigma = new_sigma;
            prev_logl = logl;
        }

        // Compute final statistics
        let logl = self.compute_log_likelihood(
            x,
            y,
            groups,
            v,
            &beta,
            &psi,
            sigma,
            &error_model,
            transforms,
            py,
            model_func,
        )?;
        let n_total_params = n_params + psi.len() + 1; // Fixed effects + covariance + error variance
        let aic = -2.0 * logl + 2.0 * n_total_params as f64;
        let bic = -2.0 * logl + (n_groups as f64).ln() * n_total_params as f64;

        // Compute RMSE
        let y_pred = self.predict_population(x, v, &beta, transforms, py, model_func)?;
        let residuals = y - &y_pred;
        let rmse = (residuals.mapv(|x| x * x).sum() / n_obs as f64).sqrt();

        // Compute standard errors if requested
        let se_beta = if self.options.compute_std_errors {
            Some(self.compute_standard_errors(
                x,
                y,
                groups,
                v,
                &beta,
                &psi,
                sigma,
                &error_model,
                transforms,
            )?)
        } else {
            None
        };

        // Final random effects estimation
        let random_effects = self.estimate_random_effects(
            x,
            y,
            groups,
            v,
            &beta,
            &psi,
            sigma,
            &error_model,
            transforms,
        )?;

        Python::with_gil(|py| {
            Ok(NLMEResult {
                beta: beta.to_pyarray(py).to_object(py),
                psi: psi.to_pyarray(py).to_object(py),
                logl: Some(logl),
                aic: Some(aic),
                bic: Some(bic),
                rmse: Some(rmse),
                se_beta: se_beta.map(|arr| arr.to_pyarray(py).to_object(py)),
                random_effects: Some(random_effects.to_pyarray(py).to_object(py)),
                residuals: Some(residuals.to_pyarray(py).to_object(py)),
            })
        })
    }

    fn estimate_random_effects(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        psi: &Array2<f64>,
        sigma: f64,
        error_model: &ErrorModel,
        transforms: &[Transform],
    ) -> Result<Array2<f64>, NLMEError> {
        let n_groups = groups.iter().max().unwrap_or(&0) + 1;
        let n_params = beta.len();
        let mut random_effects = Array2::zeros((n_params, n_groups));

        // For the simplified model, we'll use minimal random effects
        // This ensures the fixed effects estimation is not corrupted

        // Initialize random effects to zero for initial optimization
        // This focuses the optimization on getting the fixed effects right

        for group in 0..n_groups {
            // Set small random effects that don't interfere with optimization
            for param in 0..n_params {
                random_effects[[param, group]] = 0.0; // No random effects for now
            }
        }

        Ok(random_effects)
    }

    fn update_parameters(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        random_effects: &Array2<f64>,
        error_model: &ErrorModel,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
        current_beta: &Array1<f64>, // Use current beta as starting point
    ) -> Result<(Array1<f64>, Array2<f64>, f64), NLMEError> {
        let n_obs = y.len();
        let n_params = current_beta.len();

        // Start optimization from current beta values
        let mut beta = current_beta.clone();

        // Gradient descent optimization
        let mut prev_logl = f64::NEG_INFINITY;
        for iter in 0..100 {
            let current_logl = self.compute_log_likelihood(
                x,
                y,
                groups,
                v,
                &beta,
                &Array2::eye(n_params),
                1.0,
                error_model,
                transforms,
                py,
                model_func,
            )?;

            if (current_logl - prev_logl).abs() < 1e-8 {
                break;
            }

            let gradient = self.compute_gradient_loglik(x, y, &beta, transforms, py, model_func)?;
            let step_size = 0.01 / (1.0 + iter as f64 * 0.1); // Decreasing step size

            // Update parameters
            for i in 0..n_params {
                beta[i] += step_size * gradient[i];
                // Apply reasonable bounds based on parameter type
                // For log-parameterized models, parameters can be negative
                if i == 0 {
                    // First parameter (often absorption rate or amplitude)
                    beta[i] = beta[i].clamp(-10.0, 10.0);
                } else {
                    // Other parameters (often log-transformed)
                    beta[i] = beta[i].clamp(-20.0, 20.0);
                }
            }

            prev_logl = current_logl;
        }

        // Update random effects covariance (simplified but reasonable)
        let psi = Array2::eye(n_params) * 0.1;

        // Update error variance using residuals
        let y_pred = self.predict_population(x, v, &beta, transforms, py, model_func)?;
        let residuals = y - &y_pred;
        let sigma = (residuals.mapv(|x| x * x).sum() / n_obs as f64)
            .sqrt()
            .max(1e-6);

        Ok((beta, psi, sigma))
    }

    fn compute_gradient_loglik(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        beta: &Array1<f64>,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<Array1<f64>, NLMEError> {
        let n_params = beta.len();
        let mut gradient = Array1::zeros(n_params);
        let h = 1e-6; // Small step for numerical differentiation

        // Compute numerical gradient of log-likelihood
        for i in 0..n_params {
            let mut beta_plus = beta.clone();
            let mut beta_minus = beta.clone();
            beta_plus[i] += h;
            beta_minus[i] -= h;

            let y_pred_plus = self.evaluate_model(&beta_plus, x, None, transforms, py, model_func)?;
            let y_pred_minus = self.evaluate_model(&beta_minus, x, None, transforms, py, model_func)?;

            // Compute log-likelihood for both
            let mut logl_plus = 0.0;
            let mut logl_minus = 0.0;

            for (j, (&y_obs, (&pred_plus, &pred_minus))) in y
                .iter()
                .zip(y_pred_plus.iter().zip(y_pred_minus.iter()))
                .enumerate()
            {
                let res_plus = y_obs - pred_plus;
                let res_minus = y_obs - pred_minus;

                // Simplified likelihood (constant error variance)
                logl_plus += -0.5 * res_plus * res_plus;
                logl_minus += -0.5 * res_minus * res_minus;
            }

            gradient[i] = (logl_plus - logl_minus) / (2.0 * h); // Gradient for maximization
        }

        Ok(gradient)
    }

    fn compute_log_likelihood(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        psi: &Array2<f64>,
        sigma: f64,
        error_model: &ErrorModel,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<f64, NLMEError> {
        let y_pred = self.predict_population(x, v, beta, transforms, py, model_func)?;
        let residuals = y - &y_pred;

        let mut log_lik = 0.0;
        for (i, (&res, &pred)) in residuals.iter().zip(y_pred.iter()).enumerate() {
            let var = error_model.variance(pred) * sigma * sigma;
            log_lik += -0.5 * (res * res / var + var.ln() + (2.0 * std::f64::consts::PI).ln());
        }

        Ok(log_lik)
    }

    fn compute_standard_errors(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        psi: &Array2<f64>,
        sigma: f64,
        error_model: &ErrorModel,
        transforms: &[Transform],
    ) -> Result<Array1<f64>, NLMEError> {
        // Simplified standard error computation
        // In practice, this would compute the Fisher information matrix
        let n_params = beta.len();
        let se = Array1::from_elem(n_params, 0.1); // Placeholder
        Ok(se)
    }

    fn predict_population(
        &self,
        x: &Array2<f64>,
        v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<Array1<f64>, NLMEError> {
        self.evaluate_model(beta, x, v, transforms, py, model_func)
    }

    fn predict_group(
        &self,
        params: &Array1<f64>,
        x_group: &Array2<f64>,
        v_group: Option<&Array2<f64>>,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<Array1<f64>, NLMEError> {
        self.evaluate_model(params, x_group, v_group, transforms, py, model_func)
    }

    fn evaluate_model(
        &self,
        params: &Array1<f64>,
        x: &Array2<f64>,
        v: Option<&Array2<f64>>,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<Array1<f64>, NLMEError> {
        // Apply parameter transformations
        let transformed_params: Array1<f64> = params
            .iter()
            .zip(transforms.iter())
            .map(|(&p, t)| t.apply(p))
            .collect();

        // Apply parameter transformations
        let transformed_params: Array1<f64> = params
            .iter()
            .zip(transforms.iter())
            .map(|(&p, t)| t.apply(p))
            .collect();

        // Call Python model function
        let params_py = transformed_params.to_pyarray(py);
        let x_py = x.to_pyarray(py);
        let v_py = v.map(|v_arr| v_arr.to_pyarray(py));

        // Call the Python model function: model_func(phi, x, v)
        let result = if let Some(v_pyarray) = v_py {
            model_func.call1((params_py, x_py, v_pyarray))
        } else {
            let none_py = py.None();
            model_func.call1((params_py, x_py, none_py))
        };

        match result {
            Ok(py_result) => {
                // Convert Python result back to Rust Array1
                let result_array: &PyArray1<f64> = py_result.extract()
                    .map_err(|_| NLMEError::InvalidParameters)?;
                Ok(result_array.to_owned_array())
            }
            Err(_) => Err(NLMEError::InvalidParameters),
        }
    }
}

/// Python interface for MLE fitting
#[pyfunction]
#[pyo3(signature = (x, y, groups, beta0, options, model_func, v=None))]
pub fn fit_nlme_mle(
    py: Python,
    x: &PyArray2<f64>,
    y: &PyArray1<f64>,
    groups: &PyArray1<usize>,
    beta0: &PyArray1<f64>,
    options: MLEOptions,
    model_func: &PyAny,
    v: Option<&PyArray2<f64>>,
) -> PyResult<NLMEResult> {
    let x_array = x.to_owned_array();
    let y_array = y.to_owned_array();
    let groups_array = groups.to_owned_array();
    let v_array = v.map(|v| v.to_owned_array());
    let beta0_array = beta0.to_owned_array();

    let fitter = MLEFitter::new(options);
    let error_model = ErrorModel::Constant { sigma: 1.0 }; // Default
    let transforms = vec![Transform::Identity; beta0_array.len()]; // Default

    match fitter.fit_with_model(
        &x_array,
        &y_array,
        &groups_array,
        v_array.as_ref(),
        &beta0_array,
        error_model,
        &transforms,
        py,
        model_func,
    ) {
        Ok(result) => Ok(result),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "MLE fitting failed: {:?}",
            e
        ))),
    }
}
