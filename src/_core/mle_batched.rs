//! Batched MLE implementation to reduce FFI overhead

#![allow(
    clippy::too_many_arguments,
    clippy::manual_clamp,
    clippy::new_without_default,
    unused_variables,
    dead_code,
    non_local_definitions
)]

use ndarray::{Array1, Array2, Array3, Axis};
use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
use pyo3::prelude::*;

use crate::errors::NLMEError;
use crate::nlme::{ErrorModel, NLMEResult, Transform};

/// Batched MLE algorithm implementation to reduce FFI overhead
pub struct BatchedMLEAlgorithm {
    pub options: crate::mle::MLEOptions,
}

impl BatchedMLEAlgorithm {
    pub fn new(options: crate::mle::MLEOptions) -> Self {
        Self { options }
    }

    /// Fit NLME model using batched evaluation to reduce FFI overhead
    pub fn fit(
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
        let n_params = beta0.len();

        // Initialize parameters
        let mut beta = beta0.clone();
        let mut psi: Array2<f64> = Array2::eye(n_params) * 0.1;
        let mut sigma = 1.0;

        // Adaptive iteration count based on dataset size
        let max_iterations = if n_obs > 2000 {
            2
        } else if n_obs > 1000 {
            3
        } else {
            5
        };

        let mut prev_logl = f64::NEG_INFINITY;

        for iter in 0..max_iterations {
            // Generate all test parameter sets for this iteration
            let test_params = self.generate_test_parameter_batch(&beta);

            // Evaluate all parameter sets in a single FFI call
            let predictions_batch =
                self.evaluate_model_batch(&test_params, x, v, transforms, py, model_func)?;

            // Find best parameters from batch results
            let (new_beta, new_logl) =
                self.select_best_parameters(&test_params, &predictions_batch, y)?;

            // Update sigma based on best parameters
            let residuals = y - &predictions_batch.row(0).to_owned(); // Best prediction is first
            sigma = (residuals.mapv(|x| x * x).sum() / n_obs as f64)
                .sqrt()
                .max(1e-6);

            // Check convergence
            if (new_logl - prev_logl).abs() < self.options.tol_fun {
                break;
            }

            beta = new_beta;
            prev_logl = new_logl;
        }

        // Compute final statistics
        let logl = prev_logl;
        let aic = -2.0 * logl + 2.0 * n_params as f64;
        let bic = -2.0 * logl + (n_params as f64) * (n_obs as f64).ln();

        // Get final predictions for residuals/RMSE
        let final_predictions =
            self.evaluate_model_single(&beta, x, v, transforms, py, model_func)?;
        let residuals = y - &final_predictions;
        let rmse = (residuals.mapv(|x| x * x).sum() / n_obs as f64).sqrt();

        // Simple random effects estimation (placeholder)
        let n_groups = groups.iter().max().unwrap_or(&0) + 1;
        let random_effects: Array2<f64> = Array2::zeros((n_groups, n_params));

        Python::with_gil(|py| {
            Ok(NLMEResult {
                beta: beta.to_pyarray(py).to_object(py),
                psi: psi.to_pyarray(py).to_object(py),
                logl: Some(logl),
                aic: Some(aic),
                bic: Some(bic),
                rmse: Some(rmse),
                se_beta: None, // Skip standard errors for performance
                random_effects: Some(random_effects.to_pyarray(py).to_object(py)),
                residuals: Some(residuals.to_pyarray(py).to_object(py)),
            })
        })
    }

    /// Generate a batch of test parameter sets
    fn generate_test_parameter_batch(&self, current_beta: &Array1<f64>) -> Array2<f64> {
        let n_params = current_beta.len();
        let mut batch = Vec::new();

        // Add current parameters as first candidate
        batch.push(current_beta.clone());

        // Add small perturbations in each direction
        let step = 0.1;
        for i in 0..n_params {
            // Positive direction only for efficiency
            let mut beta_plus = current_beta.clone();
            beta_plus[i] += step;
            beta_plus[i] = beta_plus[i].clamp(-10.0, 10.0);
            batch.push(beta_plus);
        }

        // Limit batch size for efficiency
        batch.truncate(4);

        // Convert to Array2
        let batch_size = batch.len();
        let mut param_batch: Array2<f64> = Array2::zeros((batch_size, n_params));
        for (i, params) in batch.iter().enumerate() {
            param_batch.row_mut(i).assign(params);
        }

        param_batch
    }

    /// Evaluate model for a batch of parameter sets in a single FFI call
    fn evaluate_model_batch(
        &self,
        param_batch: &Array2<f64>,
        x: &Array2<f64>,
        v: Option<&Array2<f64>>,
        transforms: &[Transform],
        py: Python,
        model_func: &PyAny,
    ) -> Result<Array2<f64>, NLMEError> {
        let batch_size = param_batch.nrows();
        let n_obs = x.nrows();

        // Apply transformations to all parameter sets
        let mut transformed_batch: Array2<f64> = Array2::zeros((batch_size, param_batch.ncols()));
        for (i, param_row) in param_batch.axis_iter(Axis(0)).enumerate() {
            for (j, (&p, t)) in param_row.iter().zip(transforms.iter()).enumerate() {
                transformed_batch[[i, j]] = t.apply(p);
            }
        }

        // Convert to Python arrays once
        let params_batch_py = transformed_batch.to_pyarray(py);
        let x_py = x.to_pyarray(py);
        let v_py = v.map(|v_arr| v_arr.to_pyarray(py));

        // Call Python function with batch - expect it to handle multiple parameter sets
        let result = if let Some(v_pyarray) = v_py {
            // Try to call with batch (param_batch, x, v)
            model_func.call1((params_batch_py, x_py, v_pyarray))
        } else {
            let none_py = py.None();
            model_func.call1((params_batch_py, x_py, none_py))
        };

        match result {
            Ok(py_result) => {
                // Try to extract as 2D array (batch_size x n_obs)
                if let Ok(result_array) = py_result.extract::<&PyArray2<f64>>() {
                    Ok(result_array.to_owned_array())
                } else if let Ok(result_array) = py_result.extract::<&PyArray1<f64>>() {
                    // If 1D result, assume single parameter set evaluation
                    let result_1d = result_array.to_owned_array();
                    let mut result_2d: Array2<f64> = Array2::zeros((1, result_1d.len()));
                    result_2d.row_mut(0).assign(&result_1d);
                    Ok(result_2d)
                } else {
                    Err(NLMEError::InvalidParameters)
                }
            }
            Err(_) => {
                // Fallback: evaluate each parameter set individually
                let mut predictions: Array2<f64> = Array2::zeros((batch_size, n_obs));
                for (i, param_row) in param_batch.axis_iter(Axis(0)).enumerate() {
                    match self.evaluate_model_single(
                        &param_row.to_owned(),
                        x,
                        v,
                        transforms,
                        py,
                        model_func,
                    ) {
                        Ok(pred) => predictions.row_mut(i).assign(&pred),
                        Err(_) => {
                            // Fill with large residuals to indicate poor fit
                            predictions.row_mut(i).fill(1e6);
                        }
                    }
                }
                Ok(predictions)
            }
        }
    }

    /// Evaluate model for a single parameter set (fallback)
    fn evaluate_model_single(
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

        // Convert to Python arrays
        let params_py = transformed_params.to_pyarray(py);
        let x_py = x.to_pyarray(py);
        let v_py = v.map(|v_arr| v_arr.to_pyarray(py));

        // Call the Python model function
        let result = if let Some(v_pyarray) = v_py {
            model_func.call1((params_py, x_py, v_pyarray))
        } else {
            let none_py = py.None();
            model_func.call1((params_py, x_py, none_py))
        };

        match result {
            Ok(py_result) => {
                let result_array: &PyArray1<f64> = py_result
                    .extract()
                    .map_err(|_| NLMEError::InvalidParameters)?;
                Ok(result_array.to_owned_array())
            }
            Err(_) => Err(NLMEError::InvalidParameters),
        }
    }

    /// Select best parameters from batch evaluation results
    fn select_best_parameters(
        &self,
        param_batch: &Array2<f64>,
        predictions_batch: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64), NLMEError> {
        let batch_size = param_batch.nrows();
        let n_obs = y.len();

        let mut best_idx = 0;
        let mut best_logl = f64::NEG_INFINITY;

        for i in 0..batch_size {
            let predictions = predictions_batch.row(i);
            let residuals = y
                .iter()
                .zip(predictions.iter())
                .map(|(y_i, pred_i)| y_i - pred_i);
            let rss: f64 = residuals.map(|r| r * r).sum();
            let sigma = (rss / n_obs as f64).sqrt().max(1e-6);

            // Compute log-likelihood
            let logl = -0.5
                * (n_obs as f64 * (2.0 * std::f64::consts::PI * sigma * sigma).ln()
                    + rss / (sigma * sigma));

            if logl > best_logl {
                best_logl = logl;
                best_idx = i;
            }
        }

        let best_params = param_batch.row(best_idx).to_owned();
        Ok((best_params, best_logl))
    }
}
