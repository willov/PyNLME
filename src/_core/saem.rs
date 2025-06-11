//! SAEM (Stochastic Approximation Expectation-Maximization) algorithm implementation

#![allow(
    clippy::too_many_arguments,
    unused_variables,
    dead_code,
    non_local_definitions
)]

use nalgebra as na;
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::errors::NLMEError;
use crate::nlme::{ErrorModel, NLMEResult, Transform};

/// SAEM algorithm options
#[pyclass]
#[derive(Clone)]
pub struct SAEMOptions {
    /// Number of iterations for each phase (warmup, main, final)
    #[pyo3(get, set)]
    pub n_iterations: (usize, usize, usize),

    /// Number of MCMC iterations for each phase
    #[pyo3(get, set)]
    pub n_mcmc_iterations: (usize, usize, usize),

    /// Number of burn-in iterations
    #[pyo3(get, set)]
    pub n_burn_in: usize,

    /// Number of MCMC chains
    #[pyo3(get, set)]
    pub n_chains: Option<usize>,

    /// Convergence tolerance
    #[pyo3(get, set)]
    pub tol_x: f64,

    /// Random seed
    #[pyo3(get, set)]
    pub random_seed: Option<u64>,

    /// Verbosity level
    #[pyo3(get, set)]
    pub verbose: usize,
}

#[pymethods]
impl SAEMOptions {
    #[new]
    pub fn new() -> Self {
        Self {
            n_iterations: (150, 150, 100),
            n_mcmc_iterations: (2, 2, 2),
            n_burn_in: 5,
            n_chains: None,
            tol_x: 1e-4,
            random_seed: None,
            verbose: 0,
        }
    }
}

impl Default for SAEMOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// SAEM algorithm implementation
pub struct SAEMFitter {
    options: SAEMOptions,
    rng: StdRng,
}

impl SAEMFitter {
    pub fn new(options: SAEMOptions) -> Self {
        let rng = match options.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self { options, rng }
    }

    /// Fit NLME model using SAEM algorithm
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        v: Option<&Array2<f64>>,
        beta0: &Array1<f64>,
        mut error_model: ErrorModel,
        transforms: &[Transform],
    ) -> Result<NLMEResult, NLMEError> {
        let n_obs = y.len();
        let n_groups = groups.iter().max().unwrap_or(&0) + 1;
        let n_params = beta0.len();

        // Initialize parameters
        let mut beta = beta0.clone();
        let mut psi = Array2::eye(n_params) * 0.1;
        let mut gamma = Array2::eye(n_params) * 0.1; // Working covariance matrix

        // Initialize random effects for each group
        let mut random_effects = Array2::zeros((n_params, n_groups));

        // Initialize sigma from error model or default
        let mut sigma = match &error_model {
            ErrorModel::Constant { sigma } => *sigma,
            _ => 0.2, // Default value
        };

        // SAEM iterations
        let phases = [
            (
                self.options.n_iterations.0,
                self.options.n_mcmc_iterations.0,
            ),
            (
                self.options.n_iterations.1,
                self.options.n_mcmc_iterations.1,
            ),
            (
                self.options.n_iterations.2,
                self.options.n_mcmc_iterations.2,
            ),
        ];

        for (phase_idx, (n_iter, n_mcmc)) in phases.iter().enumerate() {
            let step_size_decay = if phase_idx == 0 {
                1.0
            } else {
                0.5_f64.powi(phase_idx as i32)
            };

            for iter in 0..*n_iter {
                // E-step: Sample random effects using MCMC
                self.sample_random_effects(
                    &mut random_effects,
                    x,
                    y,
                    groups,
                    v,
                    &beta,
                    &gamma,
                    &error_model,
                    transforms,
                    *n_mcmc,
                )?;

                // SA-step: Update sufficient statistics
                let step_size = step_size_decay / (iter + 1) as f64;

                // M-step: Update parameters
                self.update_parameters(
                    &mut beta,
                    &mut psi,
                    &mut gamma,
                    &mut sigma,
                    &random_effects,
                    x,
                    y,
                    groups,
                    v,
                    &error_model,
                    transforms,
                    step_size,
                )?;

                // Update error model with new sigma
                error_model = ErrorModel::Constant { sigma };

                if self.options.verbose > 1 {
                    println!(
                        "Phase {}, Iteration {}: beta = {:?}",
                        phase_idx + 1,
                        iter + 1,
                        beta
                    );
                }
            }
        }

        // Compute final statistics
        let logl =
            self.compute_log_likelihood(x, y, groups, v, &beta, &psi, &error_model, transforms)?;
        let n_total_params = n_params + psi.len(); // Fixed effects + covariance parameters
        let aic = -2.0 * logl + 2.0 * n_total_params as f64;
        let bic = -2.0 * logl + (n_groups as f64).ln() * n_total_params as f64;

        // Compute RMSE
        let y_pred = self.predict_population(x, v, &beta, transforms)?;
        let residuals = y - &y_pred;
        let rmse = (residuals.mapv(|x| x * x).sum() / n_obs as f64).sqrt();

        Python::with_gil(|py| {
            Ok(NLMEResult {
                beta: beta.to_pyarray(py).to_object(py),
                psi: psi.to_pyarray(py).to_object(py),
                logl: Some(logl),
                aic: Some(aic),
                bic: Some(bic),
                rmse: Some(rmse),
                se_beta: None, // TODO: Compute standard errors
                random_effects: Some(random_effects.to_pyarray(py).to_object(py)),
                residuals: Some(residuals.to_pyarray(py).to_object(py)),
            })
        })
    }

    fn sample_random_effects(
        &mut self,
        random_effects: &mut Array2<f64>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        _v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        gamma: &Array2<f64>,
        error_model: &ErrorModel,
        transforms: &[Transform],
        n_mcmc: usize,
    ) -> Result<(), NLMEError> {
        // Simplified MCMC sampling for random effects
        // In practice, this would use more sophisticated methods like HMC or NUTS

        let n_groups = random_effects.ncols();
        let n_params = random_effects.nrows();

        for group in 0..n_groups {
            let group_mask: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| g == group)
                .map(|(i, _)| i)
                .collect();

            if group_mask.is_empty() {
                continue;
            }

            // Extract data for this group
            let x_group = x.select(Axis(0), &group_mask);
            let y_group = y.select(Axis(0), &group_mask);

            // Current random effects for this group
            let mut current_re = random_effects.column(group).to_owned();

            // MCMC sampling - now properly handle all dimensions
            for _ in 0..n_mcmc {
                // Propose new random effects for ALL parameters
                let proposal_std = 0.1; // Could be adaptive
                let mut proposal = current_re.clone();

                for p in 0..n_params {
                    proposal[p] += self.rng.sample::<f64, _>(StandardNormal) * proposal_std;
                }

                // Compute acceptance probability
                let current_logp = self.log_posterior_re(
                    &current_re,
                    &x_group,
                    &y_group,
                    beta,
                    gamma,
                    error_model,
                    transforms,
                )?;
                let proposal_logp = self.log_posterior_re(
                    &proposal,
                    &x_group,
                    &y_group,
                    beta,
                    gamma,
                    error_model,
                    transforms,
                )?;

                let alpha = (proposal_logp - current_logp).exp().min(1.0);

                if self.rng.gen::<f64>() < alpha {
                    current_re = proposal;
                }
            }

            // Update random effects
            for (i, &val) in current_re.iter().enumerate() {
                random_effects[[i, group]] = val;
            }
        }

        Ok(())
    }

    fn log_posterior_re(
        &self,
        re: &Array1<f64>,
        x_group: &Array2<f64>,
        y_group: &Array1<f64>,
        beta: &Array1<f64>,
        gamma: &Array2<f64>,
        error_model: &ErrorModel,
        transforms: &[Transform],
    ) -> Result<f64, NLMEError> {
        // Compute log posterior density for random effects
        // log p(b_i | y_i, beta, Gamma) ∝ log p(y_i | b_i, beta) + log p(b_i | Gamma)

        // Log-likelihood contribution
        let phi = beta + re; // Combined parameters
        let y_pred = self.evaluate_model(&phi, x_group, None, transforms)?;

        let mut log_lik = 0.0;
        for (&obs, &pred) in y_group.iter().zip(y_pred.iter()) {
            let var = error_model.variance(pred);
            log_lik +=
                -0.5 * ((obs - pred).powi(2) / var + var.ln() + (2.0 * std::f64::consts::PI).ln());
        }

        // Prior contribution (multivariate normal)
        let gamma_inv =
            na::DMatrix::from_row_slice(gamma.nrows(), gamma.ncols(), gamma.as_slice().unwrap())
                .try_inverse()
                .ok_or(NLMEError::SingularMatrix)?;

        let re_vec = na::DVector::from_row_slice(re.as_slice().unwrap());
        let log_prior = -0.5 * re_vec.transpose() * gamma_inv * re_vec;

        Ok(log_lik + log_prior[(0, 0)])
    }

    fn update_parameters(
        &mut self,
        beta: &mut Array1<f64>,
        psi: &mut Array2<f64>,
        gamma: &mut Array2<f64>,
        sigma: &mut f64,
        random_effects: &Array2<f64>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        groups: &Array1<usize>,
        _v: Option<&Array2<f64>>,
        _error_model: &ErrorModel,
        transforms: &[Transform],
        step_size: f64,
    ) -> Result<(), NLMEError> {
        // IMPROVED SAEM parameter updates
        let n_groups = random_effects.ncols();

        // Update beta: Use weighted least squares approach
        // For each group, we have y_ij = f(x_ij, beta + b_i) + error
        // We use a simplified approach: estimate beta using all data with current random effects
        let mut total_sum_x = Array2::<f64>::zeros((beta.len(), beta.len()));
        let mut total_sum_xy = Array1::<f64>::zeros(beta.len());

        for group in 0..n_groups {
            let group_mask: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| g == group)
                .map(|(i, _)| i)
                .collect();

            if group_mask.is_empty() {
                continue;
            }

            let x_group = x.select(Axis(0), &group_mask);
            let y_group = y.select(Axis(0), &group_mask);
            let re_group = random_effects.column(group).to_owned();

            // Linearize around current estimates
            let current_params = beta.clone() + &re_group;

            // Compute approximate gradients numerically
            let eps = 1e-6;
            let mut jacobian = Array2::<f64>::zeros((y_group.len(), beta.len()));

            for p in 0..beta.len() {
                let mut params_plus = current_params.clone();
                params_plus[p] += eps;
                let y_plus = self.evaluate_model(&params_plus, &x_group, None, transforms)?;

                let mut params_minus = current_params.clone();
                params_minus[p] -= eps;
                let y_minus = self.evaluate_model(&params_minus, &x_group, None, transforms)?;

                for i in 0..y_group.len() {
                    jacobian[[i, p]] = (y_plus[i] - y_minus[i]) / (2.0 * eps);
                }
            }

            // Add to normal equations (J^T J and J^T (y - f))
            let y_pred = self.evaluate_model(&current_params, &x_group, None, transforms)?;
            let residuals = &y_group - &y_pred;

            for i in 0..beta.len() {
                for j in 0..beta.len() {
                    for k in 0..y_group.len() {
                        total_sum_x[[i, j]] += jacobian[[k, i]] * jacobian[[k, j]];
                    }
                }

                for k in 0..y_group.len() {
                    total_sum_xy[i] += jacobian[[k, i]] * residuals[k];
                }
            }
        }

        // Solve normal equations with regularization
        for i in 0..beta.len() {
            total_sum_x[[i, i]] += 1e-6; // Regularization
        }

        // Convert to nalgebra for inversion
        let jtj =
            na::DMatrix::from_row_slice(beta.len(), beta.len(), total_sum_x.as_slice().unwrap());
        let jtr = na::DVector::from_row_slice(total_sum_xy.as_slice().unwrap());

        if let Some(jtj_inv) = jtj.try_inverse() {
            let delta_beta = jtj_inv * jtr;

            // Update beta with step size
            for i in 0..beta.len() {
                beta[i] += step_size * delta_beta[i];
            }
        }

        // Update random effects covariance (Psi)
        let mut sum_outer = Array2::<f64>::zeros((psi.nrows(), psi.ncols()));
        for group in 0..n_groups {
            let re_group = random_effects.column(group).to_owned();

            for i in 0..psi.nrows() {
                for j in 0..psi.ncols() {
                    sum_outer[[i, j]] += re_group[i] * re_group[j];
                }
            }
        }

        let new_psi = sum_outer / n_groups as f64;

        // Use stochastic approximation step size
        for i in 0..psi.nrows() {
            for j in 0..psi.ncols() {
                psi[[i, j]] = (1.0 - step_size) * psi[[i, j]] + step_size * new_psi[[i, j]];
            }
        }

        *gamma = psi.clone();

        // Update sigma (residual error)
        let mut sum_sq_residuals = 0.0;
        let mut total_obs = 0;

        for group in 0..n_groups {
            let group_mask: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| g == group)
                .map(|(i, _)| i)
                .collect();

            if group_mask.is_empty() {
                continue;
            }

            let x_group = x.select(Axis(0), &group_mask);
            let y_group = y.select(Axis(0), &group_mask);
            let re_group = random_effects.column(group).to_owned();

            let current_params = beta.clone() + &re_group;
            let y_pred = self.evaluate_model(&current_params, &x_group, None, transforms)?;

            for (&obs, &pred) in y_group.iter().zip(y_pred.iter()) {
                sum_sq_residuals += (obs - pred).powi(2);
                total_obs += 1;
            }
        }

        let new_sigma = (sum_sq_residuals / total_obs as f64).sqrt();
        *sigma = (1.0 - step_size) * *sigma + step_size * new_sigma;

        // Ensure sigma is reasonable
        *sigma = sigma.max(1e-6);

        // Add regularization to ensure positive definiteness
        for i in 0..psi.nrows() {
            psi[[i, i]] += 1e-6;
            gamma[[i, i]] += 1e-6;
        }

        Ok(())
    }

    fn compute_log_likelihood(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        _groups: &Array1<usize>,
        _v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        _psi: &Array2<f64>,
        error_model: &ErrorModel,
        transforms: &[Transform],
    ) -> Result<f64, NLMEError> {
        // Simplified log-likelihood computation
        let y_pred = self.predict_population(x, _v, beta, transforms)?;
        let residuals = y - &y_pred;

        let mut log_lik = 0.0;
        for (&res, &pred) in residuals.iter().zip(y_pred.iter()) {
            let var = error_model.variance(pred);
            log_lik += -0.5 * (res * res / var + var.ln() + (2.0 * std::f64::consts::PI).ln());
        }

        Ok(log_lik)
    }

    fn predict_population(
        &self,
        x: &Array2<f64>,
        _v: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        transforms: &[Transform],
    ) -> Result<Array1<f64>, NLMEError> {
        self.evaluate_model(beta, x, _v, transforms)
    }

    fn evaluate_model(
        &self,
        params: &Array1<f64>,
        x: &Array2<f64>,
        v: Option<&Array2<f64>>,
        transforms: &[Transform],
    ) -> Result<Array1<f64>, NLMEError> {
        // Apply parameter transformations
        let transformed_params: Array1<f64> = params
            .iter()
            .zip(transforms.iter())
            .map(|(&p, t)| t.apply(p))
            .collect();

        // Simplified model evaluation - exponential decay
        // In practice, this would call user-defined model functions
        let mut y_pred = Array1::zeros(x.nrows());

        if transformed_params.len() >= 2 {
            for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
                // Example: y = phi[0] * exp(-phi[1] * x[0])
                let t = x_row[0];
                y_pred[i] = transformed_params[0] * (-transformed_params[1] * t).exp();
            }
        } else {
            return Err(NLMEError::InvalidParameters);
        }

        Ok(y_pred)
    }
}

/// Python interface for SAEM fitting
#[pyfunction]
#[pyo3(signature = (x, y, groups, beta0, options, v=None))]
pub fn fit_nlme_saem(
    _py: Python,
    x: &PyArray2<f64>,
    y: &PyArray1<f64>,
    groups: &PyArray1<usize>,
    beta0: &PyArray1<f64>,
    options: SAEMOptions,
    v: Option<&PyArray2<f64>>,
) -> PyResult<NLMEResult> {
    let x_array = x.to_owned_array();
    let y_array = y.to_owned_array();
    let groups_array = groups.to_owned_array();
    let v_array = v.map(|v| v.to_owned_array());
    let beta0_array = beta0.to_owned_array();

    let mut fitter = SAEMFitter::new(options);
    let error_model = ErrorModel::Constant { sigma: 1.0 }; // Default
    let transforms = vec![Transform::Identity; beta0_array.len()]; // Default

    match fitter.fit(
        &x_array,
        &y_array,
        &groups_array,
        v_array.as_ref(),
        &beta0_array,
        error_model,
        &transforms,
    ) {
        Ok(result) => Ok(result),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "SAEM fitting failed: {:?}",
            e
        ))),
    }
}
