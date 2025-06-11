//! Utility functions for NLME algorithms

use ndarray::{Array1, Array2};
use crate::errors::NLMEError;

/// Convert group labels to consecutive indices
pub fn group_to_indices(groups: &[i32]) -> (Vec<usize>, Vec<i32>) {
    let mut unique_groups = Vec::new();
    let mut indices = Vec::new();
    
    for &group in groups {
        if let Some(pos) = unique_groups.iter().position(|&x| x == group) {
            indices.push(pos);
        } else {
            unique_groups.push(group);
            indices.push(unique_groups.len() - 1);
        }
    }
    
    (indices, unique_groups)
}

/// Compute matrix inverse safely
pub fn safe_inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, NLMEError> {
    use nalgebra as na;
    
    let na_matrix = na::DMatrix::from_row_slice(
        matrix.nrows(),
        matrix.ncols(),
        matrix.as_slice().ok_or(NLMEError::InvalidMatrix)?,
    );
    
    let inverse = na_matrix
        .try_inverse()
        .ok_or(NLMEError::SingularMatrix)?;
    
    let mut result = Array2::zeros((matrix.nrows(), matrix.ncols()));
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            result[[i, j]] = inverse[(i, j)];
        }
    }
    
    Ok(result)
}

/// Compute Cholesky decomposition safely
pub fn safe_cholesky(matrix: &Array2<f64>) -> Result<Array2<f64>, NLMEError> {
    use nalgebra as na;
    
    let na_matrix = na::DMatrix::from_row_slice(
        matrix.nrows(),
        matrix.ncols(),
        matrix.as_slice().ok_or(NLMEError::InvalidMatrix)?,
    );
    
    let chol = na_matrix
        .cholesky()
        .ok_or(NLMEError::SingularMatrix)?;
    
    let chol_l = chol.l();
    let mut result = Array2::zeros((matrix.nrows(), matrix.ncols()));
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            result[[i, j]] = chol_l[(i, j)];
        }
    }
    
    Ok(result)
}

/// Check if matrix is positive definite
pub fn is_positive_definite(matrix: &Array2<f64>) -> bool {
    use nalgebra as na;
    
    if let Some(slice) = matrix.as_slice() {
        let na_matrix = na::DMatrix::from_row_slice(matrix.nrows(), matrix.ncols(), slice);
        
        // Check if Cholesky decomposition exists
        na_matrix.cholesky().is_some()
    } else {
        false
    }
}

/// Regularize matrix to ensure positive definiteness
pub fn regularize_matrix(matrix: &mut Array2<f64>, regularization: f64) {
    for i in 0..matrix.nrows() {
        matrix[[i, i]] += regularization;
    }
}

/// Compute log determinant safely
pub fn log_determinant(matrix: &Array2<f64>) -> Result<f64, NLMEError> {
    use nalgebra as na;
    
    let na_matrix = na::DMatrix::from_row_slice(
        matrix.nrows(),
        matrix.ncols(),
        matrix.as_slice().ok_or(NLMEError::InvalidMatrix)?,
    );
    
    let lu = na_matrix.lu();
    let det = lu.determinant();
    
    if det <= 0.0 {
        Err(NLMEError::SingularMatrix)
    } else {
        Ok(det.ln())
    }
}

/// Multivariate normal log probability density
pub fn mvn_logpdf(
    x: &Array1<f64>,
    mean: &Array1<f64>,
    cov: &Array2<f64>,
) -> Result<f64, NLMEError> {
    use nalgebra as na;
    
    let k = x.len() as f64;
    let diff = x - mean;
    
    // Convert to nalgebra
    let diff_vec = na::DVector::from_row_slice(diff.as_slice().ok_or(NLMEError::InvalidMatrix)?);
    let cov_matrix = na::DMatrix::from_row_slice(
        cov.nrows(),
        cov.ncols(),
        cov.as_slice().ok_or(NLMEError::InvalidMatrix)?,
    );
    
    let cov_inv = cov_matrix
        .try_inverse()
        .ok_or(NLMEError::SingularMatrix)?;
    
    let log_det = log_determinant(cov)?;
    let mahal = diff_vec.transpose() * cov_inv * diff_vec;
    
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let logpdf = -0.5 * (k * log_2pi + log_det + mahal[(0, 0)]);
    
    Ok(logpdf)
}

/// Sample from multivariate normal distribution
pub fn sample_mvn(
    mean: &Array1<f64>,
    cov: &Array2<f64>,
    rng: &mut impl rand::Rng,
) -> Result<Array1<f64>, NLMEError> {
    use rand_distr::{Normal, Distribution};
    
    let k = mean.len();
    let chol = safe_cholesky(cov)?;
    
    // Generate standard normal samples
    let normal = Normal::new(0.0, 1.0).map_err(|_| NLMEError::SamplingError)?;
    let z: Array1<f64> = (0..k)
        .map(|_| normal.sample(rng))
        .collect();
    
    // Transform to desired distribution: x = mean + L * z
    let mut result = mean.clone();
    for i in 0..k {
        for j in 0..=i {
            result[i] += chol[[i, j]] * z[j];
        }
    }
    
    Ok(result)
}

/// Compute effective sample size for MCMC chains
pub fn effective_sample_size(chain: &Array1<f64>) -> f64 {
    let n = chain.len();
    if n < 4 {
        return n as f64;
    }
    
    // Compute autocorrelation
    let mean = chain.mean().unwrap_or(0.0);
    let var = chain.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
    
    if var < 1e-12 {
        return n as f64;
    }
    
    let mut sum_autocorr = 1.0; // lag 0 autocorrelation is 1
    
    for lag in 1..n.min(100) {
        let mut autocorr = 0.0;
        let count = n - lag;
        
        for i in 0..count {
            autocorr += (chain[i] - mean) * (chain[i + lag] - mean);
        }
        autocorr /= count as f64 * var;
        
        if autocorr < 0.05 {
            break;
        }
        
        sum_autocorr += 2.0 * autocorr;
    }
    
    n as f64 / sum_autocorr.max(1.0)
}

/// Check convergence using Gelman-Rubin statistic
pub fn gelman_rubin_statistic(chains: &[Array1<f64>]) -> Result<f64, NLMEError> {
    let m = chains.len();
    if m < 2 {
        return Ok(1.0);
    }
    
    let n = chains[0].len();
    if n < 2 {
        return Ok(1.0);
    }
    
    // Check all chains have same length
    if !chains.iter().all(|chain| chain.len() == n) {
        return Err(NLMEError::InvalidChains);
    }
    
    // Chain means
    let chain_means: Vec<f64> = chains.iter()
        .map(|chain| chain.mean().unwrap_or(0.0))
        .collect();
    
    // Overall mean
    let overall_mean = chain_means.iter().sum::<f64>() / m as f64;
    
    // Between-chain variance
    let B = n as f64 * chain_means.iter()
        .map(|&mean| (mean - overall_mean).powi(2))
        .sum::<f64>() / (m - 1) as f64;
    
    // Within-chain variance
    let W = chains.iter()
        .map(|chain| {
            let mean = chain.mean().unwrap_or(0.0);
            chain.mapv(|x| (x - mean).powi(2)).sum() / (n - 1) as f64
        })
        .sum::<f64>() / m as f64;
    
    // Potential scale reduction factor
    let var_plus = ((n - 1) as f64 * W + B) / n as f64;
    let rhat = (var_plus / W).sqrt();
    
    Ok(rhat)
}
