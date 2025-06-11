//! Error types for PyNLME

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NLMEError {
    #[error("Singular matrix encountered")]
    SingularMatrix,
    
    #[error("Invalid matrix data")]
    InvalidMatrix,
    
    #[error("Invalid parameters")]
    InvalidParameters,
    
    #[error("Model evaluation failed")]
    ModelEvaluation,
    
    #[error("Convergence failed after maximum iterations")]
    ConvergenceFailed,
    
    #[error("Sampling error")]
    SamplingError,
    
    #[error("Invalid MCMC chains")]
    InvalidChains,
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
}
