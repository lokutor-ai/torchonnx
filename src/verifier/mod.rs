use crate::ir::ModelIR;
use thiserror::Error;
use std::path::Path;

#[derive(Error, Debug)]
pub enum VerifierError {
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Parity error: {0}")]
    ParityError(String),
}

pub trait ParityChecker {
    fn check_parity(
        ir: &ModelIR,
        onnx_path: &Path,
        epsilon: f32,
    ) -> Result<(), VerifierError>;
}
