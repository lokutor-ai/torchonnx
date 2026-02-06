use std::path::Path;
use crate::ir::ModelIR;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(String),
    #[error("Other error: {0}")]
    Other(String),
}

pub trait ModelLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<ModelIR, LoaderError>;
}

pub mod safetensors;
