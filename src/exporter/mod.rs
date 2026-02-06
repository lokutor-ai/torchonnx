pub mod onnx {
    include!("onnx.rs");
}

pub mod onnx_exporter;

use crate::ir::ModelIR;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExporterError {
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),
}

pub trait ModelExporter {
    fn export(ir: &ModelIR, path: &std::path::Path) -> Result<(), ExporterError>;
}
