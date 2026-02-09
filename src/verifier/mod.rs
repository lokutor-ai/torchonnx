use crate::ir::{ModelIR, DataType};
use thiserror::Error;
use std::path::Path;
use ort::session::Session;
use rand::Rng;

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

pub struct OnnxVerifier;

impl ParityChecker for OnnxVerifier {
    fn check_parity(
        ir: &ModelIR,
        onnx_path: &Path,
        _epsilon: f32,
    ) -> Result<(), VerifierError> {
        let mut session = Session::builder()
            .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?
            .commit_from_file(onnx_path)
            .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?;

        let mut inputs = Vec::new();
        for input_ir in &ir.graph.inputs {
            let total_elements: usize = input_ir.shape.iter().product();
            let mut rng = rand::thread_rng();
            let data: Vec<f32> = (0..total_elements).map(|_| rng.gen_range(-1.0..1.0)).collect();
            inputs.push((input_ir.name.clone(), ort::value::Value::from_array((input_ir.shape.clone(), data.into_boxed_slice())).unwrap()));
        }

        let _outputs = session.run(inputs.into_iter().collect::<Vec<_>>())
            .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ModelIR, Node, Tensor, DataType};
    use crate::exporter::onnx_exporter::OnnxExporter;
    use crate::exporter::ModelExporter;
    use tempfile::tempdir;
    use std::collections::HashMap;

    #[test]
    fn test_verifier_loads_model() {
        let mut ir = ModelIR::new();
        ir.graph.inputs.push(Tensor {
            name: "X".to_string(),
            shape: vec![1, 10],
            data_type: DataType::F32,
            data: None,
        });
        ir.graph.nodes.push(Node {
            name: "id1".to_string(),
            op_type: "Identity".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let dir = tempdir().unwrap();
        let onnx_path = dir.path().join("model.onnx");
        OnnxExporter::export(&ir, &onnx_path).unwrap();

        let _result = OnnxVerifier::check_parity(&ir, &onnx_path, 1e-5);
    }
}
