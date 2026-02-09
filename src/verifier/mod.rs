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
        inputs: HashMap<String, crate::ir::Tensor>,
        epsilon: f32,
    ) -> Result<(), VerifierError>;
}

pub struct OnnxVerifier;

use std::collections::HashMap;

impl ParityChecker for OnnxVerifier {
    fn check_parity(
        ir: &ModelIR,
        onnx_path: &Path,
        inputs: HashMap<String, crate::ir::Tensor>,
        epsilon: f32,
    ) -> Result<(), VerifierError> {
        let mut session = Session::builder()
            .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?
            .commit_from_file(onnx_path)
            .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?;

        let mut ort_inputs = Vec::new();
        for (name, tensor) in &inputs {
            if let Some(ref data) = tensor.data {
                let val = match tensor.data_type {
                    DataType::F32 => {
                        let f32_data: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };
                        ort::value::Value::from_array((tensor.shape.clone(), f32_data.to_vec().into_boxed_slice())).unwrap()
                    }
                    _ => return Err(VerifierError::InferenceError("Unsupported verifier data type".to_string())),
                };
                ort_inputs.push((name.clone(), val));
            }
        }

        let outputs = session.run(ort_inputs.into_iter().collect::<Vec<_>>())
            .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?;

        for (name, expected) in &ir.graph.expected_outputs {
            let actual = outputs.get(name).ok_or_else(|| VerifierError::ParityError(format!("Output {} not found in ONNX result", name)))?;
            
            // Extract data from actual (Value)
            // In ort 2.0, try_extract_tensor returns Result<(&Shape, &[T])>
            let (_shape, actual_data) = actual.try_extract_tensor::<f32>()
                .map_err(|e| VerifierError::InferenceError(format!("{:?}", e)))?;
            
            let expected_data: &[f32] = unsafe { 
                std::slice::from_raw_parts(
                    expected.data.as_ref().unwrap().as_ptr() as *const f32, 
                    expected.data.as_ref().unwrap().len() / 4
                ) 
            };

            if actual_data.len() != expected_data.len() {
                return Err(VerifierError::ParityError(format!("Output {} length mismatch: got {}, expected {}", name, actual_data.len(), expected_data.len())));
            }

            for j in 0..actual_data.len() {
                if (actual_data[j] - expected_data[j]).abs() > epsilon {
                    return Err(VerifierError::ParityError(format!("Output {} numerical mismatch at index {}: got {}, expected {}", name, j, actual_data[j], expected_data[j])));
                }
            }
        }

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
            shape: vec![1, 2],
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
        ir.graph.outputs.push(Tensor {
            name: "Y".to_string(),
            shape: vec![1, 2],
            data_type: DataType::F32,
            data: None,
        });

        let input_data = vec![1.0f32, 2.0f32];
        let input_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(input_data.as_ptr() as *const u8, 8).to_vec() };
        
        let mut inputs = HashMap::new();
        inputs.insert("X".to_string(), Tensor {
            name: "X".to_string(),
            shape: vec![1, 2],
            data_type: DataType::F32,
            data: Some(input_bytes.clone()),
        });

        ir.graph.expected_outputs.insert("Y".to_string(), Tensor {
            name: "Y".to_string(),
            shape: vec![1, 2],
            data_type: DataType::F32,
            data: Some(input_bytes),
        });

        let dir = tempdir().unwrap();
        let onnx_path = dir.path().join("model.onnx");
        OnnxExporter::export(&ir, &onnx_path).unwrap();

        let result = OnnxVerifier::check_parity(&ir, &onnx_path, inputs, 1e-5);
        if let Err(ref e) = result {
            println!("Verifier Error: {:?}", e);
        }
        assert!(result.is_ok());
    }
}
