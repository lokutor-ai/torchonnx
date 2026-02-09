pub mod ir;
pub mod loader;
pub mod exporter;
pub mod optimizer;
pub mod verifier;

#[cfg(test)]
pub mod tests {
    use crate::ir::{ModelIR, Node, Tensor, DataType};
    use crate::exporter::onnx_exporter::OnnxExporter;
    use crate::verifier::{OnnxVerifier, ParityChecker};
    use crate::exporter::ModelExporter;
    use tempfile::tempdir;
    use std::collections::HashMap;

    pub fn check_node_parity(node: Node, weights: HashMap<String, Tensor>, inputs: HashMap<String, Tensor>, expected_outputs: HashMap<String, Tensor>, epsilon: f32) {
        let mut ir = ModelIR::new();
        ir.graph.nodes.push(node);
        ir.graph.weights = weights;
        ir.graph.expected_outputs = expected_outputs;
        
        for (name, tensor) in &inputs {
            ir.graph.inputs.push(Tensor {
                name: name.clone(),
                shape: tensor.shape.clone(),
                data_type: tensor.data_type.clone(),
                data: None,
            });
        }

        for (name, tensor) in &ir.graph.expected_outputs {
            ir.graph.outputs.push(Tensor {
                name: name.clone(),
                shape: tensor.shape.clone(),
                data_type: tensor.data_type.clone(),
                data: None,
            });
        }

        let dir = tempdir().unwrap();
        let onnx_path = dir.path().join("model.onnx");
        OnnxExporter::export(&ir, &onnx_path).unwrap();

        OnnxVerifier::check_parity(&ir, &onnx_path, inputs, epsilon).unwrap();
    }
}