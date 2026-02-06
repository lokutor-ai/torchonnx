use std::path::Path;
use std::fs::File;
use std::io::Write;
use crate::ir::ModelIR;
use crate::exporter::{ModelExporter, ExporterError, onnx};
use prost::Message;

pub struct OnnxExporter;

impl ModelExporter for OnnxExporter {
    fn export(ir: &ModelIR, path: &Path) -> Result<(), ExporterError> {
        let mut model = onnx::ModelProto::default();
        model.ir_version = Some(onnx::Version::IrVersion as i64);
        model.producer_name = Some("torchonnx".to_string());

        let mut graph = onnx::GraphProto::default();
        graph.name = Some("torchonnx_graph".to_string());

        for (name, tensor) in &ir.weights {
            let mut tp = onnx::TensorProto::default();
            tp.name = Some(name.clone());
            tp.dims = tensor.shape.iter().map(|&d| d as i64).collect();
            tp.data_type = Some(match tensor.data_type {
                crate::ir::DataType::F32 => onnx::tensor_proto::DataType::Float as i32,
                crate::ir::DataType::F64 => onnx::tensor_proto::DataType::Double as i32,
                crate::ir::DataType::I32 => onnx::tensor_proto::DataType::Int32 as i32,
                crate::ir::DataType::I64 => onnx::tensor_proto::DataType::Int64 as i32,
                crate::ir::DataType::U8 => onnx::tensor_proto::DataType::Uint8 as i32,
            });
            tp.raw_data = Some(tensor.data.clone().unwrap_or_default());
            graph.initializer.push(tp);
        }

        for node in &ir.nodes {
            let mut n = onnx::NodeProto::default();
            n.name = Some(node.name.clone());
            n.op_type = Some(node.op_type.clone());
            n.input = node.inputs.clone();
            n.output = node.outputs.clone();
            
            for (attr_name, attr_val) in &node.attributes {
                let mut a = onnx::AttributeProto::default();
                a.name = Some(attr_name.clone());
                match attr_val {
                    crate::ir::Attribute::Float(f) => {
                        a.f = Some(*f);
                        a.r#type = Some(onnx::attribute_proto::AttributeType::Float as i32);
                    }
                    crate::ir::Attribute::Int(i) => {
                        a.i = Some(*i);
                        a.r#type = Some(onnx::attribute_proto::AttributeType::Int as i32);
                    }
                    crate::ir::Attribute::String(s) => {
                        a.s = Some(s.as_bytes().to_vec());
                        a.r#type = Some(onnx::attribute_proto::AttributeType::String as i32);
                    }
                    crate::ir::Attribute::Floats(fs) => {
                        a.floats = fs.clone();
                        a.r#type = Some(onnx::attribute_proto::AttributeType::Floats as i32);
                    }
                    crate::ir::Attribute::Ints(is) => {
                        a.ints = is.clone();
                        a.r#type = Some(onnx::attribute_proto::AttributeType::Ints as i32);
                    }
                }
                n.attribute.push(a);
            }
            graph.node.push(n);
        }

        model.graph = Some(graph);

        let mut buf = Vec::new();
        model.encode(&mut buf).map_err(|e| ExporterError::SerializationError(e.to_string()))?;

        let mut file = File::create(path).map_err(|e| ExporterError::SerializationError(e.to_string()))?;
        file.write_all(&buf).map_err(|e| ExporterError::SerializationError(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ModelIR, Node, Tensor, DataType};
    use tempfile::tempdir;
    use std::collections::HashMap;

    #[test]
    fn test_export_basic_model() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("w1".to_string(), Tensor {
            name: "w1".to_string(),
            shape: vec![1, 1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]),
        });

        ir.nodes.push(Node {
            name: "add1".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["X".to_string(), "w1".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_export_relu_model() {
        let mut ir = ModelIR::new();
        
        ir.nodes.push(Node {
            name: "relu1".to_string(),
            op_type: "Relu".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_export_matmul_model() {
        let mut ir = ModelIR::new();
        
        ir.nodes.push(Node {
            name: "matmul1".to_string(),
            op_type: "MatMul".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_export_transpose_model() {
        let mut ir = ModelIR::new();
        
        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), crate::ir::Attribute::Ints(vec![0, 2, 1]));

        ir.nodes.push(Node {
            name: "transpose1".to_string(),
            op_type: "Transpose".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: attrs,
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_export_reshape_model() {
        let mut ir = ModelIR::new();
        
        ir.nodes.push(Node {
            name: "reshape1".to_string(),
            op_type: "Reshape".to_string(),
            inputs: vec!["X".to_string(), "shape".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_export_conv_model() {
        let mut ir = ModelIR::new();
        
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), crate::ir::Attribute::Ints(vec![1, 1]));
        attrs.insert("pads".to_string(), crate::ir::Attribute::Ints(vec![0, 0, 0, 0]));

        ir.nodes.push(Node {
            name: "conv1".to_string(),
            op_type: "Conv".to_string(),
            inputs: vec!["X".to_string(), "W".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: attrs,
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_export_batch_norm_model() {
        let mut ir = ModelIR::new();
        
        ir.nodes.push(Node {
            name: "bn1".to_string(),
            op_type: "BatchNormalization".to_string(),
            inputs: vec!["X".to_string(), "scale".to_string(), "B".to_string(), "mean".to_string(), "var".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        
        let result = OnnxExporter::export(&ir, &file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }
}
