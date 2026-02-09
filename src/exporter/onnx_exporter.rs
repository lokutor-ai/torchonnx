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
        model.ir_version = Some(7);
        model.producer_name = Some("torchonnx".to_string());

        model.graph = Some(Self::export_graph(&ir.graph));

        let mut opset = onnx::OperatorSetIdProto::default();
        opset.domain = Some("".to_string());
        opset.version = Some(15); // Use a modern opset version
        model.opset_import.push(opset);

        let mut buf = Vec::new();
        model.encode(&mut buf).map_err(|e| ExporterError::SerializationError(e.to_string()))?;

        let mut file = File::create(path).map_err(|e| ExporterError::SerializationError(e.to_string()))?;
        file.write_all(&buf).map_err(|e| ExporterError::SerializationError(e.to_string()))?;

        Ok(())
    }
}

impl OnnxExporter {
    fn export_graph(ir_graph: &crate::ir::Graph) -> onnx::GraphProto {
        let mut graph = onnx::GraphProto::default();
        graph.name = Some(ir_graph.name.clone());

        for (name, tensor) in &ir_graph.weights {
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

        for input in &ir_graph.inputs {
            let mut vi = onnx::ValueInfoProto::default();
            vi.name = Some(input.name.clone());
            let mut ty = onnx::TypeProto::default();
            let mut ten = onnx::type_proto::Tensor::default();
            ten.elem_type = Some(match input.data_type {
                crate::ir::DataType::F32 => onnx::tensor_proto::DataType::Float as i32,
                crate::ir::DataType::F64 => onnx::tensor_proto::DataType::Double as i32,
                crate::ir::DataType::I32 => onnx::tensor_proto::DataType::Int32 as i32,
                crate::ir::DataType::I64 => onnx::tensor_proto::DataType::Int64 as i32,
                crate::ir::DataType::U8 => onnx::tensor_proto::DataType::Uint8 as i32,
            });
            let mut sh = onnx::TensorShapeProto::default();
            for &d in &input.shape {
                let mut dim = onnx::tensor_shape_proto::Dimension::default();
                dim.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d as i64));
                sh.dim.push(dim);
            }
            ten.shape = Some(sh);
            ty.value = Some(onnx::type_proto::Value::TensorType(ten));
            vi.r#type = Some(ty);
            graph.input.push(vi);
        }

        for output in &ir_graph.outputs {
            let mut vi = onnx::ValueInfoProto::default();
            vi.name = Some(output.name.clone());
            let mut ty = onnx::TypeProto::default();
            let mut ten = onnx::type_proto::Tensor::default();
            ten.elem_type = Some(match output.data_type {
                crate::ir::DataType::F32 => onnx::tensor_proto::DataType::Float as i32,
                crate::ir::DataType::F64 => onnx::tensor_proto::DataType::Double as i32,
                crate::ir::DataType::I32 => onnx::tensor_proto::DataType::Int32 as i32,
                crate::ir::DataType::I64 => onnx::tensor_proto::DataType::Int64 as i32,
                crate::ir::DataType::U8 => onnx::tensor_proto::DataType::Uint8 as i32,
            });
            let mut sh = onnx::TensorShapeProto::default();
            for &d in &output.shape {
                let mut dim = onnx::tensor_shape_proto::Dimension::default();
                dim.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d as i64));
                sh.dim.push(dim);
            }
            ten.shape = Some(sh);
            ty.value = Some(onnx::type_proto::Value::TensorType(ten));
            vi.r#type = Some(ty);
            graph.output.push(vi);
        }

        for node in &ir_graph.nodes {
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
                    crate::ir::Attribute::Graph(g) => {
                        a.g = Some(Self::export_graph(g));
                        a.r#type = Some(onnx::attribute_proto::AttributeType::Graph as i32);
                    }
                    crate::ir::Attribute::Tensor(t) => {
                        let mut tp = onnx::TensorProto::default();
                        tp.name = Some(t.name.clone());
                        tp.dims = t.shape.iter().map(|&d| d as i64).collect();
                        tp.data_type = Some(match t.data_type {
                            crate::ir::DataType::F32 => onnx::tensor_proto::DataType::Float as i32,
                            crate::ir::DataType::F64 => onnx::tensor_proto::DataType::Double as i32,
                            crate::ir::DataType::I32 => onnx::tensor_proto::DataType::Int32 as i32,
                            crate::ir::DataType::I64 => onnx::tensor_proto::DataType::Int64 as i32,
                            crate::ir::DataType::U8 => onnx::tensor_proto::DataType::Uint8 as i32,
                        });
                        tp.raw_data = Some(t.data.clone().unwrap_or_default());
                        a.t = Some(tp);
                        a.r#type = Some(onnx::attribute_proto::AttributeType::Tensor as i32);
                    }
                }
                n.attribute.push(a);
            }
            graph.node.push(n);
        }

        graph
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
        ir.graph.weights.insert("w1".to_string(), Tensor {
            name: "w1".to_string(),
            shape: vec![1, 1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]),
        });
        ir.graph.nodes.push(Node {
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
        ir.graph.nodes.push(Node {
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
        ir.graph.nodes.push(Node {
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
        ir.graph.nodes.push(Node {
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
        ir.graph.nodes.push(Node {
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
        ir.graph.nodes.push(Node {
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
        ir.graph.nodes.push(Node {
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

    #[test]
    fn test_export_max_pool_model() {
        let mut ir = ModelIR::new();
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), crate::ir::Attribute::Ints(vec![2, 2]));
        attrs.insert("strides".to_string(), crate::ir::Attribute::Ints(vec![2, 2]));
        ir.graph.nodes.push(Node {
            name: "pool1".to_string(),
            op_type: "MaxPool".to_string(),
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
    fn test_export_softmax_model() {
        let mut ir = ModelIR::new();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));
        ir.graph.nodes.push(Node {
            name: "softmax1".to_string(),
            op_type: "Softmax".to_string(),
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
    fn test_export_average_pool_model() {
        let mut ir = ModelIR::new();
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), crate::ir::Attribute::Ints(vec![7, 7]));
        attrs.insert("strides".to_string(), crate::ir::Attribute::Ints(vec![1, 1]));
        ir.graph.nodes.push(Node {
            name: "pool1".to_string(),
            op_type: "AveragePool".to_string(),
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
    fn test_export_layer_norm_model() {
        let mut ir = ModelIR::new();
        ir.graph.nodes.push(Node {
            name: "ln1".to_string(),
            op_type: "LayerNormalization".to_string(),
            inputs: vec!["X".to_string(), "scale".to_string(), "bias".to_string()],
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
    fn test_export_concat_model() {
        let mut ir = ModelIR::new();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));
        ir.graph.nodes.push(Node {
            name: "concat1".to_string(),
            op_type: "Concat".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
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
    fn test_export_global_average_pool_model() {
        let mut ir = ModelIR::new();
        ir.graph.nodes.push(Node {
            name: "gap1".to_string(),
            op_type: "GlobalAveragePool".to_string(),
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
    fn test_export_flatten_model() {
        let mut ir = ModelIR::new();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));
        ir.graph.nodes.push(Node {
            name: "flatten1".to_string(),
            op_type: "Flatten".to_string(),
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
    fn test_export_gemm_model() {
        let mut ir = ModelIR::new();
        let mut attrs = HashMap::new();
        attrs.insert("transB".to_string(), crate::ir::Attribute::Int(1));
        ir.graph.nodes.push(Node {
            name: "gemm1".to_string(),
            op_type: "Gemm".to_string(),
            inputs: vec!["A".to_string(), "W".to_string(), "B".to_string()],
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
    fn test_export_identity_model() {
        let mut ir = ModelIR::new();
        ir.graph.nodes.push(Node {
            name: "id1".to_string(),
            op_type: "Identity".to_string(),
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
    fn test_export_gelu_model() {
        let mut ir = ModelIR::new();
        ir.graph.nodes.push(Node {
            name: "gelu1".to_string(),
            op_type: "Gelu".to_string(),
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
            fn test_export_slice_model() {
                let mut ir = ModelIR::new();
                
                ir.graph.nodes.push(Node {
                    name: "slice1".to_string(),
                    op_type: "Slice".to_string(),
                    inputs: vec!["X".to_string(), "starts".to_string(), "ends".to_string()],
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
                    fn test_export_squeeze_model() {
                        let mut ir = ModelIR::new();
                        
                        ir.graph.nodes.push(Node {
                            name: "squeeze1".to_string(),
                            op_type: "Squeeze".to_string(),
                            inputs: vec!["X".to_string(), "axes".to_string()],
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
                            fn test_export_unsqueeze_model() {
                                let mut ir = ModelIR::new();
                                
                                ir.graph.nodes.push(Node {
                                    name: "unsqueeze1".to_string(),
                                    op_type: "Unsqueeze".to_string(),
                                    inputs: vec!["X".to_string(), "axes".to_string()],
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
                                    fn test_export_gather_model() {
                                        let mut ir = ModelIR::new();
                                        
                                        ir.graph.nodes.push(Node {
                                            name: "gather1".to_string(),
                                            op_type: "Gather".to_string(),
                                            inputs: vec!["data".to_string(), "indices".to_string()],
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
                                            fn test_export_split_model() {
                                                let mut ir = ModelIR::new();
                                                
                                                let mut attrs = HashMap::new();
                                                attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));
                                        
                                                ir.graph.nodes.push(Node {
                                                    name: "split1".to_string(),
                                                    op_type: "Split".to_string(),
                                                    inputs: vec!["X".to_string()],
                                                    outputs: vec!["Y1".to_string(), "Y2".to_string()],
                                                    attributes: attrs,
                                                });
                                        
                                                let dir = tempdir().unwrap();
                                                let file_path = dir.path().join("model.onnx");
                                                
                                                        let result = OnnxExporter::export(&ir, &file_path);
                                                        assert!(result.is_ok());
                                                        assert!(file_path.exists());
                                                    }
                                                
                                                    #[test]
                                                    fn test_export_constant_model() {
                                                        let mut ir = ModelIR::new();
                                                        
                                                        let mut attrs = HashMap::new();
                                                        attrs.insert("value".to_string(), crate::ir::Attribute::Tensor(Tensor {
                                                            name: "val".to_string(),
                                                            shape: vec![2, 2],
                                                            data_type: DataType::F32,
                                                            data: Some(vec![0; 16]),
                                                        }));
                                                
                                                        ir.graph.nodes.push(Node {
                                                            name: "const1".to_string(),
                                                            op_type: "Constant".to_string(),
                                                            inputs: vec![],
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
                                                            fn test_export_resize_model() {
                                                                let mut ir = ModelIR::new();
                                                                
                                                                ir.graph.nodes.push(Node {
                                                                    name: "resize1".to_string(),
                                                                    op_type: "Resize".to_string(),
                                                                    inputs: vec!["X".to_string(), "".to_string(), "scales".to_string()],
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
                                                                    fn test_export_shape_op_model() {
                                                                        let mut ir = ModelIR::new();
                                                                        
                                                                        ir.graph.nodes.push(Node {
                                                                            name: "shape1".to_string(),
                                                                            op_type: "Shape".to_string(),
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
                                                                            fn test_export_cast_model() {
                                                                                let mut ir = ModelIR::new();
                                                                                
                                                                                let mut attrs = HashMap::new();
                                                                                attrs.insert("to".to_string(), crate::ir::Attribute::Int(7));
                                                                        
                                                                                ir.graph.nodes.push(Node {
                                                                                    name: "cast1".to_string(),
                                                                                    op_type: "Cast".to_string(),
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
                                                                                    fn test_export_expand_model() {
                                                                                        let mut ir = ModelIR::new();
                                                                                        
                                                                                        ir.graph.nodes.push(Node {
                                                                                            name: "expand1".to_string(),
                                                                                            op_type: "Expand".to_string(),
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
                                                                                            fn test_export_instance_norm_model() {
                                                                                                let mut ir = ModelIR::new();
                                                                                                
                                                                                                ir.graph.nodes.push(Node {
                                                                                                    name: "in1".to_string(),
                                                                                                    op_type: "InstanceNormalization".to_string(),
                                                                                                    inputs: vec!["X".to_string(), "scale".to_string(), "bias".to_string()],
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
                                                                                                    fn test_export_pad_model() {
                                                                                                        let mut ir = ModelIR::new();
                                                                                                        
                                                                                                        ir.graph.nodes.push(Node {
                                                                                                            name: "pad1".to_string(),
                                                                                                            op_type: "Pad".to_string(),
                                                                                                            inputs: vec!["X".to_string(), "pads".to_string()],
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
                                                                                                            fn test_export_erf_model() {
                                                                                                                let mut ir = ModelIR::new();
                                                                                                                
                                                                                                                ir.graph.nodes.push(Node {
                                                                                                                    name: "erf1".to_string(),
                                                                                                                    op_type: "Erf".to_string(),
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
                                                                                                                    fn test_export_constant_of_shape_model() {
                                                                                                                        let mut ir = ModelIR::new();
                                                                                                                        
                                                                                                                        ir.graph.nodes.push(Node {
                                                                                                                            name: "cos1".to_string(),
                                                                                                                            op_type: "ConstantOfShape".to_string(),
                                                                                                                            inputs: vec!["shape".to_string()],
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
                                                                                                                            fn test_export_sigmoid_model() {
                                                                                                                                let mut ir = ModelIR::new();
                                                                                                                                ir.graph.nodes.push(Node {
                                                                                                                                    name: "sig1".to_string(),
                                                                                                                                    op_type: "Sigmoid".to_string(),
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
                                                                                                                            fn test_export_tanh_model() {
                                                                                                                                let mut ir = ModelIR::new();
                                                                                                                                ir.graph.nodes.push(Node {
                                                                                                                                    name: "tanh1".to_string(),
                                                                                                                                    op_type: "Tanh".to_string(),
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
                                                                                                                        }
                                                                                                                        