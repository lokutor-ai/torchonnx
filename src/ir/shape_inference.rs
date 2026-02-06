use crate::ir::{ModelIR, Tensor, DataType};
use crate::optimizer::OptimizerError;
use std::collections::HashMap;

pub struct ShapeInference;

impl ShapeInference {
    pub fn infer(ir: &mut ModelIR) -> Result<(), OptimizerError> {
        let mut value_shapes = HashMap::new();

        for input in &ir.inputs {
            value_shapes.insert(input.name.clone(), input.shape.clone());
        }

        for (name, weight) in &ir.weights {
            value_shapes.insert(name.clone(), weight.shape.clone());
        }

        let mut inferred_tensors = Vec::new();

        for node in &ir.nodes {
            match node.op_type.as_str() {
                "Add" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?
                        .clone();
                    
                    value_shapes.insert(node.outputs[0].clone(), shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Relu" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?
                        .clone();

                    value_shapes.insert(node.outputs[0].clone(), shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Einsum" => {
                    let equation = match node.attributes.get("equation") {
                        Some(crate::ir::Attribute::String(s)) => s,
                        _ => return Err(OptimizerError::Error("Einsum missing equation attribute".to_string())),
                    };

                    let parts: Vec<&str> = equation.split("->").collect();
                    let output_labels = if parts.len() > 1 { parts[1].trim() } else { "" };
                    
                    let mut label_to_size = HashMap::new();
                    let input_labels_parts: Vec<&str> = parts[0].split(',').collect();
                    
                    for (i, labels) in input_labels_parts.iter().enumerate() {
                        let shape = value_shapes.get(&node.inputs[i])
                            .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[i])))?;
                        
                        let clean_labels = labels.trim();
                        for (j, label) in clean_labels.chars().enumerate() {
                            label_to_size.insert(label, shape[j]);
                        }
                    }

                    let mut output_shape = Vec::new();
                    for label in output_labels.chars() {
                        let size = label_to_size.get(&label)
                            .ok_or_else(|| OptimizerError::Error(format!("Label {} not found in inputs", label)))?;
                        output_shape.push(*size);
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "MatMul" => {
                    let shape_a = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    let shape_b = value_shapes.get(&node.inputs[1])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[1])))?;

                    if shape_a.len() < 2 || shape_b.len() < 2 {
                        return Err(OptimizerError::Error("MatMul requires at least 2D inputs".to_string()));
                    }

                    let mut output_shape = Vec::new();
                    for i in 0..shape_a.len() - 1 {
                        output_shape.push(shape_a[i]);
                    }
                    output_shape.push(shape_b[shape_b.len() - 1]);

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Transpose" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    let perm = match node.attributes.get("perm") {
                        Some(crate::ir::Attribute::Ints(p)) => p.clone(),
                        _ => {
                            let mut p: Vec<i64> = (0..shape.len() as i64).collect();
                            p.reverse();
                            p
                        }
                    };

                    let mut output_shape = Vec::with_capacity(shape.len());
                    for &p in &perm {
                        output_shape.push(shape[p as usize]);
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                _ => {}
            }
        }

        for tensor in inferred_tensors {
            if !ir.outputs.iter().any(|t| t.name == tensor.name) {
                ir.outputs.push(tensor);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Node, Tensor, DataType};
    use std::collections::HashMap;

    #[test]
    fn test_infer_add_shape() {
        let mut ir = ModelIR::new();
        
        ir.inputs.push(Tensor {
            name: "A".to_string(),
            shape: vec![1, 3, 224, 224],
            data_type: DataType::F32,
            data: None,
        });
        
        ir.inputs.push(Tensor {
            name: "B".to_string(),
            shape: vec![1, 3, 224, 224],
            data_type: DataType::F32,
            data: None,
        });

        ir.nodes.push(Node {
            name: "add".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        });

        ShapeInference::infer(&mut ir).unwrap();

                assert_eq!(ir.outputs.len(), 1);

                assert_eq!(ir.outputs[0].shape, vec![1, 3, 224, 224]);

            }

        

            #[test]

            fn test_infer_relu_shape() {

                let mut ir = ModelIR::new();

                

                ir.inputs.push(Tensor {

                    name: "X".to_string(),

                    shape: vec![1, 10],

                    data_type: DataType::F32,

                    data: None,

                });

        

                ir.nodes.push(Node {

                    name: "relu1".to_string(),

                    op_type: "Relu".to_string(),

                    inputs: vec!["X".to_string()],

                    outputs: vec!["Y".to_string()],

                    attributes: HashMap::new(),

                });

        

                ShapeInference::infer(&mut ir).unwrap();

        

                        let y_shape = ir.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

        

                        assert_eq!(y_shape, Some(&vec![1, 10]));

        

                    }

        

                

        

                    #[test]

        

                    fn test_infer_einsum_dot_product() {

        

                        let mut ir = ModelIR::new();

        

                        

        

                        ir.inputs.push(Tensor {

        

                            name: "A".to_string(),

        

                            shape: vec![10],

        

                            data_type: DataType::F32,

        

                            data: None,

        

                        });

        

                

        

                        ir.inputs.push(Tensor {

        

                            name: "B".to_string(),

        

                            shape: vec![10],

        

                            data_type: DataType::F32,

        

                            data: None,

        

                        });

        

                

        

                        let mut attrs = HashMap::new();

        

                        attrs.insert("equation".to_string(), crate::ir::Attribute::String("i,i->".to_string()));

        

                

        

                        ir.nodes.push(Node {

        

                            name: "einsum1".to_string(),

        

                            op_type: "Einsum".to_string(),

        

                            inputs: vec!["A".to_string(), "B".to_string()],

        

                            outputs: vec!["Y".to_string()],

        

                            attributes: attrs,

        

                        });

        

                

        

                        ShapeInference::infer(&mut ir).unwrap();

        

                

        

                                let y_shape = ir.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

        

                

        

                                assert_eq!(y_shape, Some(&vec![]));

        

                

        

                            }

        

                

        

                        

        

                

        

                            #[test]

        

                

        

                            fn test_infer_matmul_shape() {

        

                

        

                                let mut ir = ModelIR::new();

        

                

        

                                

        

                

        

                                ir.inputs.push(Tensor {

        

                

        

                                    name: "A".to_string(),

        

                

        

                                    shape: vec![5, 10],

        

                

        

                                    data_type: DataType::F32,

        

                

        

                                    data: None,

        

                

        

                                });

        

                

        

                        

        

                

        

                                ir.inputs.push(Tensor {

        

                

        

                                    name: "B".to_string(),

        

                

        

                                    shape: vec![10, 3],

        

                

        

                                    data_type: DataType::F32,

        

                

        

                                    data: None,

        

                

        

                                });

        

                

        

                        

        

                

        

                                ir.nodes.push(Node {

        

                

        

                                    name: "matmul1".to_string(),

        

                

        

                                    op_type: "MatMul".to_string(),

        

                

        

                                    inputs: vec!["A".to_string(), "B".to_string()],

        

                

        

                                    outputs: vec!["Y".to_string()],

        

                

        

                                    attributes: HashMap::new(),

        

                

        

                                });

        

                

        

                        

        

                

        

                                ShapeInference::infer(&mut ir).unwrap();

        

                

        

                        

        

                

        

                                        let y_shape = ir.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

        

                

        

                        

        

                

        

                                        assert_eq!(y_shape, Some(&vec![5, 3]));

        

                

        

                        

        

                

        

                                    }

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        

                                    #[test]

        

                

        

                        

        

                

        

                                    fn test_infer_transpose_shape() {

        

                

        

                        

        

                

        

                                        let mut ir = ModelIR::new();

        

                

        

                        

        

                

        

                                        

        

                

        

                        

        

                

        

                                        ir.inputs.push(Tensor {

        

                

        

                        

        

                

        

                                            name: "X".to_string(),

        

                

        

                        

        

                

        

                                            shape: vec![1, 2, 3],

        

                

        

                        

        

                

        

                                            data_type: DataType::F32,

        

                

        

                        

        

                

        

                                            data: None,

        

                

        

                        

        

                

        

                                        });

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        

                                        let mut attrs = HashMap::new();

        

                

        

                        

        

                

        

                                        attrs.insert("perm".to_string(), crate::ir::Attribute::Ints(vec![0, 2, 1]));

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        

                                        ir.nodes.push(Node {

        

                

        

                        

        

                

        

                                            name: "transpose1".to_string(),

        

                

        

                        

        

                

        

                                            op_type: "Transpose".to_string(),

        

                

        

                        

        

                

        

                                            inputs: vec!["X".to_string()],

        

                

        

                        

        

                

        

                                            outputs: vec!["Y".to_string()],

        

                

        

                        

        

                

        

                                            attributes: attrs,

        

                

        

                        

        

                

        

                                        });

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        

                                        ShapeInference::infer(&mut ir).unwrap();

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        

                                        let y_shape = ir.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

        

                

        

                        

        

                

        

                                        assert_eq!(y_shape, Some(&vec![1, 3, 2]));

        

                

        

                        

        

                

        

                                    }

        

                

        

                        

        

                

        

                                }

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        