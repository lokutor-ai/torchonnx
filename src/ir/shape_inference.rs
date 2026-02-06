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

        }

        