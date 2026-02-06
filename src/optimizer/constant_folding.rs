use crate::ir::{ModelIR, Tensor, DataType};
use crate::optimizer::{OptimizationPass, OptimizerError};

pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn apply(&self, ir: &mut ModelIR) -> Result<(), OptimizerError> {
        let mut i = 0;
        while i < ir.nodes.len() {
            let node = &ir.nodes[i];
            let mut all_constants = true;
            for input in &node.inputs {
                if !ir.weights.contains_key(input) {
                    all_constants = false;
                    break;
                }
            }

            if all_constants && node.op_type == "Add" {
                let a = &ir.weights[&node.inputs[0]];
                let b = &ir.weights[&node.inputs[1]];

                if a.data_type == DataType::F32 && b.data_type == DataType::F32 {
                    let a_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ref().unwrap().as_ptr() as *const f32,
                            a.data.as_ref().unwrap().len() / 4,
                        )
                    };
                    let b_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ref().unwrap().as_ptr() as *const f32,
                            b.data.as_ref().unwrap().len() / 4,
                        )
                    };

                    let mut res_data = Vec::with_capacity(a_data.len());
                    for j in 0..a_data.len() {
                        res_data.push(a_data[j] + b_data[j]);
                    }

                    let res_bytes: Vec<u8> = unsafe {
                        std::slice::from_raw_parts(
                            res_data.as_ptr() as *const u8,
                            res_data.len() * 4,
                        )
                    }.to_vec();

                    let output_name = node.outputs[0].clone();
                    ir.weights.insert(output_name.clone(), Tensor {
                        name: output_name,
                        shape: a.shape.clone(),
                        data_type: DataType::F32,
                        data: Some(res_bytes),
                    });

                    ir.nodes.remove(i);
                    continue;
                }
            }
            i += 1;
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
    fn test_constant_folding_basic_add() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]), // 1.0
        });
        
        ir.weights.insert("B".to_string(), Tensor {
            name: "B".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 64]), // 2.0
        });

        ir.nodes.push(Node {
            name: "add".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("C"));
    }
}
