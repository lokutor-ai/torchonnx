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

            if all_constants && (node.op_type == "Add" || node.op_type == "Sub" || node.op_type == "Mul" || node.op_type == "Div") {
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
                        if node.op_type == "Add" {
                            res_data.push(a_data[j] + b_data[j]);
                        } else if node.op_type == "Sub" {
                            res_data.push(a_data[j] - b_data[j]);
                        } else if node.op_type == "Mul" {
                            res_data.push(a_data[j] * b_data[j]);
                        } else {
                            res_data.push(a_data[j] / b_data[j]);
                        }
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
            } else if all_constants && node.op_type == "Transpose" {
                let a = &ir.weights[&node.inputs[0]];
                if a.data_type == DataType::F32 {
                    let perm = match node.attributes.get("perm") {
                        Some(crate::ir::Attribute::Ints(p)) => p.clone(),
                        _ => {
                            let mut p: Vec<i64> = (0..a.shape.len() as i64).collect();
                            p.reverse();
                            p
                        }
                    };

                    let mut output_shape = Vec::with_capacity(a.shape.len());
                    for &p in &perm {
                        output_shape.push(a.shape[p as usize]);
                    }

                    let a_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ref().unwrap().as_ptr() as *const f32,
                            a.data.as_ref().unwrap().len() / 4,
                        )
                    };

                    let mut res_data = vec![0.0f32; a_data.len()];
                    let mut strides_a = vec![1; a.shape.len()];
                    let mut strides_res = vec![1; output_shape.len()];
                    for j in (0..a.shape.len() - 1).rev() {
                        strides_a[j] = strides_a[j + 1] * a.shape[j + 1];
                        strides_res[j] = strides_res[j + 1] * output_shape[j + 1];
                    }

                    for j in 0..a_data.len() {
                        let mut remaining = j;
                        let mut coords_a = vec![0; a.shape.len()];
                        for k in 0..a.shape.len() {
                            coords_a[k] = remaining / strides_a[k];
                            remaining %= strides_a[k];
                        }

                        let mut res_idx = 0;
                        for (k, &p) in perm.iter().enumerate() {
                            res_idx += coords_a[p as usize] * strides_res[k];
                        }
                        res_data[res_idx] = a_data[j];
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
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: Some(res_bytes),
                    });

                    ir.nodes.remove(i);
                    continue;
                }
            } else if all_constants && node.op_type == "Softmax" {
                let a = &ir.weights[&node.inputs[0]];
                if a.data_type == DataType::F32 {
                    let axis = match node.attributes.get("axis") {
                        Some(crate::ir::Attribute::Int(ax)) => *ax as i64,
                        _ => -1,
                    };
                    let axis = if axis < 0 { (a.shape.len() as i64 + axis) as usize } else { axis as usize };

                    let a_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ref().unwrap().as_ptr() as *const f32,
                            a.data.as_ref().unwrap().len() / 4,
                        )
                    };

                    let mut res_data = vec![0.0f32; a_data.len()];
                    let mut outer_size = 1;
                    for j in 0..axis { outer_size *= a.shape[j]; }
                    let axis_size = a.shape[axis];
                    let mut inner_size = 1;
                    for j in axis + 1..a.shape.len() { inner_size *= a.shape[j]; }

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut max_val = f32::NEG_INFINITY;
                            for k in 0..axis_size {
                                let idx = outer * axis_size * inner_size + k * inner_size + inner;
                                if a_data[idx] > max_val { max_val = a_data[idx]; }
                            }

                            let mut sum_exp = 0.0f32;
                            for k in 0..axis_size {
                                let idx = outer * axis_size * inner_size + k * inner_size + inner;
                                res_data[idx] = (a_data[idx] - max_val).exp();
                                sum_exp += res_data[idx];
                            }

                            for k in 0..axis_size {
                                let idx = outer * axis_size * inner_size + k * inner_size + inner;
                                res_data[idx] /= sum_exp;
                            }
                        }
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
            data: Some(vec![0, 0, 128, 63]),
        });
        
        ir.weights.insert("B".to_string(), Tensor {
            name: "B".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 64]),
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

    #[test]
    fn test_constant_folding_basic_sub() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 64]), // 2.0
        });
        
        ir.weights.insert("B".to_string(), Tensor {
            name: "B".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]), // 1.0
        });

        ir.nodes.push(Node {
            name: "sub".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("C"));
    }

    #[test]
    fn test_constant_folding_basic_mul() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 64]), // 2.0
        });
        
        ir.weights.insert("B".to_string(), Tensor {
            name: "B".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 64, 64]), // 3.0
        });

        ir.nodes.push(Node {
            name: "mul".to_string(),
            op_type: "Mul".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("C"));
        let res_w = &ir.weights["C"];
        let res_data: f32 = f32::from_le_bytes(res_w.data.as_ref().unwrap()[0..4].try_into().unwrap());
        assert!((res_data - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_constant_folding_basic_div() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 64, 64]), // 3.0
        });
        
        ir.weights.insert("B".to_string(), Tensor {
            name: "B".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 64]), // 2.0
        });

        ir.nodes.push(Node {
            name: "div".to_string(),
            op_type: "Div".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("C"));
        let res_w = &ir.weights["C"];
        let res_data: f32 = f32::from_le_bytes(res_w.data.as_ref().unwrap()[0..4].try_into().unwrap());
        assert!((res_data - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_constant_folding_transpose() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![2, 3],
            data_type: DataType::F32,
            data: Some(vec![0; 24]),
        });

        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), crate::ir::Attribute::Ints(vec![1, 0]));

        ir.nodes.push(Node {
            name: "transpose".to_string(),
            op_type: "Transpose".to_string(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: attrs,
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("B"));
        assert_eq!(ir.weights["B"].shape, vec![3, 2]);
    }

    #[test]
    fn test_constant_folding_softmax() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1, 2],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 0, 0, 0, 0, 0]), // [0.0, 0.0]
        });

        ir.nodes.push(Node {
            name: "softmax".to_string(),
            op_type: "Softmax".to_string(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("B"));
        let res_w = &ir.weights["B"];
        let res_data: [f32; 2] = [
            f32::from_le_bytes(res_w.data.as_ref().unwrap()[0..4].try_into().unwrap()),
            f32::from_le_bytes(res_w.data.as_ref().unwrap()[4..8].try_into().unwrap()),
        ];
        assert!((res_data[0] - 0.5).abs() < 1e-4);
        assert!((res_data[1] - 0.5).abs() < 1e-4);
    }
}
