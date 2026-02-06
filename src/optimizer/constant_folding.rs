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
            } else if all_constants && node.op_type == "Relu" {
                let a = &ir.weights[&node.inputs[0]];
                if a.data_type == DataType::F32 {
                    let a_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ref().unwrap().as_ptr() as *const f32,
                            a.data.as_ref().unwrap().len() / 4,
                        )
                    };

                    let mut res_data = Vec::with_capacity(a_data.len());
                    for &val in a_data {
                        res_data.push(if val > 0.0 { val } else { 0.0 });
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
            } else if all_constants && node.op_type == "Reshape" {
                let a = &ir.weights[&node.inputs[0]];
                let target_shape_tensor = &ir.weights[&node.inputs[1]];
                
                if let Some(target_shape_data) = &target_shape_tensor.data {
                    let mut output_shape = Vec::new();
                    for j in 0..target_shape_tensor.shape[0] {
                        let offset = j * 8;
                        let val = i64::from_le_bytes(target_shape_data[offset..offset+8].try_into().unwrap());
                        output_shape.push(val as usize);
                    }

                    let output_name = node.outputs[0].clone();
                    ir.weights.insert(output_name.clone(), Tensor {
                        name: output_name,
                        shape: output_shape,
                        data_type: a.data_type.clone(),
                        data: a.data.clone(),
                    });

                    ir.nodes.remove(i);
                    continue;
                }
            } else if all_constants && node.op_type == "BatchNormalization" {
                let x = &ir.weights[&node.inputs[0]];
                let scale = &ir.weights[&node.inputs[1]];
                let bias = &ir.weights[&node.inputs[2]];
                let mean = &ir.weights[&node.inputs[3]];
                let var = &ir.weights[&node.inputs[4]];

                if x.data_type == DataType::F32 {
                    let epsilon = match node.attributes.get("epsilon") {
                        Some(crate::ir::Attribute::Float(f)) => *f,
                        _ => 1e-5,
                    };

                    let x_data: &[f32] = unsafe { std::slice::from_raw_parts(x.data.as_ref().unwrap().as_ptr() as *const f32, x.data.as_ref().unwrap().len() / 4) };
                    let scale_data: &[f32] = unsafe { std::slice::from_raw_parts(scale.data.as_ref().unwrap().as_ptr() as *const f32, scale.data.as_ref().unwrap().len() / 4) };
                    let bias_data: &[f32] = unsafe { std::slice::from_raw_parts(bias.data.as_ref().unwrap().as_ptr() as *const f32, bias.data.as_ref().unwrap().len() / 4) };
                    let mean_data: &[f32] = unsafe { std::slice::from_raw_parts(mean.data.as_ref().unwrap().as_ptr() as *const f32, mean.data.as_ref().unwrap().len() / 4) };
                    let var_data: &[f32] = unsafe { std::slice::from_raw_parts(var.data.as_ref().unwrap().as_ptr() as *const f32, var.data.as_ref().unwrap().len() / 4) };

                    let c = x.shape[1];
                    let items_per_channel = x_data.len() / (x.shape[0] * c);
                    let mut res_data = vec![0.0f32; x_data.len()];

                    for n in 0..x.shape[0] {
                        for channel in 0..c {
                            let gamma = scale_data[channel];
                            let beta = bias_data[channel];
                            let mu = mean_data[channel];
                            let sigma2 = var_data[channel];
                            let factor = gamma / (sigma2 + epsilon).sqrt();

                            for j in 0..items_per_channel {
                                let idx = (n * c + channel) * items_per_channel + j;
                                res_data[idx] = (x_data[idx] - mu) * factor + beta;
                            }
                        }
                    }

                    let res_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(res_data.as_ptr() as *const u8, res_data.len() * 4).to_vec() };

                    let output_name = node.outputs[0].clone();
                    ir.weights.insert(output_name.clone(), Tensor {
                        name: output_name,
                        shape: x.shape.clone(),
                        data_type: DataType::F32,
                        data: Some(res_bytes),
                    });

                    ir.nodes.remove(i);
                    continue;
                }
            } else if all_constants && node.op_type == "Concat" {
                let axis = match node.attributes.get("axis") {
                    Some(crate::ir::Attribute::Int(ax)) => *ax as i64,
                    _ => 0,
                };

                let first_shape = &ir.weights[&node.inputs[0]].shape;
                let axis = if axis < 0 { (first_shape.len() as i64 + axis) as usize } else { axis as usize };

                let mut output_shape = first_shape.clone();
                output_shape[axis] = 0;
                for input_name in &node.inputs {
                    output_shape[axis] += ir.weights[input_name].shape[axis];
                }

                let mut outer_size = 1;
                for j in 0..axis { outer_size *= output_shape[j]; }
                let mut inner_size = 1;
                for j in axis + 1..output_shape.len() { inner_size *= output_shape[j]; }

                let mut res_data = vec![0u8; 0];
                let mut total_elements = 1;
                for &d in &output_shape { total_elements *= d; }
                res_data.resize(total_elements * 4, 0); // Assuming F32

                let mut current_axis_offset = 0;
                for input_name in &node.inputs {
                    let weight = &ir.weights[input_name];
                    let input_axis_size = weight.shape[axis];
                    let input_data = weight.data.as_ref().unwrap();

                    for outer in 0..outer_size {
                        let src_start = outer * input_axis_size * inner_size * 4;
                        let src_end = src_start + input_axis_size * inner_size * 4;
                        let dest_start = (outer * output_shape[axis] * inner_size + current_axis_offset * inner_size) * 4;
                        let dest_end = dest_start + input_axis_size * inner_size * 4;
                        res_data[dest_start..dest_end].copy_from_slice(&input_data[src_start..src_end]);
                    }
                    current_axis_offset += input_axis_size;
                }

                let output_name = node.outputs[0].clone();
                ir.weights.insert(output_name.clone(), Tensor {
                    name: output_name,
                    shape: output_shape,
                    data_type: DataType::F32,
                    data: Some(res_data),
                });

                ir.nodes.remove(i);
                continue;
            } else if all_constants && node.op_type == "GlobalAveragePool" {
                let a = &ir.weights[&node.inputs[0]];
                if a.data_type == DataType::F32 && a.shape.len() >= 2 {
                    let a_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ref().unwrap().as_ptr() as *const f32,
                            a.data.as_ref().unwrap().len() / 4,
                        )
                    };

                    let n = a.shape[0];
                    let c = a.shape[1];
                    let mut spatial_size = 1;
                    for j in 2..a.shape.len() { spatial_size *= a.shape[j]; }

                    let mut res_data = vec![0.0f32; n * c];
                    for i_n in 0..n {
                        for i_c in 0..c {
                            let mut sum = 0.0f32;
                            for i_s in 0..spatial_size {
                                sum += a_data[(i_n * c + i_c) * spatial_size + i_s];
                            }
                            res_data[i_n * c + i_c] = sum / spatial_size as f32;
                        }
                    }

                    let res_bytes: Vec<u8> = unsafe {
                        std::slice::from_raw_parts(
                            res_data.as_ptr() as *const u8,
                            res_data.len() * 4,
                        )
                    }.to_vec();

                    let mut output_shape = vec![n, c];
                    for _ in 2..a.shape.len() { output_shape.push(1); }

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

    #[test]
    fn test_constant_folding_relu() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![2],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 191, 0, 0, 128, 63]), // [-1.0, 1.0]
        });

        ir.nodes.push(Node {
            name: "relu".to_string(),
            op_type: "Relu".to_string(),
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
        assert_eq!(res_data[0], 0.0);
        assert_eq!(res_data[1], 1.0);
    }

    #[test]
    fn test_constant_folding_reshape() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1, 6],
            data_type: DataType::F32,
            data: Some(vec![0; 24]),
        });

        ir.weights.insert("shape".to_string(), Tensor {
            name: "shape".to_string(),
            shape: vec![2],
            data_type: DataType::I64,
            data: Some(vec![2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]),
        });

        ir.nodes.push(Node {
            name: "reshape".to_string(),
            op_type: "Reshape".to_string(),
            inputs: vec!["A".to_string(), "shape".to_string()],
            outputs: vec!["B".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("B"));
        assert_eq!(ir.weights["B"].shape, vec![2, 3]);
    }

    #[test]
    fn test_constant_folding_batch_norm() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("X".to_string(), Tensor {
            name: "X".to_string(),
            shape: vec![1, 1, 1, 1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]), // 1.0
        });
        ir.weights.insert("scale".to_string(), Tensor {
            name: "scale".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]), // 1.0
        });
        ir.weights.insert("bias".to_string(), Tensor {
            name: "bias".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 0]), // 0.0
        });
        ir.weights.insert("mean".to_string(), Tensor {
            name: "mean".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 0]), // 0.0
        });
        ir.weights.insert("var".to_string(), Tensor {
            name: "var".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]), // 1.0
        });

        ir.nodes.push(Node {
            name: "bn".to_string(),
            op_type: "BatchNormalization".to_string(),
            inputs: vec!["X".to_string(), "scale".to_string(), "bias".to_string(), "mean".to_string(), "var".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("Y"));
        let res_w = &ir.weights["Y"];
        let res_data: f32 = f32::from_le_bytes(res_w.data.as_ref().unwrap()[0..4].try_into().unwrap());
        assert!((res_data - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_constant_folding_concat() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1, 2],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63, 0, 0, 0, 64]), // [1.0, 2.0]
        });

        ir.weights.insert("B".to_string(), Tensor {
            name: "B".to_string(),
            shape: vec![1, 2],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 64, 64, 0, 0, 128, 64]), // [3.0, 4.0]
        });

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));

        ir.nodes.push(Node {
            name: "concat".to_string(),
            op_type: "Concat".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: attrs,
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("C"));
        assert_eq!(ir.weights["C"].shape, vec![1, 4]);
        let res_w = &ir.weights["C"];
        let res_data: Vec<f32> = unsafe {
            std::slice::from_raw_parts(
                res_w.data.as_ref().unwrap().as_ptr() as *const f32,
                4,
            ).to_vec()
        };
        assert_eq!(res_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_constant_folding_global_average_pool() {
        let mut ir = ModelIR::new();
        
        ir.weights.insert("A".to_string(), Tensor {
            name: "A".to_string(),
            shape: vec![1, 1, 2, 2],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]), // [1.0, 2.0, 3.0, 4.0]
        });

        ir.nodes.push(Node {
            name: "gap".to_string(),
            op_type: "GlobalAveragePool".to_string(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: HashMap::new(),
        });

        let folding = ConstantFolding;
        folding.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
        assert!(ir.weights.contains_key("B"));
        assert_eq!(ir.weights["B"].shape, vec![1, 1, 1, 1]);
        let res_w = &ir.weights["B"];
        let res_data: f32 = f32::from_le_bytes(res_w.data.as_ref().unwrap()[0..4].try_into().unwrap());
        assert!((res_data - 2.5).abs() < 1e-4);
    }
}
