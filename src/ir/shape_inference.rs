use crate::ir::{ModelIR, Tensor, DataType};
use crate::optimizer::OptimizerError;
use std::collections::HashMap;

pub struct ShapeInference;

impl ShapeInference {
    pub fn infer(ir: &mut ModelIR) -> Result<(), OptimizerError> {
        let mut value_shapes = HashMap::new();

        for input in &ir.graph.inputs {
            value_shapes.insert(input.name.clone(), input.shape.clone());
        }

        for (name, weight) in &ir.graph.weights {
            value_shapes.insert(name.clone(), weight.shape.clone());
        }

        let mut inferred_tensors = Vec::new();

        for node in &ir.graph.nodes {
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
                "Reshape" => {
                    let target_shape_tensor = ir.graph.weights.get(&node.inputs[1])
                        .ok_or_else(|| OptimizerError::Error(format!("Reshape target shape {} must be a constant for now", node.inputs[1])))?;
                    
                    let target_shape_data = target_shape_tensor.data.as_ref()
                        .ok_or_else(|| OptimizerError::Error("Reshape target shape has no data".to_string()))?;

                    let mut output_shape = Vec::new();
                    for j in 0..target_shape_tensor.shape[0] {
                        let offset = j * 8;
                        let val = i64::from_le_bytes(target_shape_data[offset..offset+8].try_into().unwrap());
                        output_shape.push(val as usize);
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Conv" => {
                    let shape_x = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    let shape_w = value_shapes.get(&node.inputs[1])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[1])))?;

                    let strides = match node.attributes.get("strides") {
                        Some(crate::ir::Attribute::Ints(s)) => s.clone(),
                        _ => vec![1, 1],
                    };
                    let pads = match node.attributes.get("pads") {
                        Some(crate::ir::Attribute::Ints(p)) => p.clone(),
                        _ => vec![0, 0, 0, 0],
                    };

                    let n = shape_x[0];
                    let m = shape_w[0];
                    let h_in = shape_x[2];
                    let w_in = shape_x[3];
                    let k_h = shape_w[2];
                    let w_h = shape_w[3];

                    let h_out = (h_in + pads[0] as usize + pads[2] as usize - k_h) / strides[0] as usize + 1;
                    let w_out = (w_in + pads[1] as usize + pads[3] as usize - w_h) / strides[1] as usize + 1;

                    let output_shape = vec![n, m, h_out, w_out];

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "MaxPool" => {
                    let shape_x = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    let kernel_shape = match node.attributes.get("kernel_shape") {
                        Some(crate::ir::Attribute::Ints(k)) => k.clone(),
                        _ => vec![1, 1],
                    };
                    let strides = match node.attributes.get("strides") {
                        Some(crate::ir::Attribute::Ints(s)) => s.clone(),
                        _ => vec![1, 1],
                    };
                    let pads = match node.attributes.get("pads") {
                        Some(crate::ir::Attribute::Ints(p)) => p.clone(),
                        _ => vec![0, 0, 0, 0],
                    };

                    let n = shape_x[0];
                    let c = shape_x[1];
                    let h_in = shape_x[2];
                    let w_in = shape_x[3];
                    let k_h = kernel_shape[0] as usize;
                    let k_w = kernel_shape[1] as usize;

                    let h_out = (h_in + pads[0] as usize + pads[2] as usize - k_h) / strides[0] as usize + 1;
                    let w_out = (w_in + pads[1] as usize + pads[3] as usize - k_w) / strides[1] as usize + 1;

                    let output_shape = vec![n, c, h_out, w_out];

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "BatchNormalization" => {
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
                "AveragePool" => {
                    let shape_x = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    let kernel_shape = match node.attributes.get("kernel_shape") {
                        Some(crate::ir::Attribute::Ints(k)) => k.clone(),
                        _ => vec![1, 1],
                    };
                    let strides = match node.attributes.get("strides") {
                        Some(crate::ir::Attribute::Ints(s)) => s.clone(),
                        _ => vec![1, 1],
                    };
                    let pads = match node.attributes.get("pads") {
                        Some(crate::ir::Attribute::Ints(p)) => p.clone(),
                        _ => vec![0, 0, 0, 0],
                    };

                    let n = shape_x[0];
                    let c = shape_x[1];
                    let h_in = shape_x[2];
                    let w_in = shape_x[3];
                    let k_h = kernel_shape[0] as usize;
                    let k_w = kernel_shape[1] as usize;

                    let h_out = (h_in + pads[0] as usize + pads[2] as usize - k_h) / strides[0] as usize + 1;
                    let w_out = (w_in + pads[1] as usize + pads[3] as usize - k_w) / strides[1] as usize + 1;

                    let output_shape = vec![n, c, h_out, w_out];

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "GlobalAveragePool" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    if shape.len() < 2 {
                        return Err(OptimizerError::Error("GlobalAveragePool requires at least 2D input".to_string()));
                    }

                    let mut output_shape = vec![shape[0], shape[1]];
                    for _ in 2..shape.len() {
                        output_shape.push(1);
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Flatten" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    let axis = match node.attributes.get("axis") {
                        Some(crate::ir::Attribute::Int(ax)) => *ax as i64,
                        _ => 1,
                    };
                    let axis = if axis < 0 { (shape.len() as i64 + axis) as usize } else { axis as usize };

                    let mut dim0 = 1;
                    for i in 0..axis { dim0 *= shape[i]; }
                    let mut dim1 = 1;
                    for i in axis..shape.len() { dim1 *= shape[i]; }

                    let output_shape = vec![dim0, dim1];

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Gemm" => {
                    let shape_a = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    let shape_b = value_shapes.get(&node.inputs[1])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[1])))?;

                    let trans_a = match node.attributes.get("transA") {
                        Some(crate::ir::Attribute::Int(i)) => *i != 0,
                        _ => false,
                    };
                    let trans_b = match node.attributes.get("transB") {
                        Some(crate::ir::Attribute::Int(i)) => *i != 0,
                        _ => false,
                    };

                    let m = if !trans_a { shape_a[0] } else { shape_a[1] };
                    let n = if !trans_b { shape_b[1] } else { shape_b[0] };

                    let output_shape = vec![m, n];

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Identity" => {
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
                "Gelu" => {
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
                "Scan" => {
                    let num_scan_inputs = match node.attributes.get("num_scan_inputs") {
                        Some(crate::ir::Attribute::Int(i)) => *i as usize,
                        _ => 1,
                    };

                    let num_states = node.inputs.len() - num_scan_inputs;
                    let seq_len = value_shapes.get(&node.inputs[num_states])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[num_states])))?[0];

                    for i in 0..num_states {
                        let state_shape = value_shapes.get(&node.inputs[i])
                            .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[i])))?
                            .clone();
                        
                        value_shapes.insert(node.outputs[i].clone(), state_shape.clone());
                        inferred_tensors.push(Tensor {
                            name: node.outputs[i].clone(),
                            shape: state_shape,
                            data_type: DataType::F32,
                            data: None,
                        });
                    }

                    for i in 0..(node.outputs.len() - num_states) {
                        let out_idx = num_states + i;
                        let state_shape = value_shapes.get(&node.inputs[0])
                            .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                        
                        let mut seq_shape = vec![seq_len];
                        for &d in state_shape {
                            seq_shape.push(d);
                        }

                        value_shapes.insert(node.outputs[out_idx].clone(), seq_shape.clone());
                        inferred_tensors.push(Tensor {
                            name: node.outputs[out_idx].clone(),
                            shape: seq_shape,
                            data_type: DataType::F32,
                            data: None,
                        });
                    }
                }
                "Slice" => {
                    let mut output_shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?
                        .clone();
                    
                    let starts_tensor = ir.graph.weights.get(&node.inputs[1])
                        .ok_or_else(|| OptimizerError::Error("Slice starts must be constant for now".to_string()))?;
                    let ends_tensor = ir.graph.weights.get(&node.inputs[2])
                        .ok_or_else(|| OptimizerError::Error("Slice ends must be constant for now".to_string()))?;
                    
                    let axes = if node.inputs.len() > 3 {
                        let axes_tensor = ir.graph.weights.get(&node.inputs[3])
                            .ok_or_else(|| OptimizerError::Error("Slice axes must be constant for now".to_string()))?;
                        let data = axes_tensor.data.as_ref().unwrap();
                        let mut res = Vec::new();
                        for j in 0..axes_tensor.shape[0] {
                            res.push(i64::from_le_bytes(data[j*8..j*8+8].try_into().unwrap()));
                        }
                        res
                    } else {
                        (0..output_shape.len() as i64).collect()
                    };

                    let steps = if node.inputs.len() > 4 {
                        let steps_tensor = ir.graph.weights.get(&node.inputs[4])
                            .ok_or_else(|| OptimizerError::Error("Slice steps must be constant for now".to_string()))?;
                        let data = steps_tensor.data.as_ref().unwrap();
                        let mut res = Vec::new();
                        for j in 0..steps_tensor.shape[0] {
                            res.push(i64::from_le_bytes(data[j*8..j*8+8].try_into().unwrap()));
                        }
                        res
                    } else {
                        vec![1; axes.len()]
                    };

                    let starts_data = starts_tensor.data.as_ref().unwrap();
                    let ends_data = ends_tensor.data.as_ref().unwrap();

                    for (i, &axis) in axes.iter().enumerate() {
                        let axis = if axis < 0 { (output_shape.len() as i64 + axis) as usize } else { axis as usize };
                        let start = i64::from_le_bytes(starts_data[i*8..i*8+8].try_into().unwrap());
                        let end = i64::from_le_bytes(ends_data[i*8..i*8+8].try_into().unwrap());
                        let step = steps[i];

                        let dim_size = output_shape[axis] as i64;
                        let start = if start < 0 { dim_size + start } else { start };
                        let start = start.clamp(0, dim_size);
                        let end = if end < 0 { dim_size + end } else { end };
                        let end = end.clamp(0, dim_size);

                        output_shape[axis] = (((end - start).abs() + step.abs() - 1) / step.abs()) as usize;
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Squeeze" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    let axes = if node.inputs.len() > 1 {
                        let axes_tensor = ir.graph.weights.get(&node.inputs[1])
                            .ok_or_else(|| OptimizerError::Error("Squeeze axes must be constant for now".to_string()))?;
                        let data = axes_tensor.data.as_ref().unwrap();
                        let mut res = Vec::new();
                        for j in 0..axes_tensor.shape[0] {
                            res.push(i64::from_le_bytes(data[j*8..j*8+8].try_into().unwrap()));
                        }
                        res
                    } else {
                        let mut res = Vec::new();
                        for (i, &d) in shape.iter().enumerate() {
                            if d == 1 { res.push(i as i64); }
                        }
                        res
                    };

                    let mut output_shape = Vec::new();
                    let axes_set: std::collections::HashSet<usize> = axes.iter().map(|&a| if a < 0 { (shape.len() as i64 + a) as usize } else { a as usize }).collect();
                    
                    for i in 0..shape.len() {
                        if !axes_set.contains(&i) {
                            output_shape.push(shape[i]);
                        }
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Unsqueeze" => {
                    let shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?;
                    
                    let axes_tensor = ir.graph.weights.get(&node.inputs[1])
                        .ok_or_else(|| OptimizerError::Error("Unsqueeze axes must be constant for now".to_string()))?;
                    let data = axes_tensor.data.as_ref().unwrap();
                    let mut axes = Vec::new();
                    for j in 0..axes_tensor.shape[0] {
                        axes.push(i64::from_le_bytes(data[j*8..j*8+8].try_into().unwrap()));
                    }

                    let mut output_shape = shape.clone();
                    axes.sort_unstable();
                    for &ax in &axes {
                        let ax = if ax < 0 { (output_shape.len() as i64 + 1 + ax) as usize } else { ax as usize };
                        output_shape.insert(ax, 1);
                    }

                    value_shapes.insert(node.outputs[0].clone(), output_shape.clone());
                    inferred_tensors.push(Tensor {
                        name: node.outputs[0].clone(),
                        shape: output_shape,
                        data_type: DataType::F32,
                        data: None,
                    });
                }
                "Softmax" => {
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
                "LayerNormalization" => {
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
                "Concat" => {
                    let axis = match node.attributes.get("axis") {
                        Some(crate::ir::Attribute::Int(ax)) => *ax as i64,
                        _ => 0,
                    };

                    let mut output_shape = value_shapes.get(&node.inputs[0])
                        .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[0])))?
                        .clone();
                    
                    let axis = if axis < 0 { (output_shape.len() as i64 + axis) as usize } else { axis as usize };

                    for i in 1..node.inputs.len() {
                        let shape = value_shapes.get(&node.inputs[i])
                            .ok_or_else(|| OptimizerError::Error(format!("Input {} not found", node.inputs[i])))?;
                        output_shape[axis] += shape[axis];
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
            if !ir.graph.outputs.iter().any(|t| t.name == tensor.name) {
                ir.graph.outputs.push(tensor);
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

            ir.graph.inputs.push(Tensor {

                name: "A".to_string(),

                shape: vec![1, 3, 224, 224],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.inputs.push(Tensor {

                name: "B".to_string(),

                shape: vec![1, 3, 224, 224],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "add".to_string(),

                op_type: "Add".to_string(),

                inputs: vec!["A".to_string(), "B".to_string()],

                outputs: vec!["C".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            assert_eq!(ir.graph.outputs.len(), 1);

            assert_eq!(ir.graph.outputs[0].shape, vec![1, 3, 224, 224]);

        }

    

        #[test]

        fn test_infer_relu_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 10],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "relu1".to_string(),

                op_type: "Relu".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 10]));

        }

    

        #[test]

        fn test_infer_einsum_dot_product() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "A".to_string(),

                shape: vec![10],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.inputs.push(Tensor {

                name: "B".to_string(),

                shape: vec![10],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("equation".to_string(), crate::ir::Attribute::String("i,i->".to_string()));

            ir.graph.nodes.push(Node {

                name: "einsum1".to_string(),

                op_type: "Einsum".to_string(),

                inputs: vec!["A".to_string(), "B".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![]));

        }

    

        #[test]

        fn test_infer_matmul_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "A".to_string(),

                shape: vec![5, 10],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.inputs.push(Tensor {

                name: "B".to_string(),

                shape: vec![10, 3],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "matmul1".to_string(),

                op_type: "MatMul".to_string(),

                inputs: vec!["A".to_string(), "B".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![5, 3]));

        }

    

        #[test]

        fn test_infer_transpose_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 2, 3],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("perm".to_string(), crate::ir::Attribute::Ints(vec![0, 2, 1]));

            ir.graph.nodes.push(Node {

                name: "transpose1".to_string(),

                op_type: "Transpose".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 3, 2]));

        }

    

        #[test]

        fn test_infer_reshape_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 6],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.weights.insert("shape".to_string(), Tensor {

                name: "shape".to_string(),

                shape: vec![2],

                data_type: DataType::I64,

                data: Some(vec![2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]),

            });

            ir.graph.nodes.push(Node {

                name: "reshape1".to_string(),

                op_type: "Reshape".to_string(),

                inputs: vec!["X".to_string(), "shape".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![2, 3]));

        }

    

        #[test]

        fn test_infer_conv2d_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 3, 224, 224],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.weights.insert("W".to_string(), Tensor {

                name: "W".to_string(),

                shape: vec![16, 3, 3, 3],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("strides".to_string(), crate::ir::Attribute::Ints(vec![2, 2]));

            attrs.insert("pads".to_string(), crate::ir::Attribute::Ints(vec![1, 1, 1, 1]));

            ir.graph.nodes.push(Node {

                name: "conv1".to_string(),

                op_type: "Conv".to_string(),

                inputs: vec!["X".to_string(), "W".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 16, 112, 112]));

        }

    

        #[test]

        fn test_infer_batch_norm_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 16, 112, 112],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.weights.insert("scale".to_string(), Tensor {

                name: "scale".to_string(),

                shape: vec![16],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "bn1".to_string(),

                op_type: "BatchNormalization".to_string(),

                inputs: vec!["X".to_string(), "scale".to_string(), "B".to_string(), "mean".to_string(), "var".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 16, 112, 112]));

        }

    

        #[test]

        fn test_infer_max_pool_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 16, 112, 112],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("kernel_shape".to_string(), crate::ir::Attribute::Ints(vec![2, 2]));

            attrs.insert("strides".to_string(), crate::ir::Attribute::Ints(vec![2, 2]));

            attrs.insert("pads".to_string(), crate::ir::Attribute::Ints(vec![0, 0, 0, 0]));

            ir.graph.nodes.push(Node {

                name: "pool1".to_string(),

                op_type: "MaxPool".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 16, 56, 56]));

        }

    

        #[test]

        fn test_infer_softmax_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 1000],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));

            ir.graph.nodes.push(Node {

                name: "softmax1".to_string(),

                op_type: "Softmax".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 1000]));

        }

    

        #[test]

        fn test_infer_average_pool_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 16, 112, 112],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("kernel_shape".to_string(), crate::ir::Attribute::Ints(vec![7, 7]));

            attrs.insert("strides".to_string(), crate::ir::Attribute::Ints(vec![1, 1]));

            attrs.insert("pads".to_string(), crate::ir::Attribute::Ints(vec![0, 0, 0, 0]));

            ir.graph.nodes.push(Node {

                name: "pool1".to_string(),

                op_type: "AveragePool".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 16, 106, 106]));

        }

    

        #[test]

        fn test_infer_layer_norm_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 10, 512],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "ln1".to_string(),

                op_type: "LayerNormalization".to_string(),

                inputs: vec!["X".to_string(), "scale".to_string(), "bias".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 10, 512]));

        }

    

        #[test]

        fn test_infer_concat_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "A".to_string(),

                shape: vec![1, 10, 256],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.inputs.push(Tensor {

                name: "B".to_string(),

                shape: vec![1, 10, 256],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("axis".to_string(), crate::ir::Attribute::Int(2));

            ir.graph.nodes.push(Node {

                name: "concat1".to_string(),

                op_type: "Concat".to_string(),

                inputs: vec!["A".to_string(), "B".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 10, 512]));

        }

    

        #[test]

        fn test_infer_global_average_pool_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 1280, 7, 7],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "gap1".to_string(),

                op_type: "GlobalAveragePool".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 1280, 1, 1]));

        }

    

        #[test]

        fn test_infer_flatten_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 3, 224, 224],

                data_type: DataType::F32,

                data: None,

            });

            let mut attrs = HashMap::new();

            attrs.insert("axis".to_string(), crate::ir::Attribute::Int(1));

            ir.graph.nodes.push(Node {

                name: "flatten1".to_string(),

                op_type: "Flatten".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: attrs,

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 150528]));

        }

    

        #[test]

        fn test_infer_gemm_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "A".to_string(),

                shape: vec![1, 512],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.weights.insert("W".to_string(), Tensor {

                name: "W".to_string(),

                shape: vec![512, 1000],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.weights.insert("B".to_string(), Tensor {

                name: "B".to_string(),

                shape: vec![1000],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "gemm1".to_string(),

                op_type: "Gemm".to_string(),

                inputs: vec!["A".to_string(), "W".to_string(), "B".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 1000]));

        }

    

        #[test]

        fn test_infer_identity_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 3, 224, 224],

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

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 3, 224, 224]));

        }

    

        #[test]

        fn test_infer_gelu_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "X".to_string(),

                shape: vec![1, 512],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "gelu1".to_string(),

                op_type: "Gelu".to_string(),

                inputs: vec!["X".to_string()],

                outputs: vec!["Y".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            assert_eq!(y_shape, Some(&vec![1, 512]));

        }

    

        #[test]

        fn test_infer_scan_shape() {

            let mut ir = ModelIR::new();

            ir.graph.inputs.push(Tensor {

                name: "initial_h".to_string(),

                shape: vec![1, 128],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.inputs.push(Tensor {

                name: "x_seq".to_string(),

                shape: vec![10, 1, 64],

                data_type: DataType::F32,

                data: None,

            });

            ir.graph.nodes.push(Node {

                name: "scan1".to_string(),

                op_type: "Scan".to_string(),

                inputs: vec!["initial_h".to_string(), "x_seq".to_string()],

                outputs: vec!["final_h".to_string(), "y_seq".to_string()],

                attributes: HashMap::new(),

            });

            ShapeInference::infer(&mut ir).unwrap();

            let final_h_shape = ir.graph.outputs.iter().find(|t| t.name == "final_h").map(|t| &t.shape);

            let y_seq_shape = ir.graph.outputs.iter().find(|t| t.name == "y_seq").map(|t| &t.shape);

                    assert_eq!(final_h_shape, Some(&vec![1, 128]));

                    assert_eq!(y_seq_shape, Some(&vec![10, 1, 128]));

                }

            

                #[test]

                fn test_infer_slice_shape() {

                    let mut ir = ModelIR::new();

                    

                    ir.graph.inputs.push(Tensor {

                        name: "X".to_string(),

                        shape: vec![1, 10, 20],

                        data_type: DataType::F32,

                        data: None,

                    });

            

                    ir.graph.weights.insert("starts".to_string(), Tensor {

                        name: "starts".to_string(),

                        shape: vec![1],

                        data_type: DataType::I64,

                        data: Some(vec![5, 0, 0, 0, 0, 0, 0, 0]),

                    });

            

                    ir.graph.weights.insert("ends".to_string(), Tensor {

                        name: "ends".to_string(),

                        shape: vec![1],

                        data_type: DataType::I64,

                        data: Some(vec![15, 0, 0, 0, 0, 0, 0, 0]),

                    });

            

                    ir.graph.weights.insert("axes".to_string(), Tensor {

                        name: "axes".to_string(),

                        shape: vec![1],

                        data_type: DataType::I64,

                        data: Some(vec![2, 0, 0, 0, 0, 0, 0, 0]),

                    });

            

                    ir.graph.nodes.push(Node {

                        name: "slice1".to_string(),

                        op_type: "Slice".to_string(),

                        inputs: vec!["X".to_string(), "starts".to_string(), "ends".to_string(), "axes".to_string()],

                        outputs: vec!["Y".to_string()],

                        attributes: HashMap::new(),

                    });

            

                    ShapeInference::infer(&mut ir).unwrap();

            

                            let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            

                            assert_eq!(y_shape, Some(&vec![1, 10, 10]));

            

                        }

            

                    

            

                        #[test]

            

                        fn test_infer_squeeze_shape() {

            

                            let mut ir = ModelIR::new();

            

                            

            

                            ir.graph.inputs.push(Tensor {

            

                                name: "X".to_string(),

            

                                shape: vec![1, 10, 1, 20],

            

                                data_type: DataType::F32,

            

                                data: None,

            

                            });

            

                    

            

                            ir.graph.weights.insert("axes".to_string(), Tensor {

            

                                name: "axes".to_string(),

            

                                shape: vec![2],

            

                                data_type: DataType::I64,

            

                                data: Some(vec![0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]),

            

                            });

            

                    

            

                            ir.graph.nodes.push(Node {

            

                                name: "squeeze1".to_string(),

            

                                op_type: "Squeeze".to_string(),

            

                                inputs: vec!["X".to_string(), "axes".to_string()],

            

                                outputs: vec!["Y".to_string()],

            

                                attributes: HashMap::new(),

            

                            });

            

                    

            

                            ShapeInference::infer(&mut ir).unwrap();

            

                    

            

                                    let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            

                    

            

                                    assert_eq!(y_shape, Some(&vec![10, 20]));

            

                    

            

                                }

            

                    

            

                            

            

                    

            

                                #[test]

            

                    

            

                                fn test_infer_unsqueeze_shape() {

            

                    

            

                                    let mut ir = ModelIR::new();

            

                    

            

                                    

            

                    

            

                                    ir.graph.inputs.push(Tensor {

            

                    

            

                                        name: "X".to_string(),

            

                    

            

                                        shape: vec![10, 20],

            

                    

            

                                        data_type: DataType::F32,

            

                    

            

                                        data: None,

            

                    

            

                                    });

            

                    

            

                            

            

                    

            

                                    ir.graph.weights.insert("axes".to_string(), Tensor {

            

                    

            

                                        name: "axes".to_string(),

            

                    

            

                                        shape: vec![2],

            

                    

            

                                        data_type: DataType::I64,

            

                    

            

                                        data: Some(vec![0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]),

            

                    

            

                                    });

            

                    

            

                            

            

                    

            

                                    ir.graph.nodes.push(Node {

            

                    

            

                                        name: "unsqueeze1".to_string(),

            

                    

            

                                        op_type: "Unsqueeze".to_string(),

            

                    

            

                                        inputs: vec!["X".to_string(), "axes".to_string()],

            

                    

            

                                        outputs: vec!["Y".to_string()],

            

                    

            

                                        attributes: HashMap::new(),

            

                    

            

                                    });

            

                    

            

                            

            

                    

            

                                    ShapeInference::infer(&mut ir).unwrap();

            

                    

            

                            

            

                    

            

                                    let y_shape = ir.graph.outputs.iter().find(|t| t.name == "Y").map(|t| &t.shape);

            

                    

            

                                    assert_eq!(y_shape, Some(&vec![1, 10, 1, 20]));

            

                    

            

                                }

            

                    

            

                            }

            

                    

            

                            

            

                    

            

    
                        