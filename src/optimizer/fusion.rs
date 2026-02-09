use crate::ir::{ModelIR, Tensor, DataType};
use crate::optimizer::{OptimizationPass, OptimizerError};

pub struct OperatorFusion;

impl OptimizationPass for OperatorFusion {
    fn apply(&self, ir: &mut ModelIR) -> Result<(), OptimizerError> {
        let mut i = 0;
        while i < ir.graph.nodes.len() {
            if ir.graph.nodes[i].op_type == "Conv" {
                let conv_output = ir.graph.nodes[i].outputs[0].clone();
                
                let mut next_node_idx = None;
                for j in 0..ir.graph.nodes.len() {
                    if i != j && ir.graph.nodes[j].op_type == "BatchNormalization" && ir.graph.nodes[j].inputs[0] == conv_output {
                        next_node_idx = Some(j);
                        break;
                    }
                }

                if let Some(bn_idx) = next_node_idx {
                    let conv_node = &ir.graph.nodes[i];
                    let bn_node = &ir.graph.nodes[bn_idx];

                    let weight_name = &conv_node.inputs[1];
                    let bias_name = if conv_node.inputs.len() > 2 { Some(&conv_node.inputs[2]) } else { None };

                    let scale_name = &bn_node.inputs[1];
                    let b_bn_name = &bn_node.inputs[2];
                    let mean_name = &bn_node.inputs[3];
                    let var_name = &bn_node.inputs[4];

                    let epsilon = match bn_node.attributes.get("epsilon") {
                        Some(crate::ir::Attribute::Float(f)) => *f,
                        _ => 1e-5,
                    };

                    if ir.graph.weights.contains_key(weight_name) && 
                       ir.graph.weights.contains_key(scale_name) && 
                       ir.graph.weights.contains_key(b_bn_name) && 
                       ir.graph.weights.contains_key(mean_name) && 
                       ir.graph.weights.contains_key(var_name) {
                        
                        let w = ir.graph.weights[weight_name].clone();
                        let scale = ir.graph.weights[scale_name].clone();
                        let b_bn = ir.graph.weights[b_bn_name].clone();
                        let mean = ir.graph.weights[mean_name].clone();
                        let var = ir.graph.weights[var_name].clone();

                        let w_data: Vec<f32> = unsafe { std::slice::from_raw_parts(w.data.as_ref().unwrap().as_ptr() as *const f32, w.data.as_ref().unwrap().len() / 4).to_vec() };
                        let scale_data: Vec<f32> = unsafe { std::slice::from_raw_parts(scale.data.as_ref().unwrap().as_ptr() as *const f32, scale.data.as_ref().unwrap().len() / 4).to_vec() };
                        let b_bn_data: Vec<f32> = unsafe { std::slice::from_raw_parts(b_bn.data.as_ref().unwrap().as_ptr() as *const f32, b_bn.data.as_ref().unwrap().len() / 4).to_vec() };
                        let mean_data: Vec<f32> = unsafe { std::slice::from_raw_parts(mean.data.as_ref().unwrap().as_ptr() as *const f32, mean.data.as_ref().unwrap().len() / 4).to_vec() };
                        let var_data: Vec<f32> = unsafe { std::slice::from_raw_parts(var.data.as_ref().unwrap().as_ptr() as *const f32, var.data.as_ref().unwrap().len() / 4).to_vec() };

                        let out_channels = w.shape[0];
                        let items_per_channel = w_data.len() / out_channels;

                        let mut new_w_data = vec![0.0f32; w_data.len()];
                        let mut new_b_data = vec![0.0f32; out_channels];

                        let old_b_data = if let Some(b_name) = bias_name {
                            if ir.graph.weights.contains_key(b_name) {
                                unsafe { std::slice::from_raw_parts(ir.graph.weights[b_name].data.as_ref().unwrap().as_ptr() as *const f32, out_channels).to_vec() }
                            } else {
                                vec![0.0f32; out_channels]
                            }
                        } else {
                            vec![0.0f32; out_channels]
                        };

                        for oc in 0..out_channels {
                            let gamma = scale_data[oc];
                            let beta = b_bn_data[oc];
                            let mu = mean_data[oc];
                            let sigma2 = var_data[oc];
                            let factor = gamma / (sigma2 + epsilon).sqrt();

                            for i in 0..items_per_channel {
                                let idx = oc * items_per_channel + i;
                                new_w_data[idx] = w_data[idx] * factor;
                            }
                            new_b_data[oc] = (old_b_data[oc] - mu) * factor + beta;
                        }

                        let new_w_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(new_w_data.as_ptr() as *const u8, new_w_data.len() * 4).to_vec() };
                        let new_b_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(new_b_data.as_ptr() as *const u8, new_b_data.len() * 4).to_vec() };

                        ir.graph.weights.get_mut(weight_name).unwrap().data = Some(new_w_bytes);
                        
                        if let Some(b_name) = bias_name {
                            if ir.graph.weights.contains_key(b_name) {
                                ir.graph.weights.get_mut(b_name).unwrap().data = Some(new_b_bytes);
                            } else {
                                let new_b_name = format!("{}_fused_bias", conv_node.name);
                                ir.graph.weights.insert(new_b_name.clone(), Tensor {
                                    name: new_b_name.clone(),
                                    shape: vec![out_channels],
                                    data_type: DataType::F32,
                                    data: Some(new_b_bytes),
                                });
                                ir.graph.nodes[i].inputs.push(new_b_name);
                            }
                        } else {
                            let new_b_name = format!("{}_fused_bias", conv_node.name);
                            ir.graph.weights.insert(new_b_name.clone(), Tensor {
                                name: new_b_name.clone(),
                                shape: vec![out_channels],
                                data_type: DataType::F32,
                                data: Some(new_b_bytes),
                            });
                            ir.graph.nodes[i].inputs.push(new_b_name);
                        }

                        let bn_output = ir.graph.nodes[bn_idx].outputs[0].clone();
                        ir.graph.nodes[i].outputs[0] = bn_output;
                        ir.graph.nodes.remove(bn_idx);
                        continue;
                    }
                }
            } else if ir.graph.nodes[i].op_type == "Add" {
                let add_output = ir.graph.nodes[i].outputs[0].clone();
                
                let mut next_node_idx = None;
                for j in 0..ir.graph.nodes.len() {
                    if i != j && ir.graph.nodes[j].op_type == "Relu" && ir.graph.nodes[j].inputs[0] == add_output {
                        next_node_idx = Some(j);
                        break;
                    }
                }

                if let Some(relu_idx) = next_node_idx {
                    let relu_output = ir.graph.nodes[relu_idx].outputs[0].clone();
                    
                    ir.graph.nodes[i].op_type = "FusedAddRelu".to_string();
                    ir.graph.nodes[i].outputs[0] = relu_output;
                    
                    ir.graph.nodes.remove(relu_idx);
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
    use crate::ir::{Node, Tensor, DataType, ModelIR};
    use std::collections::HashMap;

    #[test]
    fn test_fuse_conv_bn() {
        let mut ir = ModelIR::new();
        ir.graph.weights.insert("W".to_string(), Tensor {
            name: "W".to_string(),
            shape: vec![16, 3, 3, 3],
            data_type: DataType::F32,
            data: Some(vec![0; 16 * 3 * 3 * 3 * 4]),
        });
        ir.graph.weights.insert("scale".to_string(), Tensor {
            name: "scale".to_string(),
            shape: vec![16],
            data_type: DataType::F32,
            data: Some(vec![0; 16 * 4]),
        });
        ir.graph.weights.insert("bias".to_string(), Tensor {
            name: "bias".to_string(),
            shape: vec![16],
            data_type: DataType::F32,
            data: Some(vec![0; 16 * 4]),
        });
        ir.graph.weights.insert("mean".to_string(), Tensor {
            name: "mean".to_string(),
            shape: vec![16],
            data_type: DataType::F32,
            data: Some(vec![0; 16 * 4]),
        });
        ir.graph.weights.insert("var".to_string(), Tensor {
            name: "var".to_string(),
            shape: vec![16],
            data_type: DataType::F32,
            data: Some(vec![0; 16 * 4]),
        });
        ir.graph.nodes.push(Node {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            inputs: vec!["X".to_string(), "W".to_string()],
            outputs: vec!["conv_out".to_string()],
            attributes: HashMap::new(),
        });
        ir.graph.nodes.push(Node {
            name: "bn".to_string(),
            op_type: "BatchNormalization".to_string(),
            inputs: vec!["conv_out".to_string(), "scale".to_string(), "bias".to_string(), "mean".to_string(), "var".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });
        let fusion = OperatorFusion;
        fusion.apply(&mut ir).unwrap();
        assert_eq!(ir.graph.nodes.len(), 1);
        assert_eq!(ir.graph.nodes[0].op_type, "Conv");
        assert_eq!(ir.graph.nodes[0].outputs[0], "Y");
    }

    #[test]
    fn test_fuse_conv_bn_weights() {
        let mut ir = ModelIR::new();
        ir.graph.weights.insert("W".to_string(), Tensor {
            name: "W".to_string(),
            shape: vec![1, 1, 1, 1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]),
        });
        ir.graph.weights.insert("B_conv".to_string(), Tensor {
            name: "B_conv".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 0]),
        });
        ir.graph.weights.insert("scale".to_string(), Tensor {
            name: "scale".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 64]),
        });
        ir.graph.weights.insert("bias".to_string(), Tensor {
            name: "bias".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]),
        });
        ir.graph.weights.insert("mean".to_string(), Tensor {
            name: "mean".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 0, 0]),
        });
        ir.graph.weights.insert("var".to_string(), Tensor {
            name: "var".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: Some(vec![0, 0, 128, 63]),
        });
        ir.graph.nodes.push(Node {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            inputs: vec!["X".to_string(), "W".to_string(), "B_conv".to_string()],
            outputs: vec!["conv_out".to_string()],
            attributes: HashMap::new(),
        });
        let mut bn_attrs = HashMap::new();
        bn_attrs.insert("epsilon".to_string(), crate::ir::Attribute::Float(1e-5));
        ir.graph.nodes.push(Node {
            name: "bn".to_string(),
            op_type: "BatchNormalization".to_string(),
            inputs: vec!["conv_out".to_string(), "scale".to_string(), "bias".to_string(), "mean".to_string(), "var".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: bn_attrs,
        });
        let fusion = OperatorFusion;
        fusion.apply(&mut ir).unwrap();
        let fused_w = &ir.graph.weights["W"];
        let w_data: f32 = f32::from_le_bytes(fused_w.data.as_ref().unwrap()[0..4].try_into().unwrap());
        assert!((w_data - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_fuse_add_relu() {
        let mut ir = ModelIR::new();
        ir.graph.nodes.push(Node {
            name: "add".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["add_out".to_string()],
            attributes: HashMap::new(),
        });
        ir.graph.nodes.push(Node {
            name: "relu".to_string(),
            op_type: "Relu".to_string(),
            inputs: vec!["add_out".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });
        let fusion = OperatorFusion;
        fusion.apply(&mut ir).unwrap();
        assert_eq!(ir.graph.nodes.len(), 1);
        assert_eq!(ir.graph.nodes[0].op_type, "FusedAddRelu");
        assert_eq!(ir.graph.nodes[0].outputs[0], "Y");
    }
}