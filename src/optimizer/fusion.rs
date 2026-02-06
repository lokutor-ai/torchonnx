use crate::ir::ModelIR;
use crate::optimizer::{OptimizationPass, OptimizerError};

pub struct OperatorFusion;

impl OptimizationPass for OperatorFusion {
    fn apply(&self, ir: &mut ModelIR) -> Result<(), OptimizerError> {
        let mut i = 0;
        while i < ir.nodes.len() {
            if ir.nodes[i].op_type == "Conv" {
                let conv_output = ir.nodes[i].outputs[0].clone();
                
                let mut next_node_idx = None;
                for j in 0..ir.nodes.len() {
                    if i != j && ir.nodes[j].op_type == "BatchNormalization" && ir.nodes[j].inputs[0] == conv_output {
                        next_node_idx = Some(j);
                        break;
                    }
                }

                if let Some(bn_idx) = next_node_idx {
                    let bn_output = ir.nodes[bn_idx].outputs[0].clone();
                    
                    ir.nodes[i].outputs[0] = bn_output;
                    
                    ir.nodes.remove(bn_idx);
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
    use crate::ir::Node;
    use std::collections::HashMap;

    #[test]
    fn test_fuse_conv_bn() {
        let mut ir = ModelIR::new();
        
        ir.nodes.push(Node {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            inputs: vec!["X".to_string(), "W".to_string(), "B".to_string()],
            outputs: vec!["conv_out".to_string()],
            attributes: HashMap::new(),
        });

        ir.nodes.push(Node {
            name: "bn".to_string(),
            op_type: "BatchNormalization".to_string(),
            inputs: vec!["conv_out".to_string(), "scale".to_string(), "bias".to_string(), "mean".to_string(), "var".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        });

        let fusion = OperatorFusion;
        fusion.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.nodes[0].op_type, "Conv");
        assert_eq!(ir.nodes[0].outputs[0], "Y");
    }
}
