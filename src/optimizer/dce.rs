use crate::ir::ModelIR;
use crate::optimizer::{OptimizationPass, OptimizerError};
use std::collections::HashSet;

pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn apply(&self, ir: &mut ModelIR) -> Result<(), OptimizerError> {
        let mut changed = true;
        while changed {
            changed = false;
            let mut used_values = HashSet::new();
            
            for output in &ir.outputs {
                used_values.insert(output.name.clone());
            }

            for node in &ir.nodes {
                for input in &node.inputs {
                    used_values.insert(input.clone());
                }
            }

            let initial_len = ir.nodes.len();
            ir.nodes.retain(|node| {
                node.outputs.iter().any(|output| used_values.contains(output))
            });

            if ir.nodes.len() != initial_len {
                changed = true;
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
    fn test_dce_removes_unused_node() {
        let mut ir = ModelIR::new();
        
        ir.nodes.push(Node {
            name: "unused_node".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        });

        ir.outputs.push(Tensor {
            name: "Other".to_string(),
            shape: vec![1],
            data_type: DataType::F32,
            data: None,
        });

        let dce = DeadCodeElimination;
        dce.apply(&mut ir).unwrap();

        assert_eq!(ir.nodes.len(), 0);
    }
}
