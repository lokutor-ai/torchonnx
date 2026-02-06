use crate::ir::ModelIR;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("Optimization error: {0}")]
    Error(String),
}

pub trait OptimizationPass {
    fn apply(&self, ir: &mut ModelIR) -> Result<(), OptimizerError>;
}

pub mod dce;
pub mod constant_folding;
pub mod fusion;

pub struct Optimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl Optimizer {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    pub fn optimize(&self, ir: &mut ModelIR) -> Result<(), OptimizerError> {
        for pass in &self.passes {
            pass.apply(ir)?;
        }
        Ok(())
    }
}
