use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: DataType,
    pub data: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, Attribute>,
}

#[derive(Debug, Clone)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

#[derive(Debug, Clone)]
pub struct ModelIR {
    pub nodes: Vec<Node>,
    pub weights: HashMap<String, Tensor>,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
}

impl ModelIR {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            weights: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}
