use std::path::Path;
use crate::ir::{ModelIR, Tensor, DataType};
use crate::loader::{ModelLoader, LoaderError};
use safetensors::SafeTensors;
use std::fs;

pub struct SafetensorsLoader;

impl ModelLoader for SafetensorsLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<ModelIR, LoaderError> {
        let data = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;

        let mut ir = ModelIR::new();

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data_type = match view.dtype() {
                safetensors::Dtype::F32 => DataType::F32,
                safetensors::Dtype::F64 => DataType::F64,
                safetensors::Dtype::I32 => DataType::I32,
                safetensors::Dtype::I64 => DataType::I64,
                safetensors::Dtype::U8 => DataType::U8,
                _ => return Err(LoaderError::UnsupportedVersion(format!("{:?}", view.dtype()))),
            };

            let tensor = Tensor {
                name: name.clone(),
                shape,
                data_type,
                data: Some(view.data().to_vec()),
            };

            ir.weights.insert(name.clone(), tensor);
        }

        Ok(ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use safetensors::serialize;
    use std::collections::HashMap;

    #[test]
    fn test_load_valid_safetensors() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        
        let mut data: HashMap<String, Vec<f32>> = HashMap::new();
        data.insert("weight".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        
        let tensor_data: Vec<u8> = vec![0; 16];
        let metadata = HashMap::from([
            ("weight".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &tensor_data).unwrap())
        ]);
        
        let out = serialize(metadata, &None).unwrap();
        fs::write(&file_path, out).unwrap();

        let ir = SafetensorsLoader::load(&file_path).unwrap();
        assert!(ir.weights.contains_key("weight"));
        assert_eq!(ir.weights["weight"].shape, vec![2, 2]);
    }
}
