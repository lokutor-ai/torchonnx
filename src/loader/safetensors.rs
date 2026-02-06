use std::path::Path;
use crate::ir::ModelIR;
use crate::loader::{ModelLoader, LoaderError};

pub struct SafetensorsLoader;

impl ModelLoader for SafetensorsLoader {
    fn load<P: AsRef<Path>>(_path: P) -> Result<ModelIR, LoaderError> {
        // TODO: Implement
        Err(LoaderError::Other("Not implemented".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_safetensors_not_implemented() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        File::create(&file_path).unwrap();

        let result = SafetensorsLoader::load(&file_path);
        assert!(result.is_err());
    }
}
