use std::path::Path;
use crate::ir::ModelIR;
use crate::loader::{ModelLoader, LoaderError};
use zip::ZipArchive;
use std::fs::File;

pub struct PytorchLoader;

impl ModelLoader for PytorchLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<ModelIR, LoaderError> {
        let file = File::open(path)?;
        let _archive = ZipArchive::new(file)
            .map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;

        Err(LoaderError::Other("PyTorch loader not fully implemented yet".to_string()))
    }
}
