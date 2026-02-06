use std::path::Path;
use crate::ir::ModelIR;
use crate::loader::{ModelLoader, LoaderError};
use zip::ZipArchive;
use std::fs::File;
use std::io::Read;

pub struct PytorchLoader;

impl ModelLoader for PytorchLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<ModelIR, LoaderError> {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)
            .map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;

        let mut pkl_data = Vec::new();
        let mut found_pkl = false;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
            if file.name().ends_with("data.pkl") {
                file.read_to_end(&mut pkl_data)?;
                found_pkl = true;
                break;
            }
        }

        if !found_pkl {
            return Err(LoaderError::InvalidFormat("data.pkl not found in archive".to_string()));
        }

        let _decoded: serde_pickle::Value = serde_pickle::from_slice(&pkl_data, serde_pickle::DeOptions::default())
            .map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;

        Err(LoaderError::Other("Reconstruction from Pickle not yet implemented".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use zip::write::FileOptions;
    use std::io::Write;

    #[test]
    fn test_load_pt_archive_found() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.pt");
        let file = File::create(&file_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);

        zip.start_file("archive/data.pkl", FileOptions::default()).unwrap();
        let pickle_data = serde_pickle::to_vec(&vec![1, 2, 3], serde_pickle::SerOptions::default()).unwrap();
        zip.write_all(&pickle_data).unwrap();
        zip.finish().unwrap();

        let result = PytorchLoader::load(&file_path);
        match result {
            Err(LoaderError::Other(s)) => assert_eq!(s, "Reconstruction from Pickle not yet implemented"),
            _ => panic!("Expected Reconstruction error, got {:?}", result),
        }
    }
}
