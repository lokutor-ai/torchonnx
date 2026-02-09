use std::path::Path;
use crate::ir::ModelIR;
use crate::loader::{ModelLoader, LoaderError};
use zip::ZipArchive;
use std::fs::File;
use std::io::Read;

pub mod pytorch_pickle;

pub struct PytorchLoader;

impl ModelLoader for PytorchLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<ModelIR, LoaderError> {
        let loader = PytorchLoader;
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;

        let mut pickle_data = Vec::new();
        let mut data_pkl_path = String::new();
        for i in 0..archive.len() {
            let file = archive.by_index(i).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
            if file.name().ends_with("data.pkl") {
                data_pkl_path = file.name().to_string();
                break;
            }
        }

        if data_pkl_path.is_empty() {
            return Err(LoaderError::InvalidFormat("Could not find data.pkl in archive".to_string()));
        }

        {
            let mut data_pkl_file = archive.by_name(&data_pkl_path).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
            data_pkl_file.read_to_end(&mut pickle_data)?;
        }

        let mut unpickler = pytorch_pickle::PytorchUnpickler::new(&pickle_data);
        let pickle_root = unpickler.parse()?;

        let mut ir = ModelIR::new();
        loader.extract_tensors(&pickle_root, &mut archive, &mut ir)?;

        Ok(ir)
    }
}

impl PytorchLoader {
    fn extract_tensors(&self, val: &pytorch_pickle::PickleValue, archive: &mut ZipArchive<File>, ir: &mut ModelIR) -> Result<(), LoaderError> {
        match val {
            pytorch_pickle::PickleValue::Dict(dict) => {
                for (k, v) in dict {
                    if let pytorch_pickle::PickleValue::PersistentId(pid) = v {
                        self.load_tensor_from_pid(k, pid, archive, ir, None)?;
                    } else if let pytorch_pickle::PickleValue::Reduced { obj, args } = v {
                        self.handle_reduced(k, obj, args, archive, ir)?;
                    } else {
                        self.extract_tensors(v, archive, ir)?;
                    }
                }
            }
            pytorch_pickle::PickleValue::List(list) | pytorch_pickle::PickleValue::Tuple(list) => {
                for item in list {
                    self.extract_tensors(item, archive, ir)?;
                }
            }
            pytorch_pickle::PickleValue::Object { dict: Some(dict), .. } => {
                for (k, v) in dict {
                    if let pytorch_pickle::PickleValue::PersistentId(pid) = v {
                        self.load_tensor_from_pid(k, pid, archive, ir, None)?;
                    } else if let pytorch_pickle::PickleValue::Reduced { obj, args } = v {
                        self.handle_reduced(k, obj, args, archive, ir)?;
                    } else {
                        self.extract_tensors(v, archive, ir)?;
                    }
                }
            }
            pytorch_pickle::PickleValue::Reduced { obj, args } => {
                self.handle_reduced("unnamed", obj, args, archive, ir)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_reduced(&self, name: &str, obj: &pytorch_pickle::PickleValue, args: &pytorch_pickle::PickleValue, archive: &mut ZipArchive<File>, ir: &mut ModelIR) -> Result<(), LoaderError> {
        if let pytorch_pickle::PickleValue::Object { class_name, .. } = obj {
            if class_name == "_rebuild_tensor_v2" {
                if let pytorch_pickle::PickleValue::Tuple(items) = args {
                    if items.len() >= 3 {
                        let pid = &items[0];
                        let shape_val = &items[2];
                        let mut shape = Vec::new();
                        if let pytorch_pickle::PickleValue::Tuple(s_items) = shape_val {
                            for s in s_items {
                                if let pytorch_pickle::PickleValue::Int(dim) = s {
                                    shape.push(*dim as usize);
                                }
                            }
                        }
                        self.load_tensor_from_pid(name, pid, archive, ir, Some(shape))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn load_tensor_from_pid(&self, name: &str, val: &pytorch_pickle::PickleValue, archive: &mut ZipArchive<File>, ir: &mut ModelIR, shape: Option<Vec<usize>>) -> Result<(), LoaderError> {
        match val {
            pytorch_pickle::PickleValue::PersistentId(inner) => {
                self.load_tensor_from_pid(name, inner, archive, ir, shape)
            }
            pytorch_pickle::PickleValue::Reduced { obj: _, args } => {
                // Usually args[0] is the PID
                if let pytorch_pickle::PickleValue::Tuple(items) = args.as_ref() {
                    if !items.is_empty() {
                        self.load_tensor_from_pid(name, &items[0], archive, ir, shape)?;
                    }
                }
                Ok(())
            }
            pytorch_pickle::PickleValue::Tuple(items) => {
                if items.len() >= 3 {
                    if let (pytorch_pickle::PickleValue::String(s), pytorch_pickle::PickleValue::String(dt_str), pytorch_pickle::PickleValue::String(key)) = (&items[0], &items[1], &items[2]) {
                        if s == "storage" {
                            let mut data_path = String::new();
                            for i in 0..archive.len() {
                                let file = archive.by_index(i).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                                if file.name().ends_with(&format!("data/{}", key)) {
                                    data_path = file.name().to_string();
                                    break;
                                }
                            }

                            if !data_path.is_empty() {
                                let mut file = archive.by_name(&data_path).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                                let mut data = Vec::new();
                                file.read_to_end(&mut data)?;

                                let (data_type, element_size) = match dt_str.as_str() {
                                    "float" | "FloatStorage" => (crate::ir::DataType::F32, 4),
                                    "double" | "DoubleStorage" => (crate::ir::DataType::F64, 8),
                                    "long" | "LongStorage" => (crate::ir::DataType::I64, 8),
                                    "int" | "IntStorage" => (crate::ir::DataType::I32, 4),
                                    "byte" | "ByteStorage" => (crate::ir::DataType::U8, 1),
                                    _ => (crate::ir::DataType::F32, 4),
                                };

                                let final_shape = shape.unwrap_or_else(|| vec![data.len() / element_size]);

                                ir.graph.weights.insert(name.to_string(), crate::ir::Tensor {
                                    name: name.to_string(),
                                    shape: final_shape,
                                    data_type,
                                    data: Some(data),
                                });
                            }
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use zip::write::FileOptions;
    use std::io::Write;

    #[test]
    fn test_load_pt_rebuild_tensor() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.pt");
        let file = File::create(&file_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);

        zip.start_file("archive/data.pkl", FileOptions::default()).unwrap();
        // proto 2, {
        //   'weight': REDUCE(
        //     _rebuild_tensor_v2,
        //     MARK,
        //       PERSID(MARK, 'storage', 'float', '0', 'cpu', 4, TUPLE),
        //       0,
        //       (2, 2),
        //       (2, 1),
        //       False,
        //       [],
        //     TUPLE
        //   )
        // }, STOP
        let pickle_data = b"\x80\x02}(X\x06\x00\x00\x00weightc\x0btorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storageX\x05\x00\x00\x00floatX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x04tQK\x00(K\x02K\x02t(K\x02K\x01t\x89]tRu.";
        zip.write_all(pickle_data).unwrap();

        zip.start_file("archive/data/0", FileOptions::default()).unwrap();
        zip.write_all(&[0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]).unwrap();
        zip.finish().unwrap();

        let result = PytorchLoader::load(&file_path).unwrap();
        assert!(result.graph.weights.contains_key("weight"));
        assert_eq!(result.graph.weights["weight"].shape, vec![2, 2]);
    }
}