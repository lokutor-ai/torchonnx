# Loader Design

The Loader component is responsible for parsing external model formats into our internal IR.

## Supported Formats

1.  **PyTorch (.pt)**:
    *   Legacy models are serialized using Python's `pickle`.
    *   Modern models (TorchScript) are ZIP archives containing `data.pkl` and tensor files.
    *   Need a robust Pickle parser in Rust (or use an existing one if compatible with PyTorch's custom extensions).
2.  **Safetensors (.safetensors)**:
    *   A simple, memory-mappable format for tensors.
    *   Header is a JSON string followed by raw buffer data.

## Requirements

*   **Lazy Loading**: Avoid loading all tensors into memory at once. Use memory mapping (`mmap`) where possible.
*   **Metadata Extraction**: Capture model version, producer, and potential attributes.
*   **Unified Interface**:
    ```rust
    trait ModelLoader {
        fn load(path: &Path) -> Result<ModelIR, LoaderError>;
    }
    ```
