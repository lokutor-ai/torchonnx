# PyTorch (.pt) Loader Design

Loading PyTorch `.pt` files is challenging because they use Python's `pickle` serialization, often wrapped in a ZIP archive.

## File Structure (Modern TorchScript/Weights)

A `.pt` file is typically a ZIP archive with the following structure:
*   `archive/data.pkl`: The serialized model structure and metadata.
*   `archive/data/0`, `archive/data/1`, ...: Binary blobs for tensor data.
*   `archive/version`: Versioning information.

## Implementation Strategy

1.  **ZIP Unpacking**: Use the `zip` crate to access members of the archive.
2.  **Pickle Parsing**:
    *   Implement or use a Rust-based Pickle parser.
    *   PyTorch uses a custom Pickle `Unpickler` that handles "persistent IDs" to refer to tensor data stored outside the `data.pkl` file.
    *   We need to map these persistent IDs to the corresponding files in the ZIP (e.g., `data/0`).
3.  **Tensor Reconstruction**:
    *   Extract shape, dtype, and storage info from the unpickled objects.
    *   Load raw bytes from the ZIP members and convert them to `Tensor` objects in our IR.

## Challenges

*   **Custom Classes**: PyTorch serializes many custom Python classes. Our parser must be able to ignore or minimally represent these if they aren't critical for the static graph.
*   **Version Compatibility**: Support different versions of the PyTorch serialization format.
