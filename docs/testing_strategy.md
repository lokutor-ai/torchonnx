# Testing & Verification Strategy

## Unit Testing
Each component (Loader, IR, Optimizer, Exporter) will have dedicated unit tests. We will use standard Rust `#[test]` modules.

## Integration Testing
Integration tests will cover the end-to-end flow from loading a file to generating an ONNX output.

## Numerical Verification (The Gold Standard)
For every supported operator and complex model:
1.  Load weights and architecture.
2.  Generate random input tensors.
3.  Run inference using a reference implementation (e.g., `libtorch` or a minimal PyTorch runtime).
4.  Run inference on the generated ONNX model (using `onnxruntime`).
5.  Compare outputs using an epsilon-based tolerance (L1/L2 norm).

## Test Data
A `tests/data` directory will be used to store minimal `.pt` and `.safetensors` files generated from PyTorch for regression testing.
