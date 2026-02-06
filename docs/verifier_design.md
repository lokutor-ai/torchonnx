# Verifier Design

The Verifier ensures that the exported ONNX model produces numerical results that are consistent with the original PyTorch model.

## Verification Process

1.  **Input Generation**: Generate random input tensors matching the model's input shapes and types.
2.  **PyTorch Inference**:
    *   Load the original model weights.
    *   Execute inference using a reference implementation (e.g., `libtorch` or a Python script).
    *   Capture the output tensors.
3.  **ONNX Inference**:
    *   Load the exported `.onnx` file using `onnxruntime`.
    *   Execute inference with the same input tensors.
    *   Capture the output tensors.
4.  **Comparison**:
    *   Compare the output tensors from both runs.
    *   Use an epsilon tolerance for floating-point comparisons (e.g., L1/L2 norm or `allclose`).

## Components

*   **Parity Test Suite**: A set of automated tests that run the full verification process for various models and operators.
*   **Inference Engine Abstraction**: Interfaces for interacting with different inference runtimes.
