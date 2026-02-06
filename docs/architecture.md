# Architecture Overview

`torchonnx` is designed to be a high-performance, reliable bridge between PyTorch models and the ONNX format. The architecture is modularized to ensure extensibility and ease of testing.

## Core Components

1.  **Loader (Input Layer)**: Responsible for reading `.pt` (TorchScript/Pickle) and `.safetensors` files.
2.  **IR (Intermediate Representation)**: A neutral graph representation of the model that facilitates optimizations and mapping between formats.
3.  **Optimizer**: Performs graph-level optimizations (constant folding, operator fusion, etc.) on the IR.
4.  **Exporter (Output Layer)**: Translates the IR into ONNX protobuf format.
5.  **Verifier**: Executes inference using both the original PyTorch weights (via a runtime or FFI) and the exported ONNX model to ensure numerical parity.

## Data Flow

`Source File (.pt/.safetensors)` -> `Loader` -> `IR` -> `Optimizer` -> `Exporter` -> `ONNX File`
                                         |
                                         v
                                     `Verifier` (Cross-check with original)
