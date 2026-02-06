# Exporter Design

The Exporter translates the optimized IR into a valid ONNX protobuf file.

## Components

1.  **ONNX Proto Generator**: Uses `prost` to generate Rust structs from `onnx.proto`.
2.  **Node Mapper**: A registry of functions that map IR nodes to ONNX operators.
3.  **Weight Serializer**: Handles the conversion of IR tensors to ONNX `TensorProto`.
4.  **Graph Assembler**: Constructs the final `ModelProto` including the graph, initializers, and metadata.

## Mapping Logic

For each IR node:
*   Identify the corresponding ONNX operator (e.g., `torch.nn.Conv2d` -> `Conv`).
*   Map attributes (e.g., `padding` in PyTorch might need adjustment for ONNX `pads`).
*   Validate that all inputs and outputs are correctly wired.

## Supported Operators (Initial Set)

*   `Gemm` (Linear)
*   `Conv`
*   `Relu`, `Sigmoid`, `Tanh`
*   `Add`, `Sub`, `Mul`, `Div`
*   `Reshape`, `Transpose`
*   `BatchNormalization`
