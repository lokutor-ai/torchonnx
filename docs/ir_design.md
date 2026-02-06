# IR (Intermediate Representation) Design

The IR is the backbone of `torchonnx`. It must be flexible enough to represent PyTorch's dynamic nature and structured enough for ONNX's static graph requirements.

## Structure

*   **Graph**: The top-level container. Holds nodes, edges, and metadata.
*   **Node**: Represents an operator (e.g., `Conv2d`, `Add`, `Relu`).
    *   Attributes (e.g., `kernel_size`, `stride`).
    *   Inputs/Outputs (References to Tensors).
*   **Tensor**: Metadata (shape, data type) and optional data (for constants/weights).
*   **Value**: A symbolic representation of a tensor during execution (edges between nodes).

## Key Features

*   **Type Inference**: Ability to propagate shapes and types through the graph.
*   **Extensibility**: Easy to add new custom operators.
*   **Serialization**: Ability to dump the IR to a human-readable format (JSON/Protobuf) for debugging.

## Optimization Passes

The IR will support passes like:
*   Dead code elimination.
*   Constant folding.
*   Operator fusion (e.g., Conv + BN + ReLU).
