# Optimizer Design

The Optimizer component improves the IR graph by applying various transformation passes. These optimizations can reduce the complexity of the graph and improve performance in the exported ONNX model.

## Optimization Passes

1.  **Constant Folding**:
    *   Identify operations where all inputs are constants.
    *   Pre-compute the result and replace the node with a constant tensor.
2.  **Dead Code Elimination (DCE)**:
    *   Remove nodes whose outputs are not used by any other node and are not model outputs.
3.  **Operator Fusion**:
    *   Combine multiple operators into a single, more efficient operator.
    *   Example: `Conv` + `BatchNormalization` -> `Conv` (with fused weights).
    *   Example: `Add` + `Relu` -> `FusedAddRelu` (if supported by target).
4.  **Shape Inference**:
    *   Propagate tensor shapes through the graph to allow further optimizations.

## Architecture

*   **Pass Trait**:
    ```rust
    trait OptimizationPass {
        fn apply(&self, ir: &mut ModelIR) -> Result<(), OptimizerError>;
    }
    ```
*   **Optimizer Pipeline**: A sequence of passes executed in order.
