# Handling Custom Layers

When converting TensorFlow models to ONNX, you may encounter custom operations that don't have direct ONNX equivalents. This document explains how to identify and handle such custom layers in the Magenta Harmonizer model.

## Common Custom Operations in Harmonizer

The Harmonizer model may contain these custom operations:

1. **Custom Attention Mechanisms**: Magenta's harmonizer likely uses custom attention mechanisms for aligning melody and harmony.

2. **Music-Specific Operations**: There might be operations specifically designed for music processing or chord analysis.

3. **Magenta-Specific Layers**: The Magenta framework may have implemented custom TensorFlow operations.

## Identifying Custom Layers

Our conversion tool automatically identifies potential custom operations:

```bash
python handle_custom_layers.py ./model_data/extracted
```

The output will list operations that might not have direct ONNX equivalents.

## Handling Strategies

### 1. Model Modification

The primary strategy is to modify the model to replace custom operations with equivalent standard operations:

```python
# Example: Replace custom attention with standard attention
custom_attention = model.get_layer('custom_attention')
standard_attention = tf.keras.layers.Attention()(
    [custom_attention.input_1, custom_attention.input_2]
)
```

Our `handle_custom_layers.py` script attempts to:
- Identify custom layers
- Create equivalent implementations using standard TensorFlow operations
- Save a modified model that's easier to convert to ONNX

### 2. Custom ONNX Operators

If modification isn't feasible, you can implement custom ONNX operators:

1. **Define Custom Operator**:
   ```python
   from onnx import helper as onnx_helper
   
   # Define a custom operator schema
   custom_op = onnx_helper.make_opsetid("CustomDomain", 1)
   
   # Create a node with the custom operator
   node = onnx_helper.make_node(
       "CustomAttention",
       inputs=["input1", "input2"],
       outputs=["output"],
       domain="CustomDomain"
   )
   ```

2. **Register it with ONNX Runtime**:
   ```python
   # Implement the custom op in a backend
   class CustomAttentionOp:
       @staticmethod
       def run(node, inputs):
           # Implement the operation
           return [output]
   
   # Register the op
   onnx_runtime.register_custom_op("CustomDomain", "CustomAttention", CustomAttentionOp)
   ```

### 3. Alternative Path Through TensorFlow.js

If direct conversion fails, you can try an intermediate conversion through TensorFlow.js:

```bash
# Install tensorflowjs
pip install tensorflowjs

# Convert to TensorFlow.js format
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./saved_model_dir ./tfjs_model

# Then convert TensorFlow.js to ONNX (requires additional tools)
```

## Music-Specific Considerations

For music models like Harmonizer, special attention is needed for:

### Sequence Length Handling

Ensure that the model correctly handles variable-length sequences:

```python
# Adjust input shapes to be dynamic
spec = (tf.TensorSpec(shape=[None, None, input_features], 
                       dtype=tf.float32, name="input"),)
```

### Note Encoding

The harmonizer likely uses specific note encoding that should be preserved. Check if these are part of the model or preprocessing/postprocessing steps.

## Advanced Debugging

For complex issues:

1. **Visualize the TensorFlow graph**:
   ```python
   import tensorflow as tf
   
   writer = tf.summary.create_file_writer('./logs')
   with writer.as_default():
       tf.summary.graph(model.signatures["serving_default"].graph)
   ```

2. **Extract subgraphs** for testing:
   ```python
   # Extract a problematic subgraph
   inputs = model.signatures["serving_default"].inputs
   intermediate = model.signatures["serving_default"].get_concrete_function()(inputs)[intermediate_tensor]
   submodel = tf.keras.Model(inputs=inputs, outputs=intermediate)
   ```

## Future Improvements

As TensorFlow and ONNX ecosystems evolve:

1. Check for updates to the `tf2onnx` package that might add support for more operations.

2. Consider updating the model to use newer TensorFlow operations that have better ONNX support.

3. Monitor the MLC project for expanded ONNX operation support.

By understanding these techniques, you can address most custom layer issues encountered during model conversion.