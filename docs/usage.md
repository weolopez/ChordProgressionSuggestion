# Usage Guide

This guide provides detailed instructions on how to use the Magenta Harmonizer model conversion tools.

## Quick Start

For a quick conversion of the Harmonizer model to ONNX format:

```bash
# Setup the environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the conversion
python main.py
```

This will download the model, convert it to ONNX, and validate the conversion.

## Detailed Usage

### 1. Download the Model

To download the Harmonizer model without proceeding with conversion:

```bash
python download_model.py
```

This will:
- Download the Harmonizer model from Magenta's storage
- Extract the model to `./model_data/extracted/`
- Verify the model structure

The model will be downloaded as a TensorFlow SavedModel format.

### 2. Handle Custom Layers

If the model contains custom TensorFlow operations that don't have direct ONNX equivalents:

```bash
python handle_custom_layers.py ./model_data/extracted --output-dir ./modified_model
```

This will:
- Analyze the model for custom operations
- Create a modified version with standard operations

Options:
- `--attention-only`: Only handle attention mechanisms
- `--output-dir`: Specify where to save the modified model

### 3. Convert to ONNX

To convert the TensorFlow model to ONNX format:

```bash
python convert_model.py --model-dir ./model_data/extracted --optimize
```

Options:
- `--model-dir`: Path to the SavedModel directory
- `--output-dir`: Where to save the ONNX model (default: ./onnx_models)
- `--optimize`: Apply ONNX optimization for better performance

### 4. Validate the ONNX Model

To validate the converted ONNX model:

```bash
python validate_model.py ./onnx_models/harmonizer_optimized.onnx
```

This will:
- Check the model structure
- Verify ONNX compatibility
- Run inference with dummy data to ensure functionality

Options:
- `--batch-size`: Specify batch size for testing (default: 1)

## Advanced Usage

### Customizing the Conversion Process

You can customize the full conversion pipeline:

```bash
python main.py \
  --model-dir /path/to/existing/model \
  --output-dir ./custom_output \
  --skip-download \
  --skip-optimization
```

All available options:
- `--model-dir`: Path to an existing TensorFlow SavedModel
- `--output-dir`: Directory for all output files
- `--skip-download`: Skip model download (requires --model-dir)
- `--skip-custom-layers`: Skip custom layer handling
- `--skip-optimization`: Skip ONNX model optimization
- `--skip-validation`: Skip final model validation

### Working with MLC

After converting to ONNX, you can use MLC to compile the model for WebGPU or WASM:

1. Install MLC:
   ```bash
   pip install mlc-ai
   ```

2. Compile the ONNX model:
   ```bash
   python -m mlc_ai.compile \
     --model ./onnx_models/harmonizer_optimized.onnx \
     --target webgpu \
     --output ./web_model/harmonizer_web.so
   ```

3. Use the compiled model in your web application as per MLC documentation.

## Troubleshooting

### Common Issues

1. **Model download fails**
   - Check your internet connection
   - The URL might have changed; update `MAGENTA_MODEL_URL` in `download_model.py`

2. **Custom layer handling fails**
   - Inspect the model with `tf.summary.create_file_writer` to understand custom operations
   - Implement specific handlers in `handle_custom_layers.py` for your model

3. **ONNX conversion errors**
   - Check if your TensorFlow version is compatible with tf2onnx
   - Some operations might not be supported by ONNX; consider model modification

4. **MLC compilation issues**
   - Ensure the ONNX model only uses operations supported by MLC
   - Check MLC documentation for supported ONNX versions and operations

### Getting Help

If you encounter issues not covered here, please:
1. Check the error messages for specific details
2. Look at the TensorFlow and ONNX documentation
3. Open an issue in the GitHub repository with details of your problem