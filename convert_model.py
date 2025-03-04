#!/usr/bin/env python
"""
Convert the Magenta Harmonizer model from TensorFlow SavedModel to ONNX format.

This script loads the TensorFlow SavedModel, inspects it to identify input/output
signatures, converts it to ONNX format, and optionally optimizes the resulting model.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import tensorflow as tf
import tf2onnx
import onnx
import onnxsim
from onnx import helper as onnx_helper

# Add local modules
from download_model import main as download_model


def inspect_model(model_dir: Path) -> Dict[str, Any]:
    """
    Load and inspect a TensorFlow SavedModel to identify its input/output signatures.
    
    Args:
        model_dir: Path to the TensorFlow SavedModel directory
        
    Returns:
        Dict containing model information
    """
    print(f"Loading model from {model_dir}")
    model = tf.saved_model.load(str(model_dir))
    
    # Get model signatures
    signatures = model.signatures
    
    # Get the default serving signature
    serving_signature = signatures.get("serving_default")
    if serving_signature is None:
        raise ValueError("Model doesn't have a serving_default signature")
    
    # Extract input and output specs
    input_specs = {name: tensor.shape for name, tensor in serving_signature.inputs.items()}
    output_specs = {name: tensor.shape for name, tensor in serving_signature.outputs.items()}
    
    # Print some info about the model
    print("Model signatures:")
    print(f"- Inputs: {input_specs}")
    print(f"- Outputs: {output_specs}")
    
    # Get operations to identify potential custom layers
    model_info = {
        "model": model,
        "input_specs": input_specs,
        "output_specs": output_specs,
        "signature": serving_signature,
    }
    
    return model_info


def convert_to_onnx(
    model_dir: Path, 
    output_path: Path, 
    model_info: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Convert a TensorFlow SavedModel to ONNX format using tf2onnx.
    
    Args:
        model_dir: Path to the TensorFlow SavedModel directory
        output_path: Path where the ONNX model will be saved
        model_info: Optional dictionary with model information (if already inspected)
        
    Returns:
        Path to the converted ONNX model
    """
    if model_info is None:
        model_info = inspect_model(model_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convert the model using tf2onnx
    print(f"Converting model to ONNX format: {output_path}")
    
    # We'll use the model_proto approach for more control
    input_signature = []
    for name, shape in model_info["input_specs"].items():
        # Create a TensorSpec for each input
        # Note: We use None for batch dimension to allow dynamic batching
        dynamic_shape = [None if dim is None or dim < 0 else dim for dim in shape.as_list()]
        input_signature.append(
            tf.TensorSpec(shape=dynamic_shape, dtype=tf.float32, name=name)
        )
    
    # Convert the model
    model_proto, _ = tf2onnx.convert.from_saved_model(
        str(model_dir),
        input_signature=input_signature,
        output_path=str(output_path),
    )
    
    print(f"Model converted and saved to {output_path}")
    return output_path


def optimize_onnx_model(input_path: Path, output_path: Path) -> Path:
    """
    Optimize an ONNX model using onnx-simplifier.
    
    Args:
        input_path: Path to the input ONNX model
        output_path: Path where the optimized model will be saved
        
    Returns:
        Path to the optimized ONNX model
    """
    print(f"Optimizing ONNX model: {input_path} -> {output_path}")
    
    # Load the ONNX model
    model = onnx.load(str(input_path))
    
    # Simplify the model
    simplified_model, check = onnxsim.simplify(model)
    
    if not check:
        print("WARNING: Simplified model may not be equivalent to the original model")
    
    # Save the optimized model
    onnx.save(simplified_model, str(output_path))
    
    print(f"Optimized model saved to {output_path}")
    return output_path


def check_for_custom_layers(model_dir: Path) -> List[str]:
    """
    Check for potential custom layers in the TensorFlow model.
    
    Args:
        model_dir: Path to the TensorFlow SavedModel directory
        
    Returns:
        List of potential custom operation names
    """
    # Load the model
    model = tf.saved_model.load(str(model_dir))
    
    # Get all operation types
    op_types = set()
    for func in model.signatures.values():
        for op in func.graph.as_graph_def().node:
            op_types.add(op.op)
    
    # Common TensorFlow operations that are typically handled by tf2onnx
    standard_ops = {
        'Const', 'Placeholder', 'Identity', 'Add', 'Mul', 'Sub', 'Div',
        'MatMul', 'Conv2D', 'MaxPool', 'BiasAdd', 'Relu', 'Softmax',
        'Reshape', 'Concat', 'Split', 'Slice', 'Transpose',
        'LSTM', 'GRU', 'BatchNormalization'
    }
    
    # Identify potential custom operations
    potential_custom_ops = op_types - standard_ops
    
    if potential_custom_ops:
        print("Potential custom operations detected:")
        for op in potential_custom_ops:
            print(f"- {op}")
    else:
        print("No potential custom operations detected.")
    
    return list(potential_custom_ops)


def main() -> int:
    """
    Main function to download and convert the Harmonizer model.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to ONNX")
    parser.add_argument(
        "--model-dir", 
        type=Path, 
        help="Path to the TensorFlow SavedModel directory (if already downloaded)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("./onnx_models"),
        help="Directory where the ONNX models will be saved"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true", 
        help="Optimize the ONNX model after conversion"
    )
    
    args = parser.parse_args()
    
    try:
        # Download the model if not provided
        model_dir = args.model_dir
        if model_dir is None:
            print("No model directory provided, downloading model...")
            model_dir = download_model()
            if model_dir is None:
                print("Failed to download model")
                return 1
        
        # Check for custom layers
        custom_layers = check_for_custom_layers(model_dir)
        if custom_layers:
            print("\nWARNING: Custom layers detected. Conversion might not work as expected.")
            print("If conversion fails, you may need to modify the model or implement custom operators.")
        
        # Inspect the model
        model_info = inspect_model(model_dir)
        
        # Convert to ONNX
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = output_dir / "harmonizer.onnx"
        convert_to_onnx(model_dir, output_path, model_info)
        
        # Optimize if requested
        if args.optimize:
            optimized_path = output_dir / "harmonizer_optimized.onnx"
            optimize_onnx_model(output_path, optimized_path)
            print(f"\nSuccessfully converted and optimized model: {optimized_path}")
        else:
            print(f"\nSuccessfully converted model: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during model conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())