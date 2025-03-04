#!/usr/bin/env python
"""
Convert TensorFlow SavedModel to ONNX format.

This script loads a TensorFlow SavedModel and converts it to ONNX format
for use with web-based inference.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple


def convert_to_onnx(
    saved_model_dir: Path,
    output_path: Path,
    optimize: bool = True
) -> Path:
    """
    Convert a TensorFlow SavedModel to ONNX format using the tf2onnx command line.
    
    Args:
        saved_model_dir: Path to the TensorFlow SavedModel directory
        output_path: Path where the ONNX model will be saved
        optimize: Whether to optimize the ONNX model after conversion
        
    Returns:
        Path to the final ONNX model
    """
    print(f"Converting SavedModel at {saved_model_dir} to ONNX")
    
    # Make sure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Use the tf2onnx command line tool
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", str(saved_model_dir),
        "--output", str(output_path),
        "--opset", "15"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during conversion: {result.stderr}")
        raise RuntimeError(f"tf2onnx conversion failed with exit code {result.returncode}")
    
    print(result.stdout)
    print(f"Model converted and saved to {output_path}")
    
    # Optimize the model if requested
    if optimize:
        optimized_path = output_path.parent / f"{output_path.stem}_optimized{output_path.suffix}"
        optimize_onnx_model(output_path, optimized_path)
        return optimized_path
    
    return output_path


def optimize_onnx_model(input_path: Path, output_path: Path) -> Path:
    """
    Optimize an ONNX model to improve performance.
    
    Args:
        input_path: Path to the input ONNX model
        output_path: Path where the optimized model will be saved
        
    Returns:
        Path to the optimized ONNX model
    """
    print(f"Optimizing ONNX model: {input_path} -> {output_path}")
    
    try:
        # Try to use onnxsim command-line tool
        cmd = [
            "python", "-m", "onnxsim",
            str(input_path),
            str(output_path)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: ONNX simplification failed: {result.stderr}")
            print("Using original model")
            import shutil
            shutil.copy(input_path, output_path)
        else:
            print(result.stdout)
            print(f"Model optimized successfully")
    except Exception as e:
        print(f"Warning: Error during optimization: {e}")
        print("Using original model")
        import shutil
        shutil.copy(input_path, output_path)
    
    print(f"Final model saved to {output_path}")
    return output_path


def validate_onnx_model(model_path: Path) -> bool:
    """
    Validate an ONNX model using onnxruntime.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        True if validation is successful
    """
    print(f"Validating ONNX model: {model_path}")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check the model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        print("ONNX model structure is valid")
        
        # Create an inference session
        session = ort.InferenceSession(str(model_path))
        
        # Get input details
        input_details = []
        for input_meta in session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            input_details.append({"name": name, "shape": shape})
        
        print("Input details:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}: name={detail['name']}, shape={detail['shape']}")
        
        # For the first input (melody), create a random input with appropriate shape
        # Replace any None or 0 dimensions with reasonable values
        if len(input_details) > 0:
            input_shape = list(input_details[0]["shape"])
            # Replace None or 0 with default values
            input_shape = [1 if dim is None or dim == 0 else dim for dim in input_shape]
            
            # Create a random input (use int32 for melody input)
            input_data = {
                input_details[0]["name"]: np.random.randint(0, 128, input_shape).astype(np.int32)
            }
            
            # Run inference
            print(f"Running inference with random input of shape {input_shape}")
            outputs = session.run(None, input_data)
            
            # Print output details
            print("Inference successful!")
            print("Output details:")
            for i, output in enumerate(outputs):
                print(f"  Output {i} shape: {output.shape}")
                
            return True
        else:
            print("No inputs found in the model")
            return False
        
    except Exception as e:
        print(f"Error validating ONNX model: {e}")
        return False


def main() -> int:
    """
    Main function to convert a TensorFlow SavedModel to ONNX.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to ONNX")
    parser.add_argument(
        "--saved-model-dir", 
        type=Path, 
        default=Path("./model_data/saved_model"),
        help="Path to the TensorFlow SavedModel directory"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("./model_data/model.onnx"),
        help="Path where the ONNX model will be saved"
    )
    parser.add_argument(
        "--no-optimize", 
        action="store_true", 
        help="Skip ONNX optimization"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate the ONNX model after conversion"
    )
    
    args = parser.parse_args()
    
    try:
        if not args.saved_model_dir.exists():
            print(f"Error: SavedModel directory not found: {args.saved_model_dir}")
            return 1
        
        # Convert to ONNX
        final_model_path = convert_to_onnx(
            args.saved_model_dir,
            args.output,
            optimize=not args.no_optimize
        )
        
        # Validate if requested
        if args.validate:
            validate_onnx_model(final_model_path)
        
        print(f"\nConversion complete! Final model saved to: {final_model_path}")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())