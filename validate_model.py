#!/usr/bin/env python
"""
Validate a converted ONNX model by checking its structure and running inference.

This script loads an ONNX model, validates its structure, and runs inference
with dummy input data to verify functionality.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import onnx
import onnxruntime as ort


def check_model_structure(model_path: Path) -> Dict[str, Any]:
    """
    Check the structure of an ONNX model and extract its metadata.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        Dict containing model metadata
    """
    print(f"Checking model structure: {model_path}")
    
    # Load the ONNX model
    model = onnx.load(str(model_path))
    
    # Check the model's validity
    onnx.checker.check_model(model)
    print("✓ Model structure is valid")
    
    # Extract model metadata
    graph = model.graph
    
    # Get input information
    inputs = []
    for input_proto in graph.input:
        name = input_proto.name
        shape = []
        for dim in input_proto.type.tensor_type.shape.dim:
            if dim.dim_param:
                # Symbolic dimension (like 'batch_size')
                shape.append(dim.dim_param)
            else:
                # Numeric dimension
                shape.append(dim.dim_value)
        inputs.append({"name": name, "shape": shape})
    
    # Get output information
    outputs = []
    for output_proto in graph.output:
        name = output_proto.name
        shape = []
        for dim in output_proto.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        outputs.append({"name": name, "shape": shape})
    
    # Print model info
    print("\nModel Information:")
    print(f"- Inputs: {inputs}")
    print(f"- Outputs: {outputs}")
    
    # Get operator types
    op_types = {node.op_type for node in graph.node}
    print(f"- Operators: {sorted(op_types)}")
    
    metadata = {
        "inputs": inputs,
        "outputs": outputs,
        "op_types": op_types
    }
    
    return metadata


def create_dummy_input(
    input_info: List[Dict[str, Any]], 
    batch_size: int = 1
) -> Dict[str, np.ndarray]:
    """
    Create dummy input data based on the model's input information.
    
    Args:
        input_info: List of input metadata dictionaries
        batch_size: Batch size for dynamic batch dimensions
        
    Returns:
        Dict mapping input names to dummy numpy arrays
    """
    dummy_inputs = {}
    
    for input_meta in input_info:
        name = input_meta["name"]
        shape = input_meta["shape"]
        
        # Create a concrete shape, replacing dynamic dimensions
        concrete_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim == 0:
                # Dynamic dimension (symbolic name or 0 dimension)
                concrete_shape.append(batch_size)
            else:
                concrete_shape.append(dim)
        
        # Create random dummy data
        dummy_data = np.random.randn(*concrete_shape).astype(np.float32)
        dummy_inputs[name] = dummy_data
    
    return dummy_inputs


def run_inference(
    model_path: Path, 
    dummy_inputs: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Run inference on the ONNX model with dummy input data.
    
    Args:
        model_path: Path to the ONNX model
        dummy_inputs: Dict mapping input names to dummy input arrays
        
    Returns:
        Dict mapping output names to output arrays
    """
    print("\nRunning inference with dummy data")
    
    # Create an ONNX Runtime session
    session = ort.InferenceSession(str(model_path))
    
    # Get output names
    output_names = [output.name for output in session.get_outputs()]
    
    # Print input shapes for debugging
    for name, data in dummy_inputs.items():
        print(f"- Input '{name}' shape: {data.shape}")
    
    # Run inference
    outputs = session.run(output_names, dummy_inputs)
    
    # Create a dictionary mapping output names to values
    output_dict = {name: value for name, value in zip(output_names, outputs)}
    
    # Print output shapes
    for name, data in output_dict.items():
        print(f"- Output '{name}' shape: {data.shape}")
    
    return output_dict


def main() -> int:
    """
    Main function to validate an ONNX model.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Validate an ONNX model")
    parser.add_argument(
        "model_path", 
        type=Path, 
        help="Path to the ONNX model to validate"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size to use for inference"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if the model file exists
        if not args.model_path.exists():
            print(f"Error: Model file not found: {args.model_path}")
            return 1
        
        # Check model structure and get metadata
        metadata = check_model_structure(args.model_path)
        
        # Create dummy input data
        dummy_inputs = create_dummy_input(metadata["inputs"], args.batch_size)
        
        # Run inference
        outputs = run_inference(args.model_path, dummy_inputs)
        
        print("\n✓ Model validation successful!")
        return 0
        
    except Exception as e:
        print(f"Error during model validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())