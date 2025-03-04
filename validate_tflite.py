#!/usr/bin/env python
"""
Validate the TensorFlow Lite model by running inference.

This script loads a TensorFlow Lite model and runs inference with sample inputs
to verify that the conversion was successful.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple


def validate_tflite_model(model_path: Path) -> bool:
    """
    Validate a TensorFlow Lite model by performing inference.
    
    Args:
        model_path: Path to the TFLite model
        
    Returns:
        True if validation is successful
    """
    print(f"Validating TFLite model: {model_path}")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model has been loaded successfully")
    print(f"Number of input tensors: {len(input_details)}")
    print(f"Number of output tensors: {len(output_details)}")
    
    print("\nInput details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype']}")
    
    print("\nOutput details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype']}")
    
    # Create a sample input for the main input tensor (melody)
    melody_input = None
    for detail in input_details:
        if "melody" in detail['name'].lower() or detail['index'] == 0:
            melody_input = detail
            break
    
    if melody_input is not None:
        input_shape = melody_input['shape']
        # Use the exact shape as expected by the model
        test_shape = input_shape.copy()
        
        # Generate random melody data (MIDI values 0-127)
        print(f"\nGenerating random melody input with shape {test_shape}...")
        input_data = np.random.randint(0, 128, test_shape).astype(np.int32)
        
        # Set inputs for all tensors
        for detail in input_details:
            if detail['index'] == melody_input['index']:
                # Set the melody input tensor
                interpreter.set_tensor(detail['index'], input_data)
                print(f"Set input tensor {detail['name']} with data shape {input_data.shape}")
            else:
                # For other inputs (often resource tensors), use zeros
                shape = detail['shape']
                dtype = detail['dtype']
                if np.prod(shape) > 0:  # Only set if shape has elements
                    tensor_data = np.zeros(shape, dtype=np.dtype(dtype))
                    interpreter.set_tensor(detail['index'], tensor_data)
                    print(f"Set input tensor {detail['name']} with zeros of shape {shape}")
        
        try:
            # Run inference
            print("\nRunning inference...")
            interpreter.invoke()
            
            # Get the output
            print("\nInference completed successfully!")
            print("Output details:")
            for i, detail in enumerate(output_details):
                output_data = interpreter.get_tensor(detail['index'])
                print(f"  Output {i}: shape={output_data.shape}, dtype={output_data.dtype}")
                
                # Print sample of output data
                print("  Sample output values:")
                if len(output_data.shape) == 3:  # [batch, seq_len, num_chords]
                    # For 3D output, get the chord probabilities for the first time step
                    sample = output_data[0, 0, :10]  # First batch, first time step, first 10 chord probs
                    print(f"  First 10 chord probabilities at first time step: {sample}")
                else:
                    # For other shapes, just print first few values
                    flat = output_data.flatten()
                    if len(flat) > 0:
                        print(f"  First few values: {flat[:10]}")
            
            return True
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Could not identify the melody input tensor")
        return False


def main() -> int:
    """
    Main function to validate a TensorFlow Lite model.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Validate TensorFlow Lite model")
    parser.add_argument(
        "model_path", 
        type=Path,
        help="Path to the TensorFlow Lite model"
    )
    
    args = parser.parse_args()
    
    try:
        if not args.model_path.exists():
            print(f"Error: Model file not found: {args.model_path}")
            return 1
        
        # Validate the model
        success = validate_tflite_model(args.model_path)
        
        if success:
            print("\nModel validation was successful!")
            return 0
        else:
            print("\nModel validation failed")
            return 1
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())