#!/usr/bin/env python
"""
Main script to orchestrate the entire Magenta Harmonizer model conversion process.

This script coordinates downloading the model, handling custom layers,
converting to ONNX, and validating the result.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add local modules
from download_model import main as download_model
from handle_custom_layers import create_modified_model
from convert_model import main as convert_model
from validate_model import main as validate_model


def main() -> int:
    """
    Main function to orchestrate the Harmonizer model conversion process.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Convert Magenta Harmonizer model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir", 
        type=Path, 
        help="Path to the TensorFlow SavedModel directory (if already downloaded)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("./output"),
        help="Directory where all output files will be saved"
    )
    parser.add_argument(
        "--skip-download", 
        action="store_true", 
        help="Skip model download (use existing model_dir)"
    )
    parser.add_argument(
        "--skip-custom-layers", 
        action="store_true", 
        help="Skip custom layer handling"
    )
    parser.add_argument(
        "--skip-optimization", 
        action="store_true", 
        help="Skip ONNX model optimization"
    )
    parser.add_argument(
        "--skip-validation", 
        action="store_true", 
        help="Skip ONNX model validation"
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Download the model (if needed)
        model_dir = args.model_dir
        if model_dir is None and not args.skip_download:
            print("\n=== Step 1: Downloading Harmonizer model ===")
            model_dir = download_model()
            if model_dir is None:
                print("Failed to download model")
                return 1
        elif args.skip_download and model_dir is None:
            print("Error: --skip-download specified but no --model-dir provided")
            return 1
        elif args.skip_download:
            print("\n=== Step 1: Using existing model ===")
            print(f"Model directory: {model_dir}")
        
        # Step 2: Handle custom layers (if needed)
        modified_model_dir = model_dir
        if not args.skip_custom_layers:
            print("\n=== Step 2: Handling custom layers ===")
            custom_layers_dir = args.output_dir / "model_modified"
            modified_model_dir, _ = create_modified_model(model_dir, custom_layers_dir)
        else:
            print("\n=== Step 2: Skipping custom layer handling ===")
        
        # Step 3: Convert to ONNX
        print("\n=== Step 3: Converting model to ONNX ===")
        # Prepare arguments for convert_model.py
        convert_args = [
            "convert_model.py",
            "--model-dir", str(modified_model_dir),
            "--output-dir", str(args.output_dir / "onnx_models")
        ]
        if not args.skip_optimization:
            convert_args.append("--optimize")
        
        # Call convert_model.py
        sys.argv = convert_args
        convert_result = convert_model()
        if convert_result != 0:
            print("Failed to convert model to ONNX")
            return 1
        
        # Step 4: Validate the ONNX model
        if not args.skip_validation:
            print("\n=== Step 4: Validating ONNX model ===")
            # Find the ONNX model file
            onnx_file = args.output_dir / "onnx_models"
            if not args.skip_optimization:
                onnx_file = onnx_file / "harmonizer_optimized.onnx"
            else:
                onnx_file = onnx_file / "harmonizer.onnx"
            
            # Prepare arguments for validate_model.py
            validate_args = [
                "validate_model.py",
                str(onnx_file)
            ]
            
            # Call validate_model.py
            sys.argv = validate_args
            validate_result = validate_model()
            if validate_result != 0:
                print("Failed to validate ONNX model")
                return 1
        else:
            print("\n=== Step 4: Skipping ONNX model validation ===")
        
        print("\n=== Conversion complete! ===")
        print(f"ONNX model saved to: {args.output_dir / 'onnx_models'}")
        return 0
        
    except Exception as e:
        print(f"Error during model conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())