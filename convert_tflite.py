#!/usr/bin/env python
"""
Convert TensorFlow SavedModel to TensorFlow Lite format.

This script loads a TensorFlow SavedModel and converts it to TensorFlow Lite format
for web deployment using TensorFlow.js.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple


def convert_to_tflite(
    saved_model_dir: Path,
    output_path: Path,
    optimize: bool = True
) -> Path:
    """
    Convert a TensorFlow SavedModel to TensorFlow Lite format.
    
    Args:
        saved_model_dir: Path to the TensorFlow SavedModel directory
        output_path: Path where the TFLite model will be saved
        optimize: Whether to optimize the TFLite model after conversion
        
    Returns:
        Path to the final TFLite model
    """
    print(f"Converting SavedModel at {saved_model_dir} to TensorFlow Lite")
    
    # Make sure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Load the model
    model = tf.saved_model.load(str(saved_model_dir))
    
    # Get the concrete function
    concrete_func = model.signatures["serve"]
    
    # Create a converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Set conversion options
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Enable experimental features for model compatibility
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert the model
    print("Converting the model to TFLite format...")
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_path}")
    return output_path


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
    
    print("Input details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype']}")
    
    print("Output details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype']}")
    
    # Create a sample input
    if len(input_details) > 0:
        input_shape = input_details[0]['shape']
        # Replace any zeros with reasonable values for testing
        input_shape = [1 if dim == 0 else dim for dim in input_shape]
        
        input_type = input_details[0]['dtype']
        if input_type == np.float32:
            input_data = np.random.random(input_shape).astype(np.float32)
        else:
            # Assume int32 for melody input (MIDI values 0-127)
            input_data = np.random.randint(0, 128, input_shape).astype(np.int32)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        print(f"Running inference with random input of shape {input_shape}...")
        interpreter.invoke()
        
        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Inference successful! Output shape: {output_data.shape}")
        
        return True
    else:
        print("No inputs found in the model")
        return False


def create_web_files(tflite_path: Path, output_dir: Path) -> None:
    """
    Create web files for using the TensorFlow Lite model with TensorFlow.js.
    
    Args:
        tflite_path: Path to the TFLite model
        output_dir: Directory where web files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the TFLite model to the web directory
    import shutil
    web_model_path = output_dir / tflite_path.name
    shutil.copy(tflite_path, web_model_path)
    
    # Create a sample HTML file
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Chord Progression Suggestion</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/tf-tflite.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        textarea {{
            width: 100%;
            height: 100px;
        }}
        button {{
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }}
        .result {{
            margin-top: 20px;
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Chord Progression Suggestion</h1>
        <p>Enter melody notes (MIDI numbers) separated by commas:</p>
        <textarea id="melodyInput">60, 62, 64, 65, 67, 69, 71, 72</textarea>
        <button id="generateButton">Generate Chord Progression</button>
        <div class="result" id="result">Chord progressions will appear here</div>
    </div>

    <script>
        // Load the TFLite model
        const modelPath = '{web_model_path.name}';
        let tfliteModel;

        async function loadModel() {{
            try {{
                tfliteModel = await tflite.loadTFLiteModel(modelPath);
                console.log('Model loaded successfully');
                document.getElementById('generateButton').disabled = false;
            }} catch (error) {{
                console.error('Error loading model:', error);
                document.getElementById('result').textContent = 'Error loading model: ' + error.message;
            }}
        }}

        async function generateChords() {{
            if (!tfliteModel) {{
                document.getElementById('result').textContent = 'Model not loaded yet';
                return;
            }}

            try {{
                // Get melody input
                const melodyText = document.getElementById('melodyInput').value;
                const melodyNotes = melodyText.split(',').map(note => parseInt(note.trim()));
                
                // Create input tensor
                const input = tf.tensor2d([melodyNotes], [1, melodyNotes.length], 'int32');
                
                // Run inference
                const output = await tfliteModel.predict(input);
                
                // Process the output
                const outputData = await output.array();
                
                // Format and display the result
                const result = document.getElementById('result');
                result.innerHTML = '<h3>Suggested Chords:</h3>';
                result.innerHTML += '<pre>' + JSON.stringify(outputData, null, 2) + '</pre>';
                
                // Clean up tensors
                input.dispose();
                output.dispose();
            }} catch (error) {{
                console.error('Error generating chords:', error);
                document.getElementById('result').textContent = 'Error generating chords: ' + error.message;
            }}
        }}

        // Add event listeners
        document.addEventListener('DOMContentLoaded', () => {{
            document.getElementById('generateButton').disabled = true;
            loadModel();
            document.getElementById('generateButton').addEventListener('click', generateChords);
        }});
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"Web files created in {output_dir}")
    print(f"- TFLite model: {web_model_path}")
    print(f"- HTML file: {html_path}")


def main() -> int:
    """
    Main function to convert a TensorFlow SavedModel to TensorFlow Lite.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to TensorFlow Lite")
    parser.add_argument(
        "--saved-model-dir", 
        type=Path, 
        default=Path("./model_data/saved_model"),
        help="Path to the TensorFlow SavedModel directory"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("./model_data/model.tflite"),
        help="Path where the TFLite model will be saved"
    )
    parser.add_argument(
        "--no-optimize", 
        action="store_true", 
        help="Skip TFLite optimization"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate the TFLite model after conversion"
    )
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Create web files for using the model with TensorFlow.js"
    )
    parser.add_argument(
        "--web-dir", 
        type=Path, 
        default=Path("./web"),
        help="Directory where web files will be saved"
    )
    
    args = parser.parse_args()
    
    try:
        if not args.saved_model_dir.exists():
            print(f"Error: SavedModel directory not found: {args.saved_model_dir}")
            return 1
        
        # Convert to TFLite
        tflite_path = convert_to_tflite(
            args.saved_model_dir,
            args.output,
            optimize=not args.no_optimize
        )
        
        # Validate if requested
        if args.validate:
            validate_tflite_model(tflite_path)
        
        # Create web files if requested
        if args.web:
            create_web_files(tflite_path, args.web_dir)
        
        print(f"\nConversion complete! Final model saved to: {tflite_path}")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())