#!/usr/bin/env python
"""
Prepare models and web files for deployment.

This script prepares all necessary files for deploying the chord progression
suggestion model on the web.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import tensorflow as tf


def create_model_metadata(
    tflite_model_path: Path,
    chord_mapping_path: Path,
    output_dir: Path
) -> Path:
    """
    Create metadata JSON file for the model.
    
    Args:
        tflite_model_path: Path to the TFLite model
        chord_mapping_path: Path to the chord mapping JSON file
        output_dir: Directory where metadata JSON will be saved
        
    Returns:
        Path to the metadata JSON file
    """
    print(f"Creating model metadata from {tflite_model_path} and {chord_mapping_path}")
    
    # Load the TFLite model to get input/output info
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create input metadata
    input_metadata = []
    for detail in input_details:
        input_metadata.append({
            "name": detail["name"],
            "shape": detail["shape"].tolist(),
            "dtype": str(detail["dtype"]),
            "description": "Melody input as MIDI note numbers"
        })
    
    # Create output metadata
    output_metadata = []
    for detail in output_details:
        output_metadata.append({
            "name": detail["name"],
            "shape": detail["shape"].tolist(),
            "dtype": str(detail["dtype"]),
            "description": "Chord probabilities for each time step"
        })
    
    # Load chord mapping
    with open(chord_mapping_path, "r") as f:
        chord_mapping = json.load(f)
    
    # Create metadata
    metadata = {
        "model_name": "Melody to Chord Progression Model",
        "description": "A model that suggests chord progressions for melodies",
        "version": "1.0.0",
        "input": input_metadata,
        "output": output_metadata,
        "chord_mapping": chord_mapping,
        "num_chords": len(chord_mapping)
    }
    
    # Save metadata
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    return metadata_path


def create_web_deployment(
    tflite_model_path: Path,
    chord_mapping_path: Path,
    output_dir: Path
) -> None:
    """
    Create web deployment files.
    
    Args:
        tflite_model_path: Path to the TFLite model
        chord_mapping_path: Path to the chord mapping JSON file
        output_dir: Directory where web files will be saved
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the TFLite model
    model_filename = tflite_model_path.name
    shutil.copy(tflite_model_path, output_dir / model_filename)
    
    # Create model metadata
    metadata_path = create_model_metadata(tflite_model_path, chord_mapping_path, output_dir)
    
    # Create index.html
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chord Progression Suggestion</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.19.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js"></script>
    <style>
        :root {{
            --primary-color: #4a90e2;
            --secondary-color: #f0f7ff;
            --accent-color: #5cb85c;
            --text-color: #333;
            --light-gray: #f5f5f5;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            color: var(--text-color);
            background-color: var(--secondary-color);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        
        .app-description {{
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .main-content {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .input-section, .output-section {{
            flex: 1;
            min-width: 300px;
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .piano-container {{
            margin: 15px 0;
            overflow-x: auto;
            white-space: nowrap;
        }}
        
        .piano {{
            display: inline-flex;
            position: relative;
            height: 120px;
        }}
        
        .white-key, .black-key {{
            box-sizing: border-box;
        }}
        
        .white-key {{
            width: 40px;
            height: 120px;
            background-color: white;
            border: 1px solid #ccc;
            z-index: 1;
        }}
        
        .black-key {{
            position: absolute;
            width: 30px;
            height: 80px;
            background-color: black;
            z-index: 2;
        }}
        
        .white-key.selected {{
            background-color: var(--primary-color);
        }}
        
        .black-key.selected {{
            background-color: var(--accent-color);
        }}
        
        .melody-display {{
            margin: 15px 0;
            border: 1px solid #ddd;
            padding: 10px;
            min-height: 50px;
            border-radius: 5px;
            background-color: var(--light-gray);
        }}
        
        .note-pill {{
            display: inline-block;
            background-color: var(--primary-color);
            color: white;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 20px;
            font-size: 0.9rem;
        }}
        
        button {{
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            margin-right: 10px;
        }}
        
        button:hover {{
            background-color: #4a9d4a;
        }}
        
        button:disabled {{
            background-color: #cccccc;
            cursor: not-allowed;
        }}
        
        .chord-display {{
            margin-top: 15px;
        }}
        
        .chord {{
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: var(--light-gray);
        }}
        
        .chord-name {{
            font-weight: bold;
            font-size: 1.2rem;
            color: var(--primary-color);
        }}
        
        .chord-confidence {{
            font-size: 0.9rem;
            color: #666;
        }}
        
        .status {{
            margin-top: 20px;
            color: #666;
            font-style: italic;
        }}
        
        .model-info {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9rem;
            color: #666;
        }}
        
        @media (max-width: 768px) {{
            .main-content {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Chord Progression Suggestion</h1>
    </header>
    
    <div class="container">
        <div class="app-description">
            <h2>Turn Your Melody into Beautiful Chord Progressions</h2>
            <p>
                Use this tool to generate chord progressions that complement your melody. 
                Simply input your melody using the piano keyboard or by entering MIDI note numbers, 
                then click "Generate Chords" to see chord suggestions.
            </p>
        </div>
        
        <div class="main-content">
            <section class="input-section">
                <h2>Enter Your Melody</h2>
                
                <div class="piano-container">
                    <div id="piano" class="piano">
                        <!-- Piano keys will be generated by JavaScript -->
                    </div>
                </div>
                
                <h3>Your Melody</h3>
                <div class="melody-display" id="melodyDisplay">
                    <!-- Selected notes will appear here -->
                </div>
                
                <div class="controls">
                    <button id="generateButton" disabled>Generate Chords</button>
                    <button id="clearButton">Clear Melody</button>
                </div>
                
                <div class="status">
                    <p id="modelStatus">Loading model...</p>
                </div>
            </section>
            
            <section class="output-section">
                <h2>Suggested Chord Progression</h2>
                <div class="chord-display" id="chordDisplay">
                    <p>Suggested chords will appear here after you generate them.</p>
                </div>
            </section>
        </div>
        
        <div class="model-info">
            <p>This tool uses a deep learning model converted from Magenta's Hooktheory melody to chord progression model.</p>
            <p>Model version: 1.0.0 | Last updated: March 2025</p>
        </div>
    </div>

    <script>
        // Model path
        const MODEL_PATH = '{model_filename}';
        const METADATA_PATH = 'model_metadata.json';
        
        // State variables
        let tfliteModel;
        let modelMetadata;
        let selectedNotes = [];
        const midiToNoteName = {{
            60: 'C4', 61: 'C#4', 62: 'D4', 63: 'D#4', 64: 'E4', 65: 'F4', 66: 'F#4',
            67: 'G4', 68: 'G#4', 69: 'A4', 70: 'A#4', 71: 'B4', 72: 'C5'
        }};
        
        // Create piano keys
        function createPianoKeys() {{
            const piano = document.getElementById('piano');
            const startNote = 60; // Middle C
            const endNote = 72;   // C5
            
            // Create white keys
            for (let note = startNote; note <= endNote; note++) {{
                if ([1, 3, 6, 8, 10].indexOf(note % 12) === -1) {{ // If not a black key
                    const whiteKey = document.createElement('div');
                    whiteKey.className = 'white-key';
                    whiteKey.dataset.note = note;
                    whiteKey.addEventListener('click', () => toggleNote(note));
                    piano.appendChild(whiteKey);
                }}
            }}
            
            // Create black keys
            let whiteKeyCount = 0;
            for (let note = startNote; note <= endNote; note++) {{
                if ([1, 3, 6, 8, 10].indexOf(note % 12) !== -1) {{ // If it's a black key
                    const blackKey = document.createElement('div');
                    blackKey.className = 'black-key';
                    blackKey.dataset.note = note;
                    blackKey.style.left = `${{20 + (whiteKeyCount - 0.5) * 40}}px`;
                    blackKey.addEventListener('click', () => toggleNote(note));
                    piano.appendChild(blackKey);
                }} else {{
                    whiteKeyCount++;
                }}
            }}
        }}
        
        // Toggle note selection
        function toggleNote(note) {{
            const index = selectedNotes.indexOf(note);
            if (index === -1) {{
                // Add note
                selectedNotes.push(note);
                // Sort notes in order
                selectedNotes.sort((a, b) => a - b);
            }} else {{
                // Remove note
                selectedNotes.splice(index, 1);
            }}
            
            updatePianoUI();
            updateMelodyDisplay();
            document.getElementById('generateButton').disabled = selectedNotes.length === 0;
        }}
        
        // Update piano UI based on selected notes
        function updatePianoUI() {{
            // Reset all keys
            document.querySelectorAll('.white-key, .black-key').forEach(key => {{
                key.classList.remove('selected');
            }});
            
            // Mark selected keys
            selectedNotes.forEach(note => {{
                const key = document.querySelector(`[data-note="${{note}}"]`);
                if (key) key.classList.add('selected');
            }});
        }}
        
        // Update melody display
        function updateMelodyDisplay() {{
            const melodyDisplay = document.getElementById('melodyDisplay');
            melodyDisplay.innerHTML = '';
            
            if (selectedNotes.length === 0) {{
                melodyDisplay.textContent = 'No notes selected yet. Click on the piano keys above.';
                return;
            }}
            
            selectedNotes.forEach(note => {{
                const noteElement = document.createElement('span');
                noteElement.className = 'note-pill';
                noteElement.textContent = midiToNoteName[note] || `Note ${{note}}`;
                melodyDisplay.appendChild(noteElement);
            }});
        }}
        
        // Load the TFLite model
        async function loadModel() {{
            try {{
                const tflite = window.tflite;
                tfliteModel = await tflite.loadTFLiteModel(MODEL_PATH);
                console.log('Model loaded successfully');
                
                // Load metadata
                const response = await fetch(METADATA_PATH);
                modelMetadata = await response.json();
                console.log('Metadata loaded:', modelMetadata);
                
                document.getElementById('modelStatus').textContent = 'Model loaded successfully';
                document.getElementById('generateButton').disabled = selectedNotes.length === 0;
            }} catch (error) {{
                console.error('Error loading model:', error);
                document.getElementById('modelStatus').textContent = 'Error loading model: ' + error.message;
            }}
        }}
        
        // Generate chord progression
        async function generateChords() {{
            if (!tfliteModel || selectedNotes.length === 0) {{
                return;
            }}
            
            try {{
                document.getElementById('modelStatus').textContent = 'Generating chords...';
                document.getElementById('generateButton').disabled = true;
                
                // Get input shape from model metadata
                const inputShape = modelMetadata.input[0].shape;
                
                // Create input tensor with the right shape
                // For this model, we can only use one note at a time for now
                const inputTensor = tf.tensor(
                    [selectedNotes[0]], // Just use the first note as a demonstration
                    [inputShape[0], inputShape[1]],
                    'int32'
                );
                
                // Run inference
                const output = await tfliteModel.predict(inputTensor);
                const outputData = await output.array();
                
                // Find top 5 chord suggestions
                const chordProbs = outputData[0][0]; // [batch, sequence, chords]
                const topChords = findTopChords(chordProbs, 5);
                
                // Display results
                displayChordResults(topChords);
                
                // Clean up
                inputTensor.dispose();
                output.dispose();
                document.getElementById('modelStatus').textContent = 'Chords generated successfully';
                document.getElementById('generateButton').disabled = false;
            }} catch (error) {{
                console.error('Error generating chords:', error);
                document.getElementById('modelStatus').textContent = 'Error generating chords: ' + error.message;
                document.getElementById('generateButton').disabled = false;
            }}
        }}
        
        // Find top N chords by probability
        function findTopChords(chordProbs, n) {{
            // Create array of [index, probability] pairs
            const indexedProbs = chordProbs.map((prob, index) => [index, prob]);
            
            // Sort by probability (descending)
            indexedProbs.sort((a, b) => b[1] - a[1]);
            
            // Take top N results
            return indexedProbs.slice(0, n).map(([index, prob]) => {{
                // Map index to chord name using metadata if available
                let chordName = `Chord ${{index}}`;
                if (modelMetadata && modelMetadata.chord_mapping) {{
                    chordName = modelMetadata.chord_mapping[index] || chordName;
                }}
                
                return {{
                    index,
                    probability: prob,
                    name: chordName
                }};
            }});
        }}
        
        // Display chord results
        function displayChordResults(chords) {{
            const chordDisplay = document.getElementById('chordDisplay');
            chordDisplay.innerHTML = '<h3>Top Chord Suggestions</h3>';
            
            // Create elements for each chord
            chords.forEach((chord, i) => {{
                const chordElement = document.createElement('div');
                chordElement.className = 'chord';
                
                const chordName = document.createElement('div');
                chordName.className = 'chord-name';
                chordName.textContent = chord.name;
                
                const chordConfidence = document.createElement('div');
                chordConfidence.className = 'chord-confidence';
                // Convert raw output to percentage
                const confidencePercent = ((chord.probability + 10) / 20 * 100).toFixed(1);
                chordConfidence.textContent = `Confidence: ${{confidencePercent}}%`;
                
                chordElement.appendChild(chordName);
                chordElement.appendChild(chordConfidence);
                chordDisplay.appendChild(chordElement);
            }});
        }}
        
        // Clear melody
        function clearMelody() {{
            selectedNotes = [];
            updatePianoUI();
            updateMelodyDisplay();
            document.getElementById('generateButton').disabled = true;
            document.getElementById('chordDisplay').innerHTML = 
                '<p>Suggested chords will appear here after you generate them.</p>';
        }}
        
        // Initialize app
        document.addEventListener('DOMContentLoaded', () => {{
            createPianoKeys();
            updateMelodyDisplay();
            loadModel();
            
            // Add event listeners
            document.getElementById('generateButton').addEventListener('click', generateChords);
            document.getElementById('clearButton').addEventListener('click', clearMelody);
        }});
    </script>
</body>
</html>
"""
    
    # Save HTML file
    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"Web deployment files created in {output_dir}")
    print(f"- TFLite model: {output_dir / model_filename}")
    print(f"- Metadata: {metadata_path}")
    print(f"- HTML file: {html_path}")


def main() -> int:
    """
    Main function to prepare web deployment files.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Prepare models and web files for deployment")
    parser.add_argument(
        "--tflite-model", 
        type=Path, 
        default=Path("./model_data/model.tflite"),
        help="Path to the TFLite model"
    )
    parser.add_argument(
        "--chord-mapping", 
        type=Path, 
        default=Path("./model_data/chord_mapping.json"),
        help="Path to the chord mapping JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("./web"),
        help="Directory where web files will be saved"
    )
    
    args = parser.parse_args()
    
    try:
        if not args.tflite_model.exists():
            print(f"Error: TFLite model not found: {args.tflite_model}")
            return 1
        
        if not args.chord_mapping.exists():
            print(f"Error: Chord mapping file not found: {args.chord_mapping}")
            return 1
        
        # Create web deployment files
        create_web_deployment(args.tflite_model, args.chord_mapping, args.output_dir)
        
        print("\nWeb deployment preparation complete!")
        print(f"Files are ready in: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during web deployment preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())