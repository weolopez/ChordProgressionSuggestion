#!/usr/bin/env python
"""
Create a simplified TensorFlow.js model for chord prediction.

This script creates a simple model for melody to chord prediction directly
in a format that can be loaded in the browser with TensorFlow.js.
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path


def create_simple_model(num_chords=48):
    """
    Create a simple model for melody to chord progression prediction.
    
    Args:
        num_chords: Number of chord classes to predict
        
    Returns:
        A compiled TensorFlow model
    """
    print(f"Creating a simple model with {num_chords} chord classes")
    
    # Define model architecture - much simpler than the original model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='melody_input'),
        tf.keras.layers.Embedding(128, 64),  # MIDI note range embedding
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_chords, activation='softmax', name='chord_output')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    return model


def save_model_for_web(model, output_dir, chord_mapping_path=None):
    """
    Save the model in TensorFlow.js format for web deployment.
    
    Args:
        model: The TensorFlow model to save
        output_dir: Directory where to save the model files
        chord_mapping_path: Path to the chord mapping JSON file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in standard SavedModel format first
    saved_model_dir = Path(output_dir) / "saved_model"
    tf.saved_model.save(model, str(saved_model_dir))
    
    # Convert to TensorFlow.js format
    tfjs_dir = Path(output_dir) / "tfjs_model"
    os.makedirs(tfjs_dir, exist_ok=True)
    
    # Save model in Keras format
    keras_path = Path(output_dir) / "model.keras"
    model.save(keras_path)
    
    # Use a subprocess to run the tensorflowjs_converter
    import subprocess
    
    # Convert Keras model to TensorFlow.js format
    cmd = [
        "tensorflowjs_converter",
        "--input_format=keras",
        str(keras_path),
        str(tfjs_dir)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error converting model to TensorFlow.js format:")
        print(result.stderr)
        raise RuntimeError("Failed to convert model")
    
    print(result.stdout)
    print(f"Model saved in TensorFlow.js format to {tfjs_dir}")
    
    # Copy or create chord mapping if needed
    if chord_mapping_path:
        shutil.copy(chord_mapping_path, tfjs_dir / "chord_mapping.json")
    else:
        # Create a simple chord mapping
        chord_mapping = {}
        for i in range(48):
            if i < 12:  # Major chords
                chord_mapping[str(i)] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][i]
            elif i < 24:  # Minor chords
                chord_mapping[str(i)] = ["Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"][i - 12]
            elif i < 36:  # 7th chords
                chord_mapping[str(i)] = ["C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7"][i - 24]
            else:  # Major 7th chords
                chord_mapping[str(i)] = ["Cmaj7", "C#maj7", "Dmaj7", "D#maj7", "Emaj7", "Fmaj7", "F#maj7", "Gmaj7", "G#maj7", "Amaj7", "A#maj7", "Bmaj7"][i - 36]
        
        with open(tfjs_dir / "chord_mapping.json", "w") as f:
            json.dump(chord_mapping, f, indent=2)
    
    # Create model metadata
    model_metadata = {
        "model_name": "Simple Melody to Chord Model",
        "description": "A simple model that suggests chord progressions for melodies",
        "version": "1.0.0",
        "input": {
            "name": "melody_input",
            "shape": [null, 1],
            "dtype": "int32"
        },
        "output": {
            "name": "chord_output",
            "shape": [null, 48],
            "dtype": "float32"
        }
    }
    
    with open(tfjs_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    return tfjs_dir


def create_web_page(model_dir):
    """
    Create a web page for using the model.
    
    Args:
        model_dir: Directory where the model files are saved
    """
    # Create HTML file
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chord Progression Suggestion</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.19.0"></script>
    <style>
        :root {
            --primary-color: #4a90e2;
            --secondary-color: #f0f7ff;
            --accent-color: #5cb85c;
            --text-color: #333;
            --light-gray: #f5f5f5;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            color: var(--text-color);
            background-color: var(--secondary-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        h1 {
            margin: 0;
            font-size: 2rem;
        }
        
        .app-description {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .main-content {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        
        .input-section, .output-section {
            flex: 1;
            min-width: 300px;
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .piano-container {
            margin: 15px 0;
            overflow-x: auto;
            white-space: nowrap;
        }
        
        .piano {
            display: inline-flex;
            position: relative;
            height: 120px;
        }
        
        .white-key, .black-key {
            box-sizing: border-box;
        }
        
        .white-key {
            width: 40px;
            height: 120px;
            background-color: white;
            border: 1px solid #ccc;
            z-index: 1;
        }
        
        .black-key {
            position: absolute;
            width: 30px;
            height: 80px;
            background-color: black;
            z-index: 2;
        }
        
        .white-key.selected {
            background-color: var(--primary-color);
        }
        
        .black-key.selected {
            background-color: var(--accent-color);
        }
        
        .melody-display {
            margin: 15px 0;
            border: 1px solid #ddd;
            padding: 10px;
            min-height: 50px;
            border-radius: 5px;
            background-color: var(--light-gray);
        }
        
        .note-pill {
            display: inline-block;
            background-color: var(--primary-color);
            color: white;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            margin-right: 10px;
        }
        
        button:hover {
            background-color: #4a9d4a;
        }
        
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        
        .chord-display {
            margin-top: 15px;
        }
        
        .chord {
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: var(--light-gray);
        }
        
        .chord-name {
            font-weight: bold;
            font-size: 1.2rem;
            color: var(--primary-color);
        }
        
        .chord-confidence {
            font-size: 0.9rem;
            color: #666;
        }
        
        .status {
            margin-top: 20px;
            color: #666;
            font-style: italic;
        }
        
        .model-info {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9rem;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
        }
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
            <p>This tool uses a deep learning model for chord progression suggestions based on melody input.</p>
            <p>Model version: 1.0.0 | Last updated: March 2025</p>
        </div>
    </div>

    <script>
        // Model path
        const MODEL_PATH = 'tfjs_model/model.json';
        const METADATA_PATH = 'tfjs_model/model_metadata.json';
        const CHORD_MAPPING_PATH = 'tfjs_model/chord_mapping.json';
        
        // State variables
        let model;
        let modelMetadata;
        let chordMapping;
        let selectedNotes = [];
        const midiToNoteName = {
            60: 'C4', 61: 'C#4', 62: 'D4', 63: 'D#4', 64: 'E4', 65: 'F4', 66: 'F#4',
            67: 'G4', 68: 'G#4', 69: 'A4', 70: 'A#4', 71: 'B4', 72: 'C5'
        };
        
        // Create piano keys
        function createPianoKeys() {
            const piano = document.getElementById('piano');
            const startNote = 60; // Middle C
            const endNote = 72;   // C5
            
            // Create white keys
            for (let note = startNote; note <= endNote; note++) {
                if ([1, 3, 6, 8, 10].indexOf(note % 12) === -1) { // If not a black key
                    const whiteKey = document.createElement('div');
                    whiteKey.className = 'white-key';
                    whiteKey.dataset.note = note;
                    whiteKey.addEventListener('click', () => toggleNote(note));
                    piano.appendChild(whiteKey);
                }
            }
            
            // Create black keys
            let whiteKeyCount = 0;
            for (let note = startNote; note <= endNote; note++) {
                if ([1, 3, 6, 8, 10].indexOf(note % 12) !== -1) { // If it's a black key
                    const blackKey = document.createElement('div');
                    blackKey.className = 'black-key';
                    blackKey.dataset.note = note;
                    blackKey.style.left = `${20 + (whiteKeyCount - 0.5) * 40}px`;
                    blackKey.addEventListener('click', () => toggleNote(note));
                    piano.appendChild(blackKey);
                } else {
                    whiteKeyCount++;
                }
            }
        }
        
        // Toggle note selection
        function toggleNote(note) {
            const index = selectedNotes.indexOf(note);
            if (index === -1) {
                // Add note
                selectedNotes.push(note);
                // Sort notes in order
                selectedNotes.sort((a, b) => a - b);
            } else {
                // Remove note
                selectedNotes.splice(index, 1);
            }
            
            updatePianoUI();
            updateMelodyDisplay();
            document.getElementById('generateButton').disabled = selectedNotes.length === 0;
        }
        
        // Update piano UI based on selected notes
        function updatePianoUI() {
            // Reset all keys
            document.querySelectorAll('.white-key, .black-key').forEach(key => {
                key.classList.remove('selected');
            });
            
            // Mark selected keys
            selectedNotes.forEach(note => {
                const key = document.querySelector(`[data-note="${note}"]`);
                if (key) key.classList.add('selected');
            });
        }
        
        // Update melody display
        function updateMelodyDisplay() {
            const melodyDisplay = document.getElementById('melodyDisplay');
            melodyDisplay.innerHTML = '';
            
            if (selectedNotes.length === 0) {
                melodyDisplay.textContent = 'No notes selected yet. Click on the piano keys above.';
                return;
            }
            
            selectedNotes.forEach(note => {
                const noteElement = document.createElement('span');
                noteElement.className = 'note-pill';
                noteElement.textContent = midiToNoteName[note] || `Note ${note}`;
                melodyDisplay.appendChild(noteElement);
            });
        }
        
        // Load the model and metadata
        async function loadResources() {
            try {
                // Load chord mapping
                const chordResponse = await fetch(CHORD_MAPPING_PATH);
                chordMapping = await chordResponse.json();
                console.log('Chord mapping loaded:', chordMapping);
                
                // Load model metadata
                const metadataResponse = await fetch(METADATA_PATH);
                modelMetadata = await metadataResponse.json();
                console.log('Model metadata loaded:', modelMetadata);
                
                // Load the model
                model = await tf.loadLayersModel(MODEL_PATH);
                console.log('Model loaded successfully');
                
                document.getElementById('modelStatus').textContent = 'Model loaded successfully';
                document.getElementById('generateButton').disabled = selectedNotes.length === 0;
            } catch (error) {
                console.error('Error loading resources:', error);
                document.getElementById('modelStatus').textContent = 'Error loading model: ' + error.message;
            }
        }
        
        // Generate chord progression
        async function generateChords() {
            if (!model || selectedNotes.length === 0) {
                return;
            }
            
            try {
                document.getElementById('modelStatus').textContent = 'Generating chords...';
                document.getElementById('generateButton').disabled = true;
                
                // For simplicity, just use the first note for prediction
                const inputNote = selectedNotes[0];
                
                // Create input tensor
                const inputTensor = tf.tensor2d([inputNote], [1, 1]);
                
                // Run inference
                const predictions = model.predict(inputTensor);
                const outputData = await predictions.array();
                
                // Get the chord probabilities
                const chordProbs = outputData[0];
                
                // Find top 5 chord suggestions
                const topChords = findTopChords(chordProbs, 5);
                
                // Display results
                displayChordResults(topChords);
                
                // Clean up
                inputTensor.dispose();
                predictions.dispose();
                document.getElementById('modelStatus').textContent = 'Chords generated successfully';
                document.getElementById('generateButton').disabled = false;
            } catch (error) {
                console.error('Error generating chords:', error);
                document.getElementById('modelStatus').textContent = 'Error generating chords: ' + error.message;
                document.getElementById('generateButton').disabled = false;
            }
        }
        
        // Find top N chords by probability
        function findTopChords(chordProbs, n) {
            // Create array of [index, probability] pairs
            const indexedProbs = chordProbs.map((prob, index) => [index, prob]);
            
            // Sort by probability (descending)
            indexedProbs.sort((a, b) => b[1] - a[1]);
            
            // Take top N results
            return indexedProbs.slice(0, n).map(([index, prob]) => {
                // Map index to chord name
                const chordName = chordMapping[index] || `Chord ${index}`;
                
                return {
                    index,
                    probability: prob,
                    name: chordName
                };
            });
        }
        
        // Display chord results
        function displayChordResults(chords) {
            const chordDisplay = document.getElementById('chordDisplay');
            chordDisplay.innerHTML = '<h3>Top Chord Suggestions</h3>';
            
            // Create elements for each chord
            chords.forEach((chord, i) => {
                const chordElement = document.createElement('div');
                chordElement.className = 'chord';
                
                const chordName = document.createElement('div');
                chordName.className = 'chord-name';
                chordName.textContent = chord.name;
                
                const chordConfidence = document.createElement('div');
                chordConfidence.className = 'chord-confidence';
                const confidencePercent = (chord.probability * 100).toFixed(1);
                chordConfidence.textContent = `Confidence: ${confidencePercent}%`;
                
                chordElement.appendChild(chordName);
                chordElement.appendChild(chordConfidence);
                chordDisplay.appendChild(chordElement);
            });
        }
        
        // Clear melody
        function clearMelody() {
            selectedNotes = [];
            updatePianoUI();
            updateMelodyDisplay();
            document.getElementById('generateButton').disabled = true;
            document.getElementById('chordDisplay').innerHTML = 
                '<p>Suggested chords will appear here after you generate them.</p>';
        }
        
        // Initialize app
        document.addEventListener('DOMContentLoaded', () => {
            createPianoKeys();
            updateMelodyDisplay();
            loadResources();
            
            // Add event listeners
            document.getElementById('generateButton').addEventListener('click', generateChords);
            document.getElementById('clearButton').addEventListener('click', clearMelody);
        });
    </script>
</body>
</html>
"""
    
    # Save HTML file
    html_path = Path(model_dir).parent / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"Web page created at {html_path}")


def main():
    # Create output directory
    output_dir = Path("../web")
    
    # Create model
    model = create_simple_model(num_chords=48)
    
    # Save model for web
    model_dir = save_model_for_web(model, output_dir)
    
    # Create web page
    create_web_page(model_dir)
    
    print("\nWeb deployment preparation complete!")
    print(f"Files are ready in: {output_dir}")


if __name__ == "__main__":
    main()