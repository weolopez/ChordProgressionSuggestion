# Web Integration Guide

This guide explains how to use the converted ONNX model with MLC for web deployment to create your chord progression suggestion application.

## Prerequisites

- Successfully converted Harmonizer model to ONNX format
- Basic knowledge of web development
- Node.js and npm installed (for web development)

## Overview of Web Integration

After converting the Harmonizer model to ONNX and compiling it with MLC, you can integrate it into a web application using these steps:

1. Compile the ONNX model to WebGPU/WASM using MLC
2. Create a web interface for melody input
3. Run the model inference in the browser
4. Display suggested chord progressions

## Step 1: Compile ONNX Model for Web

Use MLC to compile the ONNX model to WebGPU and WASM:

```bash
# Install MLC
pip install mlc-ai mlc-web

# Compile the ONNX model
python -c "
import mlc_ai
from mlc_ai import compile_model

compile_model(
    model_path='./onnx_models/harmonizer_optimized.onnx',
    target='webgpu',
    output_path='./web/model/harmonizer.wasm',
    system_lib=True
)
"
```

This creates WebGPU-compatible WASM code that can run efficiently in browsers that support WebGPU.

## Step 2: Create a Web Application

### Basic Structure

Create a web application with this folder structure:

```
web/
├── index.html
├── js/
│   ├── app.js
│   └── model-loader.js
├── css/
│   └── style.css
└── model/
    ├── harmonizer.wasm
    └── harmonizer.json
```

### HTML Interface

Create an interface for melody input and chord display:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chord Progression Suggester</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <h1>Chord Progression Suggester</h1>
        
        <div class="melody-input">
            <h2>Enter Melody</h2>
            <div id="piano-roll"></div>
            <button id="clear-melody">Clear</button>
        </div>
        
        <div class="controls">
            <button id="generate-chords">Generate Chord Progression</button>
        </div>
        
        <div class="output">
            <h2>Suggested Chords</h2>
            <div id="chord-display"></div>
        </div>
        
        <div class="status">
            <p id="model-status">Loading model...</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/@mlc-ai/web-runtime/dist/mlc-web-runtime.js"></script>
    <script src="js/model-loader.js"></script>
    <script src="js/app.js"></script>
</body>
</html>
```

### Loading the Model

Create a model loader (model-loader.js):

```javascript
class HarmonizerModel {
    constructor() {
        this.model = null;
        this.isLoaded = false;
    }
    
    async load() {
        try {
            const mlc = window.mlc;
            this.model = await mlc.loadModel("./model/harmonizer.wasm");
            this.isLoaded = true;
            
            document.getElementById("model-status").textContent = "Model loaded successfully";
            return true;
        } catch (error) {
            console.error("Failed to load model:", error);
            document.getElementById("model-status").textContent = 
                "Failed to load model. Check if your browser supports WebGPU.";
            return false;
        }
    }
    
    async generateChords(melodyInput) {
        if (!this.isLoaded) {
            throw new Error("Model not loaded");
        }
        
        // Prepare input in the format expected by the model
        const input = this.prepareInput(melodyInput);
        
        // Run inference
        const output = await this.model.runInference({ "input": input });
        
        // Process the output to get chord progression
        return this.processOutput(output);
    }
    
    prepareInput(melodyInput) {
        // Convert melody input to the format expected by the model
        // This will depend on how the Harmonizer model expects input
        // For example, convert note pitches to one-hot encoding
        // ...
        
        return new Float32Array(/* processed input */);
    }
    
    processOutput(output) {
        // Convert model output to chord progression
        // This will depend on how the Harmonizer model outputs chords
        // ...
        
        return /* chord progression */;
    }
}

// Create and export the model instance
window.harmonizerModel = new HarmonizerModel();
```

### User Interface Logic

Create the application logic (app.js):

```javascript
document.addEventListener('DOMContentLoaded', async () => {
    // Load the model
    await window.harmonizerModel.load();
    
    // Set up piano roll interface for melody input
    setupPianoRoll();
    
    // Set up generate button
    document.getElementById('generate-chords').addEventListener('click', async () => {
        // Get melody input from piano roll
        const melodyInput = getMelodyFromPianoRoll();
        
        try {
            // Generate chords
            const chords = await window.harmonizerModel.generateChords(melodyInput);
            
            // Display the chord progression
            displayChords(chords);
        } catch (error) {
            console.error("Error generating chords:", error);
            document.getElementById("chord-display").textContent = 
                "Error generating chords. Please try again.";
        }
    });
    
    // Clear button
    document.getElementById('clear-melody').addEventListener('click', () => {
        clearPianoRoll();
        document.getElementById("chord-display").textContent = "";
    });
});

function setupPianoRoll() {
    // Implement piano roll interface for melody input
    // ...
}

function getMelodyFromPianoRoll() {
    // Extract melody input from piano roll
    // ...
    return /* melody data */;
}

function displayChords(chords) {
    // Display the chord progression
    const chordDisplay = document.getElementById("chord-display");
    chordDisplay.innerHTML = "";
    
    // Create visual representation of chords
    // ...
}

function clearPianoRoll() {
    // Clear the piano roll interface
    // ...
}
```

## Step 3: Testing and Deployment

### Local Testing

1. Start a local web server:
   ```bash
   npx http-server ./web
   ```

2. Open in a browser that supports WebGPU (e.g., Chrome with WebGPU enabled)

3. Test with different melody inputs

### Deployment

To deploy to weolopez.com:

1. Build your web application:
   ```bash
   # If using a build tool like webpack
   npm run build
   ```

2. Upload the built files to your web server:
   ```bash
   # Example using rsync
   rsync -av ./web/ user@weolopez.com:/path/to/web/directory/
   ```

3. Configure your web server to serve the application

## Advanced Features

### Offline Support

Add Service Worker for offline capability:

```javascript
// In app.js
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registered:', registration);
            })
            .catch(error => {
                console.log('ServiceWorker registration failed:', error);
            });
    });
}
```

### Performance Optimization

For better user experience:

1. Add a loading indicator during model initialization and inference
2. Use Web Workers to run inference without blocking the UI
3. Implement caching for previously generated chord progressions

### Customization Options

Enhance the application with:

1. Options for different chord styles (jazz, pop, classical)
2. Ability to export chord progressions as MIDI
3. Playback functionality to hear the melody with chords

## Troubleshooting

### Common Issues

1. **WebGPU not supported**:
   - Offer a WebGL fallback version
   - Display clear message about browser requirements

2. **Model size too large**:
   - Consider model quantization or pruning
   - Implement progressive loading

3. **Inference too slow**:
   - Optimize the model further
   - Consider running inference on smaller chunks of melody

4. **Unexpected chord outputs**:
   - Check input preprocessing
   - Verify output postprocessing logic

## Resources

- [MLC Web Documentation](https://mlc.ai/web-demo/)
- [WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [Music Theory Libraries](https://github.com/tonaljs/tonal)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)