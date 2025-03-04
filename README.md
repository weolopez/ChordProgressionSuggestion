# Chord Progression Suggestion Tool

This project implements a web-based tool that suggests chord progressions for melodies, allowing musicians to quickly find harmonies that complement their musical ideas.

## Overview

Originally planned as a conversion of Magenta's Harmonizer model to a web-based format, this project evolved to use a custom JavaScript implementation that suggests chords based on music theory principles. The tool features an interactive piano interface that lets users input melodies and receive chord suggestions.

## Features

- Interactive piano keyboard for melody input
- Support for multiple note selection
- Chord suggestions based on music theory relationships
- Ranked list of chord options with confidence scores
- Pure JavaScript implementation with no dependencies
- Responsive web interface

## Prerequisites

- A modern web browser (Chrome, Firefox, Safari, Edge)
- No server-side requirements - the application runs entirely in the browser

## Installation and Usage

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ChordProgressionSuggestion.git
   ```

2. Open the web interface:
   ```
   open ChordProgressionSuggestion/modelConversion/web/index.html
   ```
   
   Alternatively, you can simply double-click the `index.html` file or drag it into your browser.

3. Using the tool:
   - Click on piano keys to select melody notes
   - Click the "Generate Chords" button to see chord suggestions
   - Use "Clear Melody" to start over

## Implementation Details

The application consists of three main files:

1. `web/index.html` - The web interface with piano keyboard and chord display
2. `web/model.js` - JavaScript implementation of the chord suggestion algorithm  
3. `web/model_metadata.json` - Chord mapping and model configuration data

### How the Model Works

The JavaScript model:
1. Maps each melody note to a set of potential chords using music theory relationships
2. Assigns probabilities based on:
   - Whether the chord contains the melody note
   - Common harmonic relationships (e.g., relative major/minor)
   - Voice leading principles
3. Combines probabilities across all selected notes
4. Ranks chords by their overall probability

## Customization

You can customize the model by modifying:

1. The chord mappings in `model_metadata.json`
2. The weighting algorithm in the `initializeRelationships()` method in `model.js`
3. The UI appearance through CSS in `index.html`

## Hosting

To deploy the tool on a web server:
1. Upload all three files from the `web/` directory to your server
2. No server-side processing is required - everything runs in the browser

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [Magenta's](https://github.com/magenta/magenta) Harmonizer model
- Piano UI implementation based on web music interface best practices
- Chord theory relationships derived from music theory principles# modelConversion
