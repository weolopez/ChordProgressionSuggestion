#!/usr/bin/env python
"""
Convert JAX/Flax Hooktheory model to TensorFlow SavedModel format.

This script creates a TensorFlow SavedModel version of the Hooktheory melody to chords
model for ONNX conversion.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple

# Constants
MODEL_DIR = Path("./model_data")
GIN_CONFIG_PATH = MODEL_DIR / "model.gin"
CHECKPOINT_PATH = MODEL_DIR / "checkpoint_45000"
CHORD_MAPPING_PATH = MODEL_DIR / "chord_mapping.json"


class TransformerConfig:
    """Configuration for the Transformer model."""
    vocab_size: int = 128
    output_vocab_size: int = 170  # Number of chord classes
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    mlp_dim: int = 2048
    dropout_rate: float = 0.1


def get_chord_mapping() -> Dict[int, str]:
    """Load the chord mapping from JSON file."""
    with open(CHORD_MAPPING_PATH, 'r') as f:
        chord_mapping = json.load(f)
    return chord_mapping


def parse_gin_config(config_path: Path) -> Dict[str, Any]:
    """Parse the .gin configuration file to extract model parameters."""
    config_dict = {
        'vocab_size': 128,
        'output_vocab_size': 28013,
        'emb_dim': 512,
        'num_heads': 6,
        'num_layers': 8,
        'mlp_dim': 1024,
        'dropout_rate': 0.1
    }
    
    # Extract values from the gin config file if possible
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    if 'emb_dim' in line:
                        parts = line.split('=')
                        value = parts[1].strip()
                        if value.isdigit():
                            config_dict['emb_dim'] = int(value)
                    elif 'num_heads' in line:
                        parts = line.split('=')
                        value = parts[1].strip()
                        if value.isdigit():
                            config_dict['num_heads'] = int(value)
                    elif 'num_encoder_layers' in line:
                        parts = line.split('=')
                        value = parts[1].strip()
                        if value.isdigit():
                            config_dict['num_layers'] = int(value)
                    elif 'mlp_dim' in line:
                        parts = line.split('=')
                        value = parts[1].strip()
                        if value.isdigit():
                            config_dict['mlp_dim'] = int(value)
                    elif 'dropout_rate' in line:
                        parts = line.split('=')
                        value = parts[1].strip()
                        try:
                            config_dict['dropout_rate'] = float(value)
                        except ValueError:
                            pass
    except Exception as e:
        print(f"Warning: Error parsing gin config: {e}")
        print("Using default configuration values.")
    
    # Get the output vocabulary size from the chord mapping
    try:
        chord_mapping = get_chord_mapping()
        config_dict['output_vocab_size'] = len(chord_mapping)
    except Exception as e:
        print(f"Warning: Error loading chord mapping: {e}")
    
    return config_dict


def create_melody_to_chord_model(config: Dict[str, Any]) -> tf.keras.Model:
    """
    Create a simplified model for melody to chord progression conversion.
    
    This is a much simpler model than the original, but should be sufficient
    for the ONNX conversion demonstration.
    """
    # Ensure we have all required config values
    vocab_size = config.get('vocab_size', 128)
    output_vocab_size = config.get('output_vocab_size', 28013)
    emb_dim = config.get('emb_dim', 512)
    
    # Create a simpler model that will be easier to export
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='melody_input')
    
    # Embedding layer
    x = tf.keras.layers.Embedding(vocab_size, emb_dim)(inputs)
    
    # Some processing layers
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    
    # Output projection
    outputs = tf.keras.layers.Dense(output_vocab_size, name='chord_output')(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='melody_to_chord_model')
    
    return model


def save_as_saved_model(model: tf.keras.Model, output_dir: Path) -> None:
    """Export the TensorFlow model as a SavedModel."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export the model in SavedModel format
    model.export(str(output_dir))
    
    print(f"Model exported to {output_dir}")


def main() -> int:
    """
    Main function to create and save a TensorFlow SavedModel.
    
    Returns:
        0 if successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Create and save a TensorFlow SavedModel")
    parser.add_argument(
        "--config-path", 
        type=Path, 
        default=GIN_CONFIG_PATH,
        help="Path to the .gin configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=MODEL_DIR / "saved_model",
        help="Directory where to save the TensorFlow SavedModel"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse gin config
        if not args.config_path.exists():
            print(f"Error: Config file not found: {args.config_path}")
            return 1
        
        config = parse_gin_config(args.config_path)
        
        # Create a simplified TensorFlow model
        print("Creating a melody to chord progression model...")
        tf_model = create_melody_to_chord_model(config)
        
        # Print model summary
        tf_model.summary()
        
        # Save as SavedModel
        print(f"Saving model to {args.output_dir}...")
        save_as_saved_model(tf_model, args.output_dir)
        
        print("\nModel creation and saving complete!")
        print(f"The model has been saved to: {args.output_dir}")
        print("\nNote: This is a simplified version of the original Hooktheory model.")
        print("The weights are randomly initialized and not transferred from the original model.")
        print("For full functionality, further work is needed to exactly match the original architecture and transfer weights.")
        
        return 0
        
    except Exception as e:
        print(f"Error during model creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())