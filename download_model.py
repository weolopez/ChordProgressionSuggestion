#!/usr/bin/env python
"""
Download and extract the Magenta Hooktheory melody to chords model.

This script downloads the Hooktheory melody to chords model from Magenta's
Google Cloud Storage and prepares it for conversion to ONNX.
"""

import os
import sys
import requests
import tarfile
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

# Constants
MODEL_DIR = Path("./model_data")
HOOKTHEORY_MODEL_PATH = "gs://magentadata/models/hooktheory/mel2chords/checkpoint_45000/"
MODEL_CONFIG_PATH = "gs://magentadata/models/hooktheory/mel2chords/model.gin"
CHORD_MAPPING_PATH = "gs://magentadata/models/hooktheory/chord_mapping.json"

def download_gsutil(source_path: str, output_path: Path) -> bool:
    """
    Download a file or directory from Google Cloud Storage using gsutil.
    
    Args:
        source_path: GCS path to download from
        output_path: Local path to save to
    
    Returns:
        bool: True if download was successful
    """
    os.makedirs(output_path.parent, exist_ok=True)
    
    print(f"Downloading {source_path} to {output_path}")
    
    # Use rsync for directories, cp for files
    if source_path.endswith('/'):
        cmd = ["gsutil", "-m", "rsync", "-r", source_path, str(output_path)]
    else:
        cmd = ["gsutil", "cp", source_path, str(output_path)]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from GCS: {e}")
        return False


def download_model_files() -> Optional[Path]:
    """
    Download the Hooktheory model files from Google Cloud Storage.
    
    Returns:
        Path to the model directory or None if download failed
    """
    # Create model directories
    checkpoint_dir = MODEL_DIR / "checkpoint_45000"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Download the checkpoint files
    if not download_gsutil(HOOKTHEORY_MODEL_PATH, checkpoint_dir):
        print("Failed to download model checkpoint files")
        return None
    
    # Download the model config
    if not download_gsutil(MODEL_CONFIG_PATH, MODEL_DIR / "model.gin"):
        print("Failed to download model config")
        return None
    
    # Download the chord mapping
    if not download_gsutil(CHORD_MAPPING_PATH, MODEL_DIR / "chord_mapping.json"):
        print("Failed to download chord mapping")
        return None
    
    return checkpoint_dir


def verify_saved_model(model_dir: Path) -> bool:
    """
    Verify that the directory contains a valid SavedModel.
    
    Args:
        model_dir: Directory containing the model
    
    Returns:
        bool: True if the model appears valid
    """
    # For the Hooktheory model, we need to check for checkpoint files
    checkpoint_files = list(model_dir.glob("*.index"))
    if not checkpoint_files:
        print(f"Error: No checkpoint files found in {model_dir}")
        return False
    
    # Check for the data files
    data_files = list(model_dir.glob("*.data*"))
    if not data_files:
        print(f"Error: No data files found in {model_dir}")
        return False
    
    # Check for the model config file
    config_file = MODEL_DIR / "model.gin"
    if not config_file.exists():
        print(f"Error: model.gin not found")
        return False
    
    # Check for the chord mapping file
    mapping_file = MODEL_DIR / "chord_mapping.json"
    if not mapping_file.exists():
        print(f"Error: chord_mapping.json not found")
        return False
    
    return True


def main() -> Optional[Path]:
    """
    Download the Hooktheory melody to chords model.
    
    Returns:
        Path to the model directory or None if failed
    """
    print("Starting Hooktheory melody to chords model download")
    
    try:
        # Download model files
        model_dir = download_model_files()
        if model_dir is None:
            return None
        
        # Verify the model
        if verify_saved_model(model_dir):
            print(f"Successfully downloaded model to: {model_dir}")
            return model_dir
        else:
            print("Error: Downloaded model is not valid")
            return None
        
    except Exception as e:
        print(f"Error during download: {e}")
        return None


if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)