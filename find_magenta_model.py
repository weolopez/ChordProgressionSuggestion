#!/usr/bin/env python
"""
Find available models in the Magenta repository
"""

import requests
from bs4 import BeautifulSoup
import re
import json

def find_magenta_models():
    """Search for available models in the Magenta repository"""
    # Get the models directory
    url = "https://github.com/magenta/magenta/tree/master/magenta/models"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to access Magenta models directory: {response.status_code}")
        return None
        
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all directories in the models folder
    directories = []
    for item in soup.find_all('a', class_='js-navigation-open Link--primary'):
        if 'Directory' in item.get('aria-label', ''):
            directories.append(item.text.strip())
    
    print("Available models in Magenta repository:")
    for i, model in enumerate(directories, 1):
        print(f"{i}. {model}")
    
    return directories

def find_model_checkpoints(model_name):
    """Find available checkpoints for a specific model"""
    # Check first in the pretrained folder
    url = f"https://github.com/magenta/magenta/tree/master/magenta/models/{model_name}/pretrained"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"No pretrained directory found for {model_name}")
        # Try looking in the readme for download links
        return find_checkpoint_links(model_name)
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all files in the pretrained folder
    files = []
    for item in soup.find_all('a', class_='js-navigation-open Link--primary'):
        if 'File' in item.get('aria-label', ''):
            files.append(item.text.strip())
    
    if files:
        print(f"\nAvailable checkpoints for {model_name}:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
    
    return files

def find_checkpoint_links(model_name):
    """Look for checkpoint download links in the model's README.md"""
    url = f"https://github.com/magenta/magenta/blob/master/magenta/models/{model_name}/README.md"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"No README.md found for {model_name}")
        return []
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links in the README
    links = []
    for item in soup.find_all('a'):
        href = item.get('href', '')
        if any(ext in href for ext in ['.mag', '.tar.gz', '.zip', '.ckpt', '.tgz']):
            links.append(href)
    
    if links:
        print(f"\nCheckpoint download links found in README for {model_name}:")
        for i, link in enumerate(links, 1):
            print(f"{i}. {link}")
    
    return links

def search_google_storage():
    """Search for Magenta models in Google Cloud Storage"""
    print("\nSearching Google Cloud Storage for Magenta models...")
    url = "https://storage.googleapis.com/magentadata/models/"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to access Google Cloud Storage: {response.status_code}")
        return None
    
    # Parse XML
    soup = BeautifulSoup(response.text, 'xml')
    
    # Find all directories
    directories = []
    for item in soup.find_all('Contents'):
        key = item.find('Key')
        if key:
            path = key.text.strip()
            if '/' in path:
                directory = path.split('/')[0]
                if directory not in directories:
                    directories.append(directory)
    
    print("Available model directories in Google Cloud Storage:")
    for i, directory in enumerate(directories, 1):
        print(f"{i}. {directory}")
    
    return directories

def main():
    # Find all models
    models = find_magenta_models()
    if not models:
        print("Failed to find models in Magenta repository")
        return
    
    # Ask user which model to check
    model_idx = input("\nEnter the number of the model to check for checkpoints (or 'q' to quit): ")
    if model_idx.lower() == 'q':
        return
    
    try:
        model_idx = int(model_idx) - 1
        if model_idx < 0 or model_idx >= len(models):
            print("Invalid model number")
            return
        
        model_name = models[model_idx]
        # Find checkpoints for the model
        checkpoints = find_model_checkpoints(model_name)
        if not checkpoints:
            print(f"No checkpoints found for {model_name}")
    except ValueError:
        print("Invalid input. Please enter a number or 'q'.")
    
    # Search Google Cloud Storage
    search_google_storage()

if __name__ == "__main__":
    main()