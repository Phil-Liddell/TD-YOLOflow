#!/usr/bin/env python3
"""
Model downloader for YOLOflow
Downloads genuine YOLOv11 models from official sources
"""

import os
import json
import argparse
import urllib.request
import sys
from tqdm import tqdm
import subprocess
import pkg_resources

# Model configuration file path
MODELS_JSON = os.path.join(os.path.dirname(__file__), "models", "models.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Required Ultralytics version for YOLOv11
REQUIRED_VERSION = "8.3.0"

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download from URL with progress bar"""
    print(f"Downloading {url} to {output_path}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    return output_path

def load_model_config():
    """Load model configuration from JSON file"""
    try:
        with open(MODELS_JSON, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading models config: {e}")
        sys.exit(1)

def check_ultralytics_version():
    """Check if ultralytics is installed and has the required version"""
    try:
        from ultralytics import __version__ as ultralytics_version
        print(f"Found Ultralytics version: {ultralytics_version}")
        
        # Convert version strings to tuples for comparison
        current_version = pkg_resources.parse_version(ultralytics_version)
        required_version = pkg_resources.parse_version(REQUIRED_VERSION)
        
        if current_version < required_version:
            print(f"Warning: Ultralytics version {ultralytics_version} is older than required version {REQUIRED_VERSION}")
            print("YOLOv11 models require a newer version of Ultralytics.")
            
            # Ask user for permission to upgrade
            response = input(f"Would you like to upgrade Ultralytics to version {REQUIRED_VERSION}? (y/n): ")
            if response.lower() in ['y', 'yes']:
                print(f"Upgrading Ultralytics to version {REQUIRED_VERSION}...")
                subprocess.run([sys.executable, "-m", "pip", "install", f"ultralytics>={REQUIRED_VERSION}"], check=True)
                print("Upgrade completed. Please restart the application.")
            else:
                print("Continuing with current version, but YOLOv11 models may not work correctly.")
        else:
            print(f"Ultralytics version is compatible with YOLOv11 models.")
        
    except ImportError:
        print("Ultralytics is not installed. Installing required version...")
        subprocess.run([sys.executable, "-m", "pip", "install", f"ultralytics>={REQUIRED_VERSION}"], check=True)
        print(f"Installed Ultralytics version {REQUIRED_VERSION}")

def download_models(model_name=None):
    """Download models based on configuration"""
    config = load_model_config()
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check Ultralytics version
    check_ultralytics_version()
    
    # Get models to download
    if model_name:
        models = [m for m in config['models'] if m['name'].lower() == model_name.lower()]
        if not models:
            print(f"Model '{model_name}' not found in config.")
            sys.exit(1)
    else:
        # Default: download only the default model
        default_model = config.get('default_model', 'YOLOv11n')
        models = [m for m in config['models'] if m['name'] == default_model or m['file'] == default_model]
        if not models:
            # Fall back to just the smallest model if default not found
            print("Default model not found in config. Downloading smallest model (YOLOv11n)...")
            models = [m for m in config['models'] if 'n.pt' in m.get('file', '')][:1]
            if not models:
                # If still nothing, just take the first one
                models = [config['models'][0]]
    
    # Download each model
    for model in models:
        # Support both 'file' and 'filename' keys for backward compatibility
        filename = model.get('file', model.get('filename'))
        if not filename:
            print(f"Error: No filename specified for model {model['name']}")
            continue
            
        output_path = os.path.join(MODELS_DIR, filename)
        
        # Skip if model already exists
        if os.path.exists(output_path):
            print(f"Model {model['name']} already exists at {output_path}")
            continue
            
        try:
            download_url(model['url'], output_path)
            # Handle case where size_mb might not be specified
            size_info = f"({model['size_mb']} MB)" if 'size_mb' in model else ""
            print(f"Downloaded {model['name']} {size_info}")
        except Exception as e:
            print(f"Error downloading {model['name']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download detection models for YOLOflow")
    parser.add_argument('--model', help='Specific model to download (default: all)')
    args = parser.parse_args()
    
    print("YOLOflow Model Downloader")
    print("-" * 40)
    download_models(args.model)
    print("Download process completed")