#!/usr/bin/env python
"""
Download and verify the FLAN-T5 model.
"""

import os
import sys
import shutil
from pathlib import Path
import logging
import argparse

# Add the project root to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config.model_config import MODELS_DIR, T5_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_t5_model(model_name="google/flan-t5-base", force_download=False):
    """
    Download and prepare the FLAN-T5 model.
    
    Args:
        model_name: HuggingFace model identifier for the T5 model
        force_download: Whether to force download even if the model exists
    """
    # Create the target directory
    os.makedirs(T5_PATH, exist_ok=True)
    
    # Check if model already exists
    if Path(T5_PATH / "config.json").exists() and not force_download:
        logger.info(f"T5 model already exists at {T5_PATH}. Use --force to download again.")
        return
    
    # If force download and directory exists, delete it first
    if force_download and Path(T5_PATH).exists():
        logger.info(f"Removing existing T5 model files for forced download")
        shutil.rmtree(T5_PATH)
        os.makedirs(T5_PATH, exist_ok=True)
    
    logger.info(f"Downloading T5 model '{model_name}' to {T5_PATH}")
    
    try:
        # Download the tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(T5_PATH)
        logger.info("Tokenizer downloaded successfully")
        
        # Download the model
        logger.info("Downloading model (this might take a while)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(T5_PATH)
        logger.info("Model downloaded successfully")
        
        # Verify the download
        logger.info("Verifying downloaded files...")
        required_files = ["config.json", "tokenizer_config.json"]
        for file in required_files:
            if not (T5_PATH / file).exists():
                logger.error(f"Missing required file: {file}")
                return
        
        if not list(T5_PATH.glob("*.bin")) and not list(T5_PATH.glob("*.safetensors")):
            logger.error("No model weight files found")
            return
        
        logger.info("Model verification completed successfully")
        logger.info(f"T5 model downloaded and saved to {T5_PATH}")
        
    except Exception as e:
        logger.error(f"Error downloading T5 model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download FLAN-T5 model for AlyaAloft")
    parser.add_argument(
        "--model", 
        default="google/flan-t5-base", 
        help="HuggingFace model name/path (default: google/flan-t5-base)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    try:
        download_t5_model(args.model, args.force)
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 