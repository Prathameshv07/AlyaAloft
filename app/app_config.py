"""
Configuration settings for the AlyaAloft application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# App information
APP_NAME = "AlyaAloft"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AlyaAloft PDF Question-Answering System with FLAN-T5"

# API configuration
API_V1_STR = "/api/v1"

# Paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
CONVERSATIONS_DIR = ROOT_DIR / "conversations"
LOGS_DIR = ROOT_DIR / "logs"
OFFLOAD_DIR = ROOT_DIR / "offload"
MODELS_DIR = ROOT_DIR / "models"
CHAT_HISTORY_DIR = CONVERSATIONS_DIR  # Adding missing directory reference

# File size limits
MAX_UPLOAD_SIZE_MB = 25  # Maximum upload size in MB
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Convert to bytes

# App paths
SQLITE_DB_PATH = DATA_DIR / "alyaaloft.db"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = "INFO"  # Adding missing log level

# Web interface settings
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": False,
    "reload": False
}

# Logging settings
LOGGING_CONFIG = {
    "log_level": logging.INFO,
    "file_log_level": logging.DEBUG,
    "console_log_level": logging.INFO,
    "log_file": LOGS_DIR / "alyaaloft.log",
    "max_log_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 3
}

# PDF processing settings
PDF_CONFIG = {
    "chunk_size": 1024,          # Characters per chunk
    "chunk_overlap": 256,        # Characters overlap between chunks
    "separator": "\n\n",         # Prefer paragraph separation
    "include_metadata": True,    # Add metadata to chunks
    "extract_images": True,      # Extract images where supported
    "extract_tables": True,      # Extract tables where supported
    "ocr_enabled": False,        # Only enable OCR when needed
    "ocr_languages": ["eng"]     # Default OCR language
}

# OCR configuration
ENABLE_OCR = PDF_CONFIG["ocr_enabled"]  # Enable/disable OCR globally

# Storage settings
STORAGE_CONFIG = {
    "storage_type": "sqlite",        # sqlite, memory
    "max_documents": 100,            # Maximum documents to store
    "max_conversations": 50,         # Maximum conversations to store
    "auto_delete_threshold": 30,     # Days before auto-delete (if enabled)
    "auto_delete_enabled": False,    # Whether to auto-delete old items
    "embedding_dimensions": 384      # Dimensions for the embeddings
}

# Model configuration
MISTRAL_CONFIG = {
    "model_type": "mistral",
    "gpu_layers": 24,                # Use GPU for these layers
    "context_length": 2048,          # Context window size
    "threads": 4,                    # CPU threads
    "batch_size": 1                  # Smaller batch size for memory efficiency
}

MISTRAL_GENERATION_PARAMS = {
    "temperature": 0.7,              # Higher value = more creative
    "top_p": 0.95,                   # Nucleus sampling
    "repetition_penalty": 1.1,       # Penalize repetition
}

# T5 transformer configuration
T5_CONFIG = {
    "max_length": 1024,              # Maximum generation length
    "min_length": 64,                # Minimum generation length
    "temperature": 0.6,              # Temperature for generation
    "top_p": 0.92,                   # Top-p sampling
    "repetition_penalty": 1.3,       # Penalize repetition
    "no_repeat_ngram_size": 3,       # Avoid repeating n-grams
    "do_sample": True,               # Enable sampling
    "num_beams": 4,                  # Number of beams for beam search
    "early_stopping": True           # Stop when finished
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # all-MiniLM-L6-v2 is a good balance of speed and quality
    "use_gpu": True,
    "max_seq_length": 256,
    "normalize_embeddings": True
}

# UI Configuration
UI_CONFIG = {
    "theme": "light",              # light, dark
    "code_highlighting": True,     # Syntax highlighting in code blocks
    "show_citations": True,        # Show citations for responses
    "show_confidence": True,       # Show confidence scores
    "show_processing_time": True,  # Show time taken to process
    "max_display_chunks": 3,       # Maximum context chunks to display
    "citation_style": "numbered"   # numbered, brackets, footnotes
}

# Conversation settings
CONVERSATION_CONFIG = {
    "max_history_length": 5,       # Number of messages to keep in context
    "max_context_chunks": 4,       # Maximum document chunks to include
    "max_tokens_per_message": 512  # Maximum tokens per user message
}

# Feature flags
FEATURE_FLAGS = {
    "enable_pdf_upload": True,
    "enable_web_search": False,
    "enable_conversation_export": True,
    "enable_conversation_import": True,
    "enable_document_deletion": True,
    "enable_auto_citation": True,
    "enable_advanced_prompting": True,
    "enable_model_selection": False,
}

def validate_config():
    """
    Validate the configuration settings.
    Checks that all required directories exist and essential settings are valid.
    """
    # Check that all required directories exist
    required_dirs = [DATA_DIR, LOGS_DIR, CONVERSATIONS_DIR, OFFLOAD_DIR, MODELS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            os.makedirs(directory, exist_ok=True)
    
    # Validate app settings
    if not APP_NAME or not APP_VERSION:
        raise ValueError("APP_NAME and APP_VERSION must be defined")
    
    # Validate API configuration
    if not API_V1_STR:
        raise ValueError("API_V1_STR must be defined")
    
    # Validate file size limits
    if MAX_UPLOAD_SIZE_MB <= 0:
        raise ValueError("MAX_UPLOAD_SIZE_MB must be positive")
    
    return True 