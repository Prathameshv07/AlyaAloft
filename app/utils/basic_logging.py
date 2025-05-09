"""
Basic logging setup for bootstrap purposes.
This module MUST NOT import from app.config to avoid circular dependencies.
"""

import logging
import os
import sys
from pathlib import Path

def setup_basic_logging():
    """Set up basic logging configuration without importing app.config."""
    # Calculate base directory - no imports needed
    current_file = Path(__file__)
    app_dir = current_file.parent.parent
    base_dir = app_dir.parent
    logs_dir = base_dir / "logs"
    
    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure basic logging
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level to INFO
    root_logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Redirect Uvicorn's logger to prevent duplicate logs
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True
    
    # Log startup
    root_logger.info("Basic logging initialized for application bootstrap")
    
    return root_logger 