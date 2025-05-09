"""
Logging utilities for the AlyaAloft application.
Provides support for different logging levels (DEBUG, DEV, PROD).
"""

import json
import logging
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import asyncio

# Create loggers dictionary to cache loggers
loggers = {}

# Initialize with default logging level
DEFAULT_LOG_LEVEL = "DEV"

def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """
    Configure the logging system based on the specified level.
    
    Args:
        level: Logging level (DEBUG, DEV, PROD)
    """
    # Import config here to avoid circular imports
    from app import app_config as config
    
    # Create logs directory if it doesn't exist
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level based on environment
    if level == "DEBUG":
        log_level = logging.DEBUG
    elif level == "DEV":
        log_level = logging.INFO
    else:  # PROD
        log_level = logging.WARNING
    
    root_logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create file handler for all logs
    log_file = config.LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatters
    if level == "DEBUG":
        # Detailed formatter for DEBUG level
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "func": "%(funcName)s", "line": %(lineno)d, '
            '"message": "%(message)s"}'
        )
    elif level == "DEV":
        # Moderate detail for DEV level
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}'
        )
    else:  # PROD
        # Minimal logging for PROD level
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"message": "%(message)s"}'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_handler = logging.getLogger("uvicorn")
    root_handler.handlers = []
    root_logger.addHandler(file_handler)
    
    # Log setup complete
    root_logger.info(f"Logging initialized with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    if name in loggers:
        return loggers[name]
    
    logger = logging.getLogger(name)
    
    # Import config here to avoid circular imports
    from app import app_config as config
    
    # Set module-specific level based on config
    logger.setLevel(logging.DEBUG if config.LOG_LEVEL == "DEBUG" else 
                   logging.INFO if config.LOG_LEVEL == "DEV" else 
                   logging.WARNING)
    
    loggers[name] = logger
    return logger


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls (detailed for DEBUG level).
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    logger = get_logger(func.__module__)
    
    # Check if the function is a coroutine function (async)
    is_async = asyncio.iscoroutinefunction(func)
    
    if is_async:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import config here to avoid circular imports
            from app import app_config as config
            
            # Only log function details in DEBUG mode
            if config.LOG_LEVEL == "DEBUG":
                func_args = ", ".join([repr(a) for a in args])
                func_kwargs = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [func_args, func_kwargs]))
                
                logger.debug(f"Calling async {func.__name__}({all_args})")
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    logger.debug(
                        f"Async {func.__name__} completed in {end_time - start_time:.4f} seconds"
                    )
                    return result
                except Exception as e:
                    end_time = time.time()
                    logger.exception(
                        f"Async {func.__name__} failed after {end_time - start_time:.4f} seconds: {str(e)}"
                    )
                    raise
            else:
                # Just call the function in non-DEBUG mode
                return await func(*args, **kwargs)
        
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import config here to avoid circular imports
            from app import app_config as config
            
            # Only log function details in DEBUG mode
            if config.LOG_LEVEL == "DEBUG":
                func_args = ", ".join([repr(a) for a in args])
                func_kwargs = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [func_args, func_kwargs]))
                
                logger.debug(f"Calling {func.__name__}({all_args})")
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    logger.debug(
                        f"{func.__name__} completed in {end_time - start_time:.4f} seconds"
                    )
                    return result
                except Exception as e:
                    end_time = time.time()
                    logger.exception(
                        f"{func.__name__} failed after {end_time - start_time:.4f} seconds: {str(e)}"
                    )
                    raise
            else:
                # Just call the function in non-DEBUG mode
                return func(*args, **kwargs)
        
        return sync_wrapper


def log_app_startup() -> None:
    """Log application startup information."""
    logger = get_logger(__name__)
    
    # Import config here to avoid circular imports
    from app import app_config as config
    
    startup_info = {
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
        "log_level": config.LOG_LEVEL,
        "data_dir": str(config.DATA_DIR),
        "models_dir": str(config.MODELS_DIR),
        "enable_ocr": config.ENABLE_OCR,
        "event": "Application startup"
    }
    
    logger.info(json.dumps(startup_info))


def log_request(request: Any, response_time: Optional[float] = None) -> None:
    """
    Log HTTP request details (level varies by configuration).
    
    Args:
        request: HTTP request object
        response_time: Time taken to process the request (optional)
    """
    logger = get_logger("request")
    
    # Import config here to avoid circular imports
    from app import app_config as config
    
    if config.LOG_LEVEL == "DEBUG":
        # Detailed request logging for DEBUG level
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if hasattr(request, "client") else None,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "event": "HTTP Request"
        }
        
        if response_time is not None:
            log_data["response_time"] = f"{response_time:.4f}s"
        
        logger.debug(json.dumps(log_data))
    elif config.LOG_LEVEL == "DEV":
        # Moderate request logging for DEV level
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if hasattr(request, "client") else None,
            "event": "HTTP Request"
        }
        
        if response_time is not None:
            log_data["response_time"] = f"{response_time:.4f}s"
        
        logger.info(json.dumps(log_data))
    else:  # PROD level - log minimal information
        if response_time is not None and response_time > 1.0:
            # Only log slow requests in PROD mode
            logger.warning(
                json.dumps({
                    "method": request.method,
                    "url": str(request.url),
                    "response_time": f"{response_time:.4f}s",
                    "event": "Slow HTTP Request"
                })
            )

# DON'T call setup_logging here - we'll call it from main.py
# setup_logging(DEFAULT_LOG_LEVEL) 