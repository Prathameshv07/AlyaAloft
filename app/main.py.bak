"""
Main entry point for the AlyaAloft application.
"""

import os
from pathlib import Path
import logging
import importlib

# Set up basic logging first - no imports from app modules
from app.utils.basic_logging import setup_basic_logging
logger = setup_basic_logging()
logger.info("Starting application initialization")

# Now we can safely import other modules
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import the renamed app_config.py file
try:
    # Import the app_config module directly
    from app import app_config as config
    logger.info(f"Configuration loaded: {config.APP_NAME} v{config.APP_VERSION}")
except ImportError as e:
    logger.error(f"Error importing config: {str(e)}")
    raise

# Import routes - this is safe now because basic logging is already set up
from app.routes import api_routes, web_routes

# Now set up proper logging
from app.utils.logging_utils import setup_logging, log_app_startup, log_request
logger.info("Setting up advanced logging")
setup_logging(config.LOG_LEVEL)
logger.info("Advanced logging initialized")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI application
    """
    # Create app instance
    app = FastAPI(
        title=config.APP_NAME,
        description=config.APP_DESCRIPTION,
        version=config.APP_VERSION,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Validate configuration
    config.validate_config()
    
    # Mount static files
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static"
    )
    
    # Include routers
    app.include_router(web_routes.router)
    app.include_router(api_routes.router)
    
    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for all unhandled exceptions."""
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again later."}
        )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests_middleware(request: Request, call_next):
        """Middleware for logging HTTP requests."""
        start_time = None
        if config.LOG_LEVEL in ("DEBUG", "DEV"):
            start_time = __import__("time").time()
            
        response = await call_next(request)
        
        if config.LOG_LEVEL in ("DEBUG", "DEV"):
            process_time = __import__("time").time() - start_time
            log_request(request, process_time)
        
        return response
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Handle application startup events."""
        log_app_startup()
        logger.info("Application startup complete")
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Handle application shutdown events."""
        logger.info("Application shutting down")
    
    return app

# Create application instance
logger.info("Creating FastAPI application")
app = create_app()
logger.info("FastAPI application created")

# Run the application if executed directly
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENVIRONMENT", "development").lower() == "development"
    ) 