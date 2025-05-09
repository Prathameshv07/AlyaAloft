"""
Web routes for the AlyaAloft application.
Provides endpoints for rendering HTML templates.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app import app_config as config

# Create a logger for this module
from app.utils.logging_utils import get_logger
logger = get_logger(__name__)

# Create a router
router = APIRouter(tags=["web"])

# Create Jinja2 templates
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

@router.get("/", response_class=HTMLResponse)
async def intro_page(request: Request):
    """
    Render the intro page with project information.
    
    Args:
        request: HTTP request
        
    Returns:
        HTML response
    """
    # Get template context
    context = {
        "request": request,
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
        "app_description": config.APP_DESCRIPTION,
    }
    
    # Render template
    return templates.TemplateResponse("intro.html", context)

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """
    Render the chat interface.
    
    Args:
        request: HTTP request
        
    Returns:
        HTML response
    """
    # Get template context
    context = {
        "request": request,
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
        "max_upload_size_mb": config.MAX_UPLOAD_SIZE_MB,
    }
    
    # Render template
    return templates.TemplateResponse("chat.html", context) 