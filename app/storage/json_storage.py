"""
JSON file-based storage for documents and chat history.
"""

import json
import time
import uuid
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import filelock

from app import app_config
from app.utils.logging_utils import get_logger, log_function_call

# Create logger for this module
logger = get_logger(__name__)

# Storage file paths
DOCUMENTS_FILE = app_config.CHAT_HISTORY_DIR / "documents.json"

# File lock timeout in seconds
LOCK_TIMEOUT = 10

# Ensure chat history directory exists
app_config.CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Initialize documents file if it doesn't exist
if not DOCUMENTS_FILE.exists():
    with open(DOCUMENTS_FILE, "w") as f:
        json.dump({"documents": []}, f)

@log_function_call
def _get_lock_for_file(file_path: Path) -> filelock.FileLock:
    """
    Get a file lock for a given file path.
    
    Args:
        file_path: Path to the file to lock
        
    Returns:
        FileLock object
    """
    lock_file = file_path.with_suffix(file_path.suffix + ".lock")
    return filelock.FileLock(lock_file, timeout=LOCK_TIMEOUT)

@log_function_call
async def create_document(doc_obj: Dict[str, Any]) -> str:
    """
    Create a new document record.
    
    Args:
        doc_obj: Document object containing metadata
            {
                "filename": str,
                "title": str,
                "upload_time": str (ISO format),
                "file_path": str,
                ...
            }
            
    Returns:
        Document ID
    """
    # Generate a unique ID for the document
    doc_id = f"doc_{str(uuid.uuid4())[:8]}"
    
    # Add ID to document object
    doc_obj["id"] = doc_id
    
    # Set upload time if not provided
    if "upload_time" not in doc_obj:
        doc_obj["upload_time"] = datetime.now().isoformat()
    
    # Set default values for required fields
    doc_obj.setdefault("processed", False)
    
    # Write to documents file with file locking
    with _get_lock_for_file(DOCUMENTS_FILE):
        try:
            # Read existing documents
            async with aiofiles.open(DOCUMENTS_FILE, "r") as f:
                data = json.loads(await f.read())
            
            # Add new document
            data["documents"].append(doc_obj)
            
            # Write updated data
            async with aiofiles.open(DOCUMENTS_FILE, "w") as f:
                await f.write(json.dumps(data, indent=2))
            
            logger.info(f"Created document: {doc_id}")
            return doc_id
        
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            raise

@log_function_call
async def update_document(doc_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update an existing document.
    
    Args:
        doc_id: Document ID
        updates: Dictionary of fields to update
        
    Returns:
        Success boolean
    """
    with _get_lock_for_file(DOCUMENTS_FILE):
        try:
            # Read existing documents
            async with aiofiles.open(DOCUMENTS_FILE, "r") as f:
                data = json.loads(await f.read())
            
            # Find and update document
            for doc in data["documents"]:
                if doc["id"] == doc_id:
                    doc.update(updates)
                    
                    # Write updated data
                    async with aiofiles.open(DOCUMENTS_FILE, "w") as f:
                        await f.write(json.dumps(data, indent=2))
                    
                    logger.info(f"Updated document: {doc_id}")
                    return True
            
            logger.warning(f"Document not found for update: {doc_id}")
            return False
        
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise

@log_function_call
async def get_documents() -> List[Dict[str, Any]]:
    """
    Get list of all documents.
    
    Returns:
        List of document objects
    """
    try:
        async with aiofiles.open(DOCUMENTS_FILE, "r") as f:
            data = json.loads(await f.read())
        
        # Sort documents by upload time (newest first)
        documents = sorted(
            data["documents"],
            key=lambda x: x.get("upload_time", ""),
            reverse=True
        )
        
        return documents
    
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return []

@log_function_call
async def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Document object or None if not found
    """
    try:
        async with aiofiles.open(DOCUMENTS_FILE, "r") as f:
            data = json.loads(await f.read())
        
        # Find document by ID
        for doc in data["documents"]:
            if doc["id"] == doc_id:
                return doc
        
        logger.warning(f"Document not found: {doc_id}")
        return None
    
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        return None

@log_function_call
async def delete_document(doc_id: str) -> bool:
    """
    Delete a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Success boolean
    """
    with _get_lock_for_file(DOCUMENTS_FILE):
        try:
            # Read existing documents
            async with aiofiles.open(DOCUMENTS_FILE, "r") as f:
                data = json.loads(await f.read())
            
            # Find document index
            doc_index = None
            for i, doc in enumerate(data["documents"]):
                if doc["id"] == doc_id:
                    doc_index = i
                    break
            
            if doc_index is not None:
                # Remove document from list
                data["documents"].pop(doc_index)
                
                # Write updated data
                async with aiofiles.open(DOCUMENTS_FILE, "w") as f:
                    await f.write(json.dumps(data, indent=2))
                
                # Delete chat history file
                chat_file = app_config.CHAT_HISTORY_DIR / f"chat_{doc_id}.json"
                if chat_file.exists():
                    chat_file.unlink()
                
                logger.info(f"Deleted document: {doc_id}")
                return True
            
            logger.warning(f"Document not found for deletion: {doc_id}")
            return False
        
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

@log_function_call
async def save_chat_message(doc_id: str, message_obj: Dict[str, Any]) -> bool:
    """
    Save a chat message to a document's chat history.
    
    Args:
        doc_id: Document ID
        message_obj: Message object
            {
                "type": "user" or "system",
                "content": str,
                "timestamp": str (ISO format),
                ...
            }
            
    Returns:
        Success boolean
    """
    chat_file = app_config.CHAT_HISTORY_DIR / f"chat_{doc_id}.json"
    
    # Set timestamp if not provided
    if "timestamp" not in message_obj:
        message_obj["timestamp"] = datetime.now().isoformat()
    
    with _get_lock_for_file(chat_file):
        try:
            # Read existing chat history
            if chat_file.exists():
                async with aiofiles.open(chat_file, "r") as f:
                    data = json.loads(await f.read())
            else:
                data = {"messages": []}
            
            # Add new message
            data["messages"].append(message_obj)
            
            # Write updated data
            async with aiofiles.open(chat_file, "w") as f:
                await f.write(json.dumps(data, indent=2))
            
            logger.info(f"Saved chat message for document: {doc_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving chat message: {str(e)}")
            raise

@log_function_call
async def get_chat_history(doc_id: str) -> List[Dict[str, Any]]:
    """
    Get chat history for a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        List of chat messages
    """
    chat_file = app_config.CHAT_HISTORY_DIR / f"chat_{doc_id}.json"
    
    try:
        if not chat_file.exists():
            return []
        
        async with aiofiles.open(chat_file, "r") as f:
            data = json.loads(await f.read())
        
        return data.get("messages", [])
    
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return [] 