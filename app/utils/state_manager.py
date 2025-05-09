"""
Custom state management for the AlyaAloft PDF Explainer application.
This module provides a lightweight alternative to LangChain's state management.
"""

from typing import Dict, Any, Optional, List
import json
import logging
from pathlib import Path
import asyncio
from datetime import datetime

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ConversationState:
    """Manages the state of a conversation."""
    
    def __init__(self, conversation_id: str):
        """
        Initialize a new conversation state.
        
        Args:
            conversation_id: Unique identifier for the conversation
        """
        self.conversation_id = conversation_id
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_messages": 0,
            "total_tokens": 0,
            "models_used": set(),
            "validation_scores": []
        }
        self.context: Dict[str, Any] = {
            "current_document": None,
            "relevant_chunks": [],
            "document_summary": None
        }
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to the conversation.
        
        Args:
            role: Role of the message sender (user/assistant)
            content: Message content
            metadata: Optional metadata about the message
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
            # Update conversation metadata
            if "model" in metadata:
                self.metadata["models_used"].add(metadata["model"])
            if "validation_score" in metadata:
                self.metadata["validation_scores"].append(metadata["validation_score"])
            if "tokens" in metadata:
                self.metadata["total_tokens"] += metadata["tokens"]
        
        self.messages.append(message)
        self.metadata["total_messages"] += 1
        self.metadata["last_updated"] = datetime.now().isoformat()
    
    def update_context(self, **kwargs):
        """
        Update the conversation context.
        
        Args:
            **kwargs: Key-value pairs to update in the context
        """
        self.context.update(kwargs)
        self.metadata["last_updated"] = datetime.now().isoformat()
    
    def get_context(self) -> Dict[str, Any]:
        """Get the current conversation context."""
        return self.context
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation."""
        return self.messages
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get conversation metadata."""
        # Convert sets to lists for JSON serialization
        metadata = self.metadata.copy()
        metadata["models_used"] = list(metadata["models_used"])
        return metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the conversation state to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "metadata": self.get_metadata(),
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create a conversation state from a dictionary."""
        state = cls(data["conversation_id"])
        state.messages = data["messages"]
        state.metadata = data["metadata"]
        state.context = data["context"]
        # Convert lists back to sets
        state.metadata["models_used"] = set(state.metadata["models_used"])
        return state

class StateManager:
    """Manages conversation states for the application."""
    
    def __init__(self, storage_dir: str = "conversations"):
        """
        Initialize the state manager.
        
        Args:
            storage_dir: Directory to store conversation states
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.active_conversations: Dict[str, ConversationState] = {}
    
    async def create_conversation(self, conversation_id: str) -> ConversationState:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            New conversation state
        """
        state = ConversationState(conversation_id)
        self.active_conversations[conversation_id] = state
        await self._save_state(state)
        return state
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Conversation state if found, None otherwise
        """
        # Check active conversations first
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Try to load from storage
        state = await self._load_state(conversation_id)
        if state:
            self.active_conversations[conversation_id] = state
        return state
    
    async def update_conversation(self, conversation_id: str, **kwargs) -> Optional[ConversationState]:
        """
        Update a conversation's context.
        
        Args:
            conversation_id: Unique identifier for the conversation
            **kwargs: Key-value pairs to update in the context
            
        Returns:
            Updated conversation state if found, None otherwise
        """
        state = await self.get_conversation(conversation_id)
        if state:
            state.update_context(**kwargs)
            await self._save_state(state)
        return state
    
    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[ConversationState]:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            role: Role of the message sender (user/assistant)
            content: Message content
            metadata: Optional metadata about the message
            
        Returns:
            Updated conversation state if found, None otherwise
        """
        state = await self.get_conversation(conversation_id)
        if state:
            state.add_message(role, content, metadata)
            await self._save_state(state)
        return state
    
    async def _save_state(self, state: ConversationState):
        """Save a conversation state to storage."""
        try:
            file_path = self.storage_dir / f"{state.conversation_id}.json"
            data = state.to_dict()
            await asyncio.to_thread(
                lambda: file_path.write_text(json.dumps(data, indent=2))
            )
        except Exception as e:
            logger.error(f"Error saving conversation state: {str(e)}")
    
    async def _load_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Load a conversation state from storage."""
        try:
            file_path = self.storage_dir / f"{conversation_id}.json"
            if not file_path.exists():
                return None
            
            data = await asyncio.to_thread(
                lambda: json.loads(file_path.read_text())
            )
            return ConversationState.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading conversation state: {str(e)}")
            return None
    
    async def cleanup_old_conversations(self, max_age_hours: int = 24):
        """
        Clean up old conversations from storage.
        
        Args:
            max_age_hours: Maximum age of conversations to keep
        """
        try:
            current_time = datetime.now()
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    last_updated = datetime.fromisoformat(data["metadata"]["last_updated"])
                    age_hours = (current_time - last_updated).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        file_path.unlink()
                        if data["conversation_id"] in self.active_conversations:
                            del self.active_conversations[data["conversation_id"]]
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {str(e)}")

# Create singleton instance
state_manager = StateManager() 