"""
Tests for the state manager module.
"""

import pytest
import asyncio
from pathlib import Path
import json
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.utils.state_manager import StateManager, ConversationState

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory for testing."""
    storage_dir = tmp_path / "test_conversations"
    storage_dir.mkdir()
    yield storage_dir
    shutil.rmtree(storage_dir)

@pytest.fixture
def state_manager(temp_storage_dir):
    """Create a state manager instance for testing."""
    return StateManager(str(temp_storage_dir))

@pytest.fixture
def conversation_state():
    """Create a conversation state instance for testing."""
    return ConversationState("test-conversation")

def test_conversation_state_initialization(conversation_state):
    """Test conversation state initialization."""
    assert conversation_state.conversation_id == "test-conversation"
    assert len(conversation_state.messages) == 0
    assert conversation_state.metadata["total_messages"] == 0
    assert conversation_state.metadata["total_tokens"] == 0
    assert isinstance(conversation_state.metadata["models_used"], set)
    assert len(conversation_state.metadata["validation_scores"]) == 0
    assert conversation_state.context["current_document"] is None
    assert len(conversation_state.context["relevant_chunks"]) == 0
    assert conversation_state.context["document_summary"] is None

def test_conversation_state_add_message(conversation_state):
    """Test adding messages to conversation state."""
    # Add a message without metadata
    conversation_state.add_message("user", "Test message")
    assert len(conversation_state.messages) == 1
    assert conversation_state.metadata["total_messages"] == 1
    
    # Add a message with metadata
    metadata = {
        "model": "mistral",
        "validation_score": 0.8,
        "tokens": 100
    }
    conversation_state.add_message("assistant", "Test response", metadata)
    assert len(conversation_state.messages) == 2
    assert conversation_state.metadata["total_messages"] == 2
    assert conversation_state.metadata["total_tokens"] == 100
    assert "mistral" in conversation_state.metadata["models_used"]
    assert 0.8 in conversation_state.metadata["validation_scores"]

def test_conversation_state_update_context(conversation_state):
    """Test updating conversation context."""
    # Update context with new values
    conversation_state.update_context(
        current_document="test.pdf",
        relevant_chunks=["chunk1", "chunk2"],
        document_summary="Test summary"
    )
    
    assert conversation_state.context["current_document"] == "test.pdf"
    assert len(conversation_state.context["relevant_chunks"]) == 2
    assert conversation_state.context["document_summary"] == "Test summary"

def test_conversation_state_serialization(conversation_state):
    """Test conversation state serialization and deserialization."""
    # Add some data to the state
    conversation_state.add_message("user", "Test message")
    conversation_state.update_context(current_document="test.pdf")
    
    # Convert to dictionary
    data = conversation_state.to_dict()
    
    # Create new state from dictionary
    new_state = ConversationState.from_dict(data)
    
    # Verify data
    assert new_state.conversation_id == conversation_state.conversation_id
    assert len(new_state.messages) == len(conversation_state.messages)
    assert new_state.context == conversation_state.context
    assert isinstance(new_state.metadata["models_used"], set)

@pytest.mark.asyncio
async def test_state_manager_create_conversation(state_manager):
    """Test creating a new conversation."""
    conversation_id = "test-conversation"
    state = await state_manager.create_conversation(conversation_id)
    
    assert state.conversation_id == conversation_id
    assert conversation_id in state_manager.active_conversations
    assert (state_manager.storage_dir / f"{conversation_id}.json").exists()

@pytest.mark.asyncio
async def test_state_manager_get_conversation(state_manager):
    """Test retrieving a conversation."""
    # Create a conversation
    conversation_id = "test-conversation"
    await state_manager.create_conversation(conversation_id)
    
    # Get the conversation
    state = await state_manager.get_conversation(conversation_id)
    assert state is not None
    assert state.conversation_id == conversation_id
    
    # Try to get a non-existent conversation
    state = await state_manager.get_conversation("non-existent")
    assert state is None

@pytest.mark.asyncio
async def test_state_manager_update_conversation(state_manager):
    """Test updating a conversation."""
    # Create a conversation
    conversation_id = "test-conversation"
    await state_manager.create_conversation(conversation_id)
    
    # Update the conversation
    state = await state_manager.update_conversation(
        conversation_id,
        current_document="test.pdf",
        relevant_chunks=["chunk1", "chunk2"]
    )
    
    assert state is not None
    assert state.context["current_document"] == "test.pdf"
    assert len(state.context["relevant_chunks"]) == 2

@pytest.mark.asyncio
async def test_state_manager_add_message(state_manager):
    """Test adding a message to a conversation."""
    # Create a conversation
    conversation_id = "test-conversation"
    await state_manager.create_conversation(conversation_id)
    
    # Add a message
    metadata = {
        "model": "mistral",
        "validation_score": 0.8,
        "tokens": 100
    }
    state = await state_manager.add_message(
        conversation_id,
        "assistant",
        "Test response",
        metadata
    )
    
    assert state is not None
    assert len(state.messages) == 1
    assert state.metadata["total_tokens"] == 100
    assert "mistral" in state.metadata["models_used"]

@pytest.mark.asyncio
async def test_state_manager_cleanup_old_conversations(state_manager):
    """Test cleaning up old conversations."""
    # Create conversations with different timestamps
    current_time = datetime.now()
    
    # Create an old conversation
    old_conversation = ConversationState("old-conversation")
    old_conversation.metadata["last_updated"] = (
        current_time - timedelta(hours=25)
    ).isoformat()
    await state_manager._save_state(old_conversation)
    
    # Create a recent conversation
    recent_conversation = ConversationState("recent-conversation")
    recent_conversation.metadata["last_updated"] = (
        current_time - timedelta(hours=12)
    ).isoformat()
    await state_manager._save_state(recent_conversation)
    
    # Run cleanup
    await state_manager.cleanup_old_conversations(max_age_hours=24)
    
    # Check results
    assert not (state_manager.storage_dir / "old-conversation.json").exists()
    assert (state_manager.storage_dir / "recent-conversation.json").exists()

@pytest.mark.asyncio
async def test_state_manager_persistence(state_manager):
    """Test conversation state persistence."""
    # Create and populate a conversation
    conversation_id = "test-conversation"
    state = await state_manager.create_conversation(conversation_id)
    state.add_message("user", "Test message")
    state.update_context(current_document="test.pdf")
    
    # Save the state
    await state_manager._save_state(state)
    
    # Create a new state manager instance
    new_manager = StateManager(str(state_manager.storage_dir))
    
    # Load the conversation
    loaded_state = await new_manager.get_conversation(conversation_id)
    
    # Verify data
    assert loaded_state is not None
    assert loaded_state.conversation_id == conversation_id
    assert len(loaded_state.messages) == 1
    assert loaded_state.context["current_document"] == "test.pdf" 