"""
Tests for the model orchestrator module.
"""

import pytest
import asyncio
from pathlib import Path
import json
from unittest.mock import Mock, patch, AsyncMock

from app.utils.model_orchestrator import ModelOrchestrator
from app.config.model_config import (
    T5_PATH,
    T5_CONFIG,
    STANDARD_PROMPT_TEMPLATE, 
    VALIDATION_PROMPT_TEMPLATE, 
    T5_PROMPT_TEMPLATE,
    PIPELINE_CONFIG
)

@pytest.fixture
def model_orchestrator():
    """Create a model orchestrator instance for testing."""
    return ModelOrchestrator()

@pytest.fixture
def mock_t5_model():
    """Create a mock T5 model."""
    model = Mock()
    model.generate = Mock(return_value=[0])  # T5 returns token IDs
    return model

@pytest.fixture
def mock_t5_tokenizer():
    """Create a mock T5 tokenizer."""
    tokenizer = Mock()
    tokenizer.__call__ = Mock(return_value={"input_ids": [1, 2, 3]})
    tokenizer.decode = Mock(return_value="Test response from T5")
    return tokenizer

@pytest.mark.asyncio
async def test_load_models(model_orchestrator):
    """Test loading all models."""
    with patch("app.utils.model_orchestrator.AutoTokenizer") as mock_tokenizer, \
         patch("app.utils.model_orchestrator.AutoModelForSeq2SeqLM") as mock_seq2seq:
        
        # Configure mocks
        mock_tokenizer.from_pretrained = AsyncMock()
        mock_seq2seq.from_pretrained = AsyncMock()
        
        # Patch Path.exists to simulate model files presence
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'glob', return_value=['model.bin']):
            
            # Load models
            await model_orchestrator.load_models()
            
            # Check T5 model loading
            mock_tokenizer.from_pretrained.assert_called_with(str(T5_PATH))
            mock_seq2seq.from_pretrained.assert_called_with(str(T5_PATH), device_map="auto")

@pytest.mark.asyncio
async def test_generate_response_with_t5(
    model_orchestrator,
    mock_t5_model,
    mock_t5_tokenizer
):
    """Test generating a response using T5."""
    # Set up the orchestrator with T5 model
    model_orchestrator.t5_model = mock_t5_model
    model_orchestrator.t5_tokenizer = mock_t5_tokenizer
    model_orchestrator.t5_available = True
    model_orchestrator.t5_verified = True
    
    # Mock prompt manager
    with patch("app.utils.model_orchestrator.prompt_manager") as mock_prompt_manager:
        # Configure mocks
        mock_prompt_manager.detect_complexity.return_value = {"overall": 0.5}
        mock_prompt_manager.generate_structured_prompt.return_value = "Enhanced prompt"
        mock_prompt_manager.post_process_response.return_value = "Test response from T5"
        
        # Test query and context
        query = "What is the main topic?"
        context = "The main topic is artificial intelligence."
        
        # Generate response
        response, metadata = await model_orchestrator.generate_response(query, context)
        
        # Check that T5 was used as primary model
        assert metadata["primary_model"] == "t5"
        assert response == "Test response from T5"

@pytest.mark.asyncio
async def test_generate_with_t5_enhanced(
    model_orchestrator,
    mock_t5_model,
    mock_t5_tokenizer
):
    """Test generating a response with enhanced prompts."""
    # Set up the orchestrator with T5 model
    model_orchestrator.t5_model = mock_t5_model
    model_orchestrator.t5_tokenizer = mock_t5_tokenizer
    model_orchestrator.t5_available = True
    
    # Test prompt
    prompt = "Enhanced test prompt"
    
    # Generate response
    response = await model_orchestrator._generate_with_t5_enhanced(prompt)
    
    # Check tokenizer was called correctly
    mock_t5_tokenizer.__call__.assert_called_with(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    
    # Check model generated response
    mock_t5_model.generate.assert_called()
    
    # Check response decoding
    mock_t5_tokenizer.decode.assert_called_with(0, skip_special_tokens=True)
    
    # Check returned response
    assert response == "Test response from T5"

@pytest.mark.asyncio
async def test_generate_with_iterative_refinement(
    model_orchestrator,
    mock_t5_model,
    mock_t5_tokenizer
):
    """Test generating a response with iterative refinement for complex questions."""
    # Set up the orchestrator with T5 model
    model_orchestrator.t5_model = mock_t5_model
    model_orchestrator.t5_tokenizer = mock_t5_tokenizer
    model_orchestrator.t5_available = True
    
    # Test query and prompt
    query = "What is the complex relationship between quantum mechanics and general relativity?"
    prompt = "Enhanced test prompt for complex question"
    
    # Configure mock to provide staged responses
    mock_t5_tokenizer.decode.side_effect = [
        "Key aspects include: wave-particle duality, measurement problem, spacetime curvature",
        "Initial answer about quantum mechanics and relativity",
        "Missing aspects: quantum gravity, incompatibilities at singularities",
        "Comprehensive answer covering all aspects of quantum mechanics and relativity"
    ]
    
    # Generate response with iterative refinement
    response = await model_orchestrator._generate_with_iterative_refinement(prompt, query)
    
    # Check tokenizer was called multiple times (for each stage)
    assert mock_t5_tokenizer.__call__.call_count >= 4
    
    # Check model generated response multiple times
    assert mock_t5_model.generate.call_count >= 4
    
    # Check final response
    assert response == "Comprehensive answer covering all aspects of quantum mechanics and relativity"

@pytest.mark.asyncio
async def test_no_models_available(model_orchestrator):
    """Test behavior when no models are available."""
    # Set up the orchestrator with no models
    model_orchestrator.t5_verified = False
    
    # Test query and context
    query = "What is the main topic?"
    context = "The main topic is artificial intelligence."
    
    # Generate response
    response, metadata = await model_orchestrator.generate_response(query, context)
    
    # Check error response
    assert "encountered an issue" in response.lower()
    assert metadata["primary_model"] == "none" 