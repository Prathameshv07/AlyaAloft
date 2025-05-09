"""
AI response generator using T5 language model with advanced prompt engineering.
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import app_config module directly
from app import app_config as config
from app.utils.logging_utils import log_function_call, get_logger
from app.utils.model_orchestrator import model_orchestrator

logger = get_logger(__name__)

class AIResponseGenerator:
    """
    Class for generating AI responses using the T5 model with optimized prompting.
    """
    
    def __init__(self):
        """Initialize the AI response generator."""
        # Initialize flags
        self.models_loaded = False
        self.load_attempts = 0
        self.max_load_attempts = 3
        self.load_retry_delay = 5  # seconds
    
    @log_function_call
    async def load_models(self):
        """Load T5 model through the orchestrator."""
        while self.load_attempts < self.max_load_attempts:
            try:
                if not self.models_loaded:
                    await model_orchestrator.load_models()
                    self.models_loaded = True
                    logger.info("T5 model loaded successfully")
                    return True
            except Exception as e:
                self.load_attempts += 1
                logger.error(f"Error loading T5 model (attempt {self.load_attempts}/{self.max_load_attempts}): {str(e)}")
                if self.load_attempts < self.max_load_attempts:
                    await asyncio.sleep(self.load_retry_delay)
                else:
                    logger.error("Failed to load T5 model after maximum attempts")
                    return False
        return False
    
    @log_function_call
    async def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response to a query given context using the T5 model.
        
        Args:
            query: User's query
            context: Document context retrieved from vector search
            
        Returns:
            Generated response as a string
        """
        try:
            # Ensure models are loaded
            if not self.models_loaded:
                if not await self.load_models():
                    return await self._generate_fallback_response(query, context)
            
            # Generate response using the T5 orchestrator
            response, metadata = await model_orchestrator.generate_response(query, context)
            
            # Log the template usage
            logger.info(f"Response generated using template '{metadata.get('prompt_template', 'default')}' "
                       f"with context length {metadata.get('context_length', len(context))} and query length {metadata.get('query_length', len(query))}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return await self._generate_fallback_response(query, context)
    
    async def _generate_fallback_response(self, query: str, context: str) -> str:
        """
        Generate a basic response when the T5 model fails.
        
        Args:
            query: User's query
            context: Document context
            
        Returns:
            Fallback response as a string
        """
        try:
            # Create a very simple response based on the query and context
            # This doesn't use any ML models, just basic text processing
            
            # Extract a relevant section of the context based on simple keyword matching
            query_words = set(query.lower().split())
            
            # Remove common words
            common_words = {"what", "is", "the", "a", "an", "and", "or", "of", "in", "to", "how", "why", "when", "where"}
            query_words = query_words - common_words
            
            # Find sentences that contain query words
            import re
            sentences = re.split(r'(?<=[.!?])\s+', context)
            
            relevant_sentences = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if any(word in sentence_words for word in query_words):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                response = " ".join(relevant_sentences[:3])  # Limit to first 3 matches
                logger.info("Generated fallback response using keyword matching")
                return response
            
            # If no relevant sentences found, return a generic response
            logger.warning("No relevant sentences found for fallback response")
            return "I found information related to your query in the document, but I'm unable to generate a detailed response at this time. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return "I apologize, but I'm currently experiencing technical difficulties. Please try again later."

# Create singleton instance
ai_response_generator = AIResponseGenerator() 