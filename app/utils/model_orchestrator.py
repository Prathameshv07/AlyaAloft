"""
Model orchestrator for managing language models (T5-based) with enhanced prompting.
"""

import asyncio
import gc
import re
import time
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import torch
import os

# Import directly from model_config to avoid confusion with app.config
from app.config.model_config import (
    T5_PATH,
    T5_CONFIG,
    T5_PROMPT_TEMPLATE,
    T5_DEFINITION_TEMPLATE,
    T5_EXPLANATION_TEMPLATE,
    T5_COMPARISON_TEMPLATE,
    PROMPT_SELECTION_CONFIG,
    STANDARD_PROMPT_TEMPLATE,
    VALIDATION_PROMPT_TEMPLATE,
    PIPELINE_CONFIG
)
from app.utils.logging_utils import get_logger
from app.utils.prompt_manager import prompt_manager

logger = get_logger(__name__)

class ModelOrchestrator:
    """Orchestrates T5 model for optimal response generation with enhanced prompt engineering."""
    
    def __init__(self):
        """Initialize the model orchestrator."""
        self.t5_model = None
        self.t5_tokenizer = None
        
        # Model availability/status flags
        self.t5_available = False
        
        # Model verification flags
        self.t5_verified = False
        
        # Currently active model to manage memory
        self.active_model = None
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
            logger.info(f"CUDA is available! Found {cuda_device_count} device(s). Primary device: {cuda_device_name}")
            
            # Get available GPU memory
            try:
                free_memory, total_memory = torch.cuda.mem_get_info()
                free_gb = free_memory / (1024 ** 3)
                total_gb = total_memory / (1024 ** 3)
                logger.info(f"GPU memory: {free_gb:.2f}GB free out of {total_gb:.2f}GB total")
            except Exception as e:
                logger.warning(f"Could not determine GPU memory: {str(e)}")
        else:
            logger.warning("CUDA is not available. Using CPU for inference (this will be slower).")
        
        logger.info(f"Using device for AI models: {self.device}")
    
    async def load_models(self):
        """Verify and load the T5 model."""
        await self._verify_models_health()
        
        try:
            # Load the model if verified
            if self.t5_verified:
                await self._load_t5_model()
                self.active_model = "t5"
                logger.info("T5 model loaded successfully")
            else:
                logger.error("T5 model could not be verified. Please check model files.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    
    async def _verify_models_health(self):
        """Verify the T5 model can be loaded."""
        try:
            model_path = T5_PATH
            
            if not Path(model_path).exists():
                logger.warning(f"Model path does not exist: {model_path}")
                return
            
            logger.info(f"Verifying T5 model...")
            
            # Check for necessary configuration files
            config_exists = Path(model_path / "config.json").exists()
            tokenizer_config_exists = Path(model_path / "tokenizer_config.json").exists()
            model_exists = Path(model_path / "model.safetensors").exists() or any(Path(model_path).glob("*.bin"))
            
            self.t5_verified = config_exists and tokenizer_config_exists and model_exists
            
            if self.t5_verified:
                logger.info("T5 model verified successfully")
            else:
                missing_files = []
                if not config_exists:
                    missing_files.append("config.json")
                if not tokenizer_config_exists:
                    missing_files.append("tokenizer_config.json")
                if not model_exists:
                    missing_files.append("model files (*.bin or *.safetensors)")
                logger.warning(f"T5 model files not found or incomplete. Missing: {', '.join(missing_files)}")
            
        except Exception as e:
            logger.warning(f"Error verifying T5 model: {str(e)}")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory to free resources."""
        if self.device == "cuda":
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache")
    
    async def _unload_all_models(self):
        """Unload all models from memory to free resources."""
        try:
            # Setting model references to None
            self.t5_model = None
            self.t5_tokenizer = None
            
            # Reset availability flags
            self.t5_available = False
            
            # Reset active model
            self.active_model = None
            
            # Clear GPU memory
            self._clear_gpu_memory()
            
            # Additional forced cleanup
            for _ in range(3):
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            logger.info("All models unloaded from memory")
        except Exception as e:
            logger.error(f"Error unloading models: {str(e)}")
    
    def _select_prompt_template(self, query: str) -> str:
        """Select the appropriate prompt template based on query type."""
        query_lower = query.lower()
        
        # Check for definition questions
        for keyword in PROMPT_SELECTION_CONFIG["definition_keywords"]:
            if keyword in query_lower:
                return PROMPT_SELECTION_CONFIG["definition"]
        
        # Check for explanation questions
        for keyword in PROMPT_SELECTION_CONFIG["explanation_keywords"]:
            if keyword in query_lower:
                return PROMPT_SELECTION_CONFIG["explanation"]
        
        # Check for comparison questions
        for keyword in PROMPT_SELECTION_CONFIG["comparison_keywords"]:
            if keyword in query_lower:
                return PROMPT_SELECTION_CONFIG["comparison"]
        
        # Default template
        return PROMPT_SELECTION_CONFIG["default"]
    
    def _preprocess_context(self, context: str) -> str:
        """Preprocess context to be more easily digestible by T5."""
        # Remove excessive whitespace
        context = re.sub(r'\s+', ' ', context).strip()
        
        # Add paragraph breaks for better readability
        context = re.sub(r'\.(?=\s*[A-Z])', '.\n', context)
        
        # Add bullet points for lists
        context = re.sub(r'(?<=\n|\s)(\d+\.|\-|\*)\s+', '\n• ', context)
        
        return context
    
    def _postprocess_response(self, response: str) -> str:
        """Clean up and enhance the model's response."""
        # Remove any unwanted prefixes
        response = re.sub(r'^(answer:|definition:|explanation:|comparison:)\s*', '', response, flags=re.IGNORECASE)
        
        # Ensure response ends with a period if it's a complete sentence
        if response and not response.endswith(('.', '?', '!', ':', ';')):
            last_char = response[-1]
            if last_char.isalpha() or last_char.isdigit():
                response += '.'
        
        # Capitalize first letter if it starts with a lowercase letter
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        return response.strip()
    
    async def generate_response(self, query: str, context: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using the T5 model with optimized prompting.
        
        Args:
            query: User's query
            context: Document context
            
        Returns:
            Tuple of (response, metadata)
        """
        metadata = {
            "primary_model": "none",
            "validation_model": "none",
            "validation_score": 0.0,
            "needed_correction": False,
            "prompt_template": "default",
            "context_length": len(context),
            "query_length": len(query),
            "query_complexity": 0.0
        }
        
        # Free any existing models to start with a clean memory state
        await self._unload_all_models()
        
        if self.t5_verified:
            try:
                # Load T5 model
                logger.info("Loading T5 model for response generation")
                await self._load_t5_model()
                self.active_model = "t5"
                
                # Check if model was successfully loaded
                if not self.t5_available:
                    logger.error("Model loading reported success but model is not available")
                    return "I'm sorry, I encountered an issue loading the language model. Please try again later.", metadata
                
                # Analyze query complexity using the prompt manager
                complexity = prompt_manager.detect_complexity(query)
                metadata["query_complexity"] = complexity["overall"]
                
                # Generate enhanced prompt using the prompt manager
                structured_prompt = prompt_manager.generate_structured_prompt(query, context)
                
                # Apply chain-of-thought reasoning for complex queries
                if complexity["overall"] > 0.6 or len(query.split()) > 10 or "explain" in query.lower() or "why" in query.lower():
                    structured_prompt = prompt_manager.apply_chain_of_thought(structured_prompt)
                    metadata["prompt_template"] = "chain_of_thought"
                else:
                    metadata["prompt_template"] = "structured_enhanced"
                
                # For extremely complex queries, use iterative refinement
                if complexity["overall"] > 0.8:
                    logger.info("Using iterative refinement for complex query")
                    response = await self._generate_with_iterative_refinement(structured_prompt, query)
                    metadata["primary_model"] = "t5_iterative"
                else:
                    # Standard enhanced generation
                    response = await self._generate_with_t5_enhanced(structured_prompt)
                metadata["primary_model"] = "t5"
                
                # Clean and enhance the response
                response = prompt_manager.post_process_response(response, query)
                
                # For short responses to complex questions, try to get more detail
                if complexity["overall"] > 0.7 and len(response.split()) < 50:
                    logger.info("Response too short for complex query, attempting to expand")
                    expand_prompt = f"{structured_prompt}\n\nEnsure your answer is detailed and comprehensive. The previous answer was: {response}\n\nExpanded answer:"
                    expanded_response = await self._generate_with_t5_enhanced(expand_prompt)
                    if len(expanded_response.split()) > len(response.split()) * 1.5:
                        response = prompt_manager.post_process_response(expanded_response, query)
                        metadata["expanded_response"] = True
                
                # Unload model
                logger.info("Unloading T5 model")
                await self._unload_all_models()
                
                return response, metadata
            except Exception as e:
                logger.error(f"Error with T5 model: {str(e)}")
                await self._unload_all_models()
        
        # If all fails, return a generic error response
        logger.error("Failed to generate response with T5 model")
        return "I'm sorry, I encountered an issue processing your question. Please try rephrasing or ask another question.", metadata
    
    async def _load_t5_model(self):
        """Load the T5 model."""
        try:
            # Clear GPU memory before loading
            self._clear_gpu_memory()
            
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            # Load tokenizer first
            logger.info("Loading T5 tokenizer...")
            self.t5_tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                str(T5_PATH)
            )
            
            # Then load model
            logger.info(f"Loading T5 model on device: {self.device}")
            
            # Load model in 8-bit if on CUDA to save memory
            if self.device == "cuda":
                try:
                    from bitsandbytes.nn import Linear8bitLt
                    from transformers import BitsAndBytesConfig
                    logger.info("Using 8-bit quantization for T5 model")
                    
                    # Create quantization config instead of using load_in_8bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                    
                    # Load with device_map but without calling .to()
                    self.t5_model = await asyncio.to_thread(
                        lambda: AutoModelForSeq2SeqLM.from_pretrained(
                            str(T5_PATH),
                            device_map="auto",
                            quantization_config=quantization_config
                        )
                    )
                except ImportError:
                    logger.info("bitsandbytes not available, loading model without quantization")
                    self.t5_model = await asyncio.to_thread(
                        lambda: AutoModelForSeq2SeqLM.from_pretrained(
                            str(T5_PATH)
                        ).to(self.device)
                    )
            else:
                # CPU mode - no quantization
                self.t5_model = await asyncio.to_thread(
                    lambda: AutoModelForSeq2SeqLM.from_pretrained(
                        str(T5_PATH)
                    ).to(self.device)
                )
            
            self.t5_available = True
            logger.info("T5 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading T5 model: {str(e)}")
            raise
    
    async def _generate_with_t5_enhanced(self, prompt: str) -> str:
        """Generate response using T5 model with the enhanced prompt."""
        try:
            # Create inputs with tokenizer
            inputs = await asyncio.to_thread(
                self.t5_tokenizer,
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Use full context length for enhanced prompts
            )
            
            # Always move inputs to CUDA if available - prevents the warning
            if self.device == "cuda":
                # Force all inputs to CUDA regardless of model config
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Add generation parameters optimized for factual accuracy
            generation_params = dict(T5_CONFIG)
            generation_params.update({
                "temperature": 0.3,  # Lower temperature for more predictable outputs
                "num_beams": 5,      # More beams for better search
                "early_stopping": True,
                "repetition_penalty": 1.3,  # Stronger repetition penalty
                "length_penalty": 1.0,      # Balanced length penalty
                "no_repeat_ngram_size": 3   # Avoid repeating 3-grams
            })
            
            # Generate with optimized parameters
            outputs = await asyncio.to_thread(
                lambda: self.t5_model.generate(
                    **inputs,
                    **generation_params
                )
            )
            
            # Decode response
            response = await asyncio.to_thread(
                self.t5_tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            )
            
            return response
        except Exception as e:
            logger.error(f"Error generating with enhanced T5 prompt: {str(e)}")
            # Fallback to regular generation method with minimal parameters
            try:
                # Create a much simpler prompt for fallback
                simple_prompt = f"Answer this question based on the context: {prompt.split('Question:')[-1]}"
                
                inputs = await asyncio.to_thread(
                    self.t5_tokenizer,
                    simple_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Always move to CUDA if available
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                outputs = await asyncio.to_thread(
                    lambda: self.t5_model.generate(
                        **inputs,
                        max_length=256,
                        min_length=32,
                        do_sample=False
                    )
                )
                
                response = await asyncio.to_thread(
                    self.t5_tokenizer.decode,
                    outputs[0],
                    skip_special_tokens=True
                )
                
                return response
            except Exception as e2:
                logger.error(f"Error in fallback generation: {str(e2)}")
                return "I couldn't generate a proper response. Please try a different question."
    
    async def _generate_with_iterative_refinement(self, prompt: str, query: str) -> str:
        """
        Generate a response for complex questions using iterative refinement.
        This approaches the question in stages to build a more comprehensive answer.
        
        Args:
            prompt: The structured prompt
            query: The original user query
        
        Returns:
            Refined response
        """
        logger.info("Starting iterative refinement for complex question")
        
        # Stage 1: Get key aspects of the question
        key_aspects_prompt = f"""Identify the key aspects that need to be addressed to fully answer this question:

Question: {query}

List the main components that a comprehensive answer must address:"""
        
        try:
            logger.info("Stage 1: Identifying key aspects")
            
            # Generate key aspects
            inputs = await asyncio.to_thread(
                self.t5_tokenizer,
                key_aspects_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = await asyncio.to_thread(
                lambda: self.t5_model.generate(
                    **inputs,
                    max_length=256,
                    min_length=64,
                    temperature=0.3,
                    num_beams=4,
                    early_stopping=True
                )
            )
            
            key_aspects = await asyncio.to_thread(
                self.t5_tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            )
            
            logger.info(f"Key aspects identified: {key_aspects[:100]}...")
            
            # Stage 2: Generate initial answer with the enhanced prompt
            logger.info("Stage 2: Generating initial answer")
            initial_answer = await self._generate_with_t5_enhanced(prompt)
            
            # Stage 3: Evaluate completeness and identify gaps
            evaluation_prompt = f"""Evaluate if this answer completely addresses the question and identify any gaps:

Question: {query}

Key aspects to address:
{key_aspects}

Current answer:
{initial_answer}

What aspects are missing or need more detail?"""
            
            logger.info("Stage 3: Evaluating completeness")
            
            inputs = await asyncio.to_thread(
                self.t5_tokenizer,
                evaluation_prompt,
                return_tensors="pt", 
                truncation=True,
                max_length=768
            )
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = await asyncio.to_thread(
                lambda: self.t5_model.generate(
                    **inputs,
                    max_length=256,
                    min_length=32,
                    temperature=0.4,
                    num_beams=3
                )
            )
            
            gaps = await asyncio.to_thread(
                self.t5_tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            )
            
            logger.info(f"Evaluation identified gaps: {gaps[:100]}...")
            
            # Stage 4: Generate refined answer addressing the gaps
            refinement_prompt = f"""Generate a comprehensive answer that addresses all aspects of this question:

Question: {query}

Initial answer:
{initial_answer}

Missing aspects to address:
{gaps}

Provide a complete, improved answer:"""
            
            logger.info("Stage 4: Generating refined answer")
            
            inputs = await asyncio.to_thread(
                self.t5_tokenizer,
                refinement_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = await asyncio.to_thread(
                lambda: self.t5_model.generate(
                    **inputs,
                    max_length=1024,
                    min_length=128,
                    temperature=0.3,
                    num_beams=5,
                    length_penalty=1.2,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            )
            
            refined_answer = await asyncio.to_thread(
                self.t5_tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            )
            
            logger.info("Iterative refinement process completed")
            
            return refined_answer
            
        except Exception as e:
            logger.error(f"Error in iterative refinement: {str(e)}")
            # Fall back to standard generation
            return await self._generate_with_t5_enhanced(prompt)

# Create singleton instance
model_orchestrator = ModelOrchestrator() 