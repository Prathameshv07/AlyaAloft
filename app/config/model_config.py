"""
Configuration settings for language models.
"""

from pathlib import Path
from typing import Dict, Any

# Base model paths
MODELS_DIR = Path("models")

# Model-specific paths - focus only on T5 now that we've removed other models
T5_PATH = MODELS_DIR / "flan-t5-base"

# Flan-T5 model configuration with optimized parameters
T5_CONFIG = {
    "max_length": 1024,        # Increased for more complete answers
    "min_length": 64,          # Increased for more detailed answers
    "temperature": 0.6,        # Slightly reduced for more consistent outputs
    "top_p": 0.92,
    "repetition_penalty": 1.3, # Increased to reduce repetition issues
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "num_beams": 4,            # Added beam search for better coherence
    "early_stopping": True     # Stop when viable candidates are found
}

# Enhanced T5 prompt templates for different question types
T5_PROMPT_TEMPLATE = """Answer the following question based only on the provided context. Be comprehensive and complete in your answer:

Context: {context}

Question: {query}

Answer:"""

T5_DEFINITION_TEMPLATE = """Define the following concept based on the provided context. Include a complete explanation with all relevant aspects:

Context: {context}

Define: {query}

Complete definition:"""

T5_EXPLANATION_TEMPLATE = """Explain the following topic using the provided context. Include all important points and be thorough in your explanation:

Context: {context}

Explain: {query}

Thorough explanation:"""

T5_COMPARISON_TEMPLATE = """Compare and contrast the following items based on the provided context. Include all relevant similarities and differences:

Context: {context}

Compare: {query}

Complete comparison:"""

# New advanced template for technical analysis
T5_TECHNICAL_ANALYSIS_TEMPLATE = """Perform a technical analysis of the following topic using the provided context. Include theoretical foundations, implementation details, and practical implications:

Context: {context}

Analyze: {query}

Technical analysis:"""

# New template for step-by-step instructions
T5_STEP_BY_STEP_TEMPLATE = """Provide step-by-step instructions for the following process based on the provided context:

Context: {context}

Process to explain: {query}

Step-by-step instructions:"""

# New template for historical development
T5_HISTORICAL_TEMPLATE = """Describe the historical development of the following topic based on the provided context. Include key events, figures, and turning points:

Context: {context}

Topic: {query}

Historical development:"""

# Templates for extremely complex questions
T5_COMPLEX_ACADEMIC_TEMPLATE = """Provide a scholarly analysis of this complex academic question. Apply advanced theoretical frameworks and incorporate relevant methodologies. Address nuances, competing perspectives, and limitations:

Context: {context}

Complex academic question: {query}

Scholarly analysis:"""

T5_INTERDISCIPLINARY_TEMPLATE = """Analyze this interdisciplinary question by examining multiple domains and their interconnections. Identify cross-domain insights, methodological approaches, and conceptual frameworks:

Context: {context}

Interdisciplinary question: {query}

Cross-domain analysis:"""

T5_THEORETICAL_FRAMEWORK_TEMPLATE = """Create a comprehensive theoretical framework addressing this complex question. Identify core principles, causal mechanisms, boundary conditions, and practical implications:

Context: {context}

Question requiring framework: {query}

Theoretical framework:"""

# Domain-specific advanced templates
T5_NLP_ADVANCED_TEMPLATE = """Provide a technical NLP analysis addressing all linguistic and computational aspects. Include relevant algorithms, models, evaluation metrics, and limitations:

Context: {context}

NLP question: {query}

Technical NLP analysis:"""

T5_COMPUTATIONAL_LINGUISTICS_TEMPLATE = """Analyze this computational linguistics question by addressing both linguistic theory and computational implementation. Include formal representations, algorithms, and empirical validation approaches:

Context: {context}

Computational linguistics question: {query}

Analysis:"""

T5_MULTIMODAL_ANALYSIS_TEMPLATE = """Analyze how this question relates to multimodal processing across text, speech, and other modalities. Address cross-modal integration challenges and technical approaches:

Context: {context}

Multimodal question: {query}

Multimodal analysis:"""

# Standard template for backward compatibility
STANDARD_PROMPT_TEMPLATE = """Answer the following question based only on the provided context. Be comprehensive and complete in your answer:

Context: {context}

Question: {query}

Answer:"""

# Validation template for backward compatibility
VALIDATION_PROMPT_TEMPLATE = """Evaluate if this response accurately answers the question based on the context. Rate the quality from 0 to 1 and suggest improvements if needed:

Context: {context}

Question: {query}

Response to evaluate: {response}

Evaluation (score 0-1, reasons, suggested improvements):"""

# Model pipeline configuration (now focused on T5 only)
PIPELINE_CONFIG = {
    "primary_model": "t5",            # Using T5 as primary model
    "validator_model": "t5",          # Using T5 as validator also
    "fallback_model": "t5",           # T5 as fallback
    "validation_threshold": 0.7,
    "max_validation_attempts": 2,
    "load_all_models_at_startup": False,
    "offload_unused_models": True,
    "use_cpu_offloading": False,      # T5 should fit in memory without offloading
    "chunk_size": 1024,               # Larger chunk size for more context
    "chunk_overlap": 256,             # Significant overlap between chunks
    "max_context_chunks": 4           # Control total context length for T5
}

# Conversation configuration
CONVERSATION_CONFIG = {
    "max_history_length": 5,          # Reduced to fit within T5's context window
    "max_context_chunks": 4,          # Control number of chunks sent to model
    "max_tokens_per_message": 512     # Control message size
}

# Enhanced prompt selection configuration with additional question types
PROMPT_SELECTION_CONFIG = {
    "default": T5_PROMPT_TEMPLATE,
    "definition": T5_DEFINITION_TEMPLATE,
    "explanation": T5_EXPLANATION_TEMPLATE,
    "comparison": T5_COMPARISON_TEMPLATE,
    "technical_analysis": T5_TECHNICAL_ANALYSIS_TEMPLATE,
    "step_by_step": T5_STEP_BY_STEP_TEMPLATE,
    "historical": T5_HISTORICAL_TEMPLATE,
    "complex_academic": T5_COMPLEX_ACADEMIC_TEMPLATE,
    "interdisciplinary": T5_INTERDISCIPLINARY_TEMPLATE,
    "theoretical_framework": T5_THEORETICAL_FRAMEWORK_TEMPLATE,
    "nlp_advanced": T5_NLP_ADVANCED_TEMPLATE,
    "computational_linguistics": T5_COMPUTATIONAL_LINGUISTICS_TEMPLATE,
    "multimodal_analysis": T5_MULTIMODAL_ANALYSIS_TEMPLATE,
    
    # Keywords for identifying question types
    "definition_keywords": ["what is", "define", "meaning of", "concept of", "definition of", "describe what", "term"],
    "explanation_keywords": ["explain", "how does", "describe", "elaborate on", "clarify", "tell me about", "elucidate"],
    "comparison_keywords": ["compare", "difference between", "similarities", "versus", "vs", "differentiate", "how do they differ", "contrast"],
    "technical_analysis_keywords": ["analyze", "technical details", "implementation", "architecture", "under the hood", "how it works", "mechanisms"],
    "step_by_step_keywords": ["how to", "steps for", "procedure", "process for", "instructions", "guide me through", "walkthrough"],
    "historical_keywords": ["history of", "evolution of", "development of", "origins", "timeline", "when was", "historical"],
    "complex_academic_keywords": ["theoretical", "framework", "methodology", "epistemological", "paradigm", "discourse", "conceptual", "critically analyze"],
    "interdisciplinary_keywords": ["interdisciplinary", "cross-domain", "intersection", "multiple fields", "across disciplines", "integrative", "transdisciplinary"],
    "theoretical_framework_keywords": ["theoretical framework", "conceptual model", "theory", "foundational concepts", "principles", "axioms", "theoretical basis"],
    "nlp_advanced_keywords": ["nlp", "natural language processing", "computational linguistics", "language model", "parsing", "semantic analysis", "syntactic"],
    "computational_linguistics_keywords": ["computational linguistics", "grammar formalism", "parsing algorithm", "formal language", "statistical model", "language theory"],
    "multimodal_analysis_keywords": ["multimodal", "cross-modal", "text-to-speech", "speech-to-text", "vision-language", "multi-modal processing", "audio-visual"]
} 