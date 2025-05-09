"""
Configuration settings export.
"""

# Import all configuration variables to make them available
from app.config.model_config import (
    MODELS_DIR,
    T5_PATH,
    T5_CONFIG,
    PIPELINE_CONFIG, CONVERSATION_CONFIG,
    STANDARD_PROMPT_TEMPLATE, VALIDATION_PROMPT_TEMPLATE,
    T5_PROMPT_TEMPLATE, T5_DEFINITION_TEMPLATE, T5_EXPLANATION_TEMPLATE, T5_COMPARISON_TEMPLATE,
    T5_TECHNICAL_ANALYSIS_TEMPLATE, T5_STEP_BY_STEP_TEMPLATE, T5_HISTORICAL_TEMPLATE,
    PROMPT_SELECTION_CONFIG
)

# Note: This file ensures the config directory is a proper Python package. 