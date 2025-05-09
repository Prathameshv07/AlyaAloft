#!/usr/bin/env python
"""
Start AlyaAloft with enhanced FLAN-T5 model.

This script launches the AlyaAloft application with the optimized FLAN-T5 model 
utilizing advanced prompting techniques for improved response quality.

Features:
- Domain-specific prompt templates
- Chain-of-thought reasoning for complex questions
- Iterative refinement for comprehensive answers
- 8-bit quantization for CUDA-enabled devices (if bitsandbytes is installed)
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Start AlyaAloft with enhanced FLAN-T5 model")
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to run the server on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-reload", 
        action="store_true", 
        help="Disable automatic reload on code changes"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["ALYA_MODEL"] = "t5"
    os.environ["ALYA_LOG_LEVEL"] = args.log_level
    
    print(f"""
Starting AlyaAloft with enhanced FLAN-T5 model...
Server: http://{args.host}:{args.port}
Log level: {args.log_level}
Auto-reload: {"Disabled" if args.no_reload else "Enabled"}

Enhanced prompting features:
- Domain-specific templates
- Chain-of-thought reasoning
- Iterative refinement for complex questions
- 8-bit quantization (if CUDA available)
    """)
    
    # Run the server
    uvicorn.run(
        "app.main:app", 
        host=args.host, 
        port=args.port, 
        reload=not args.no_reload,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()
