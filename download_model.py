#!/usr/bin/env python
"""
Redirects to the T5 model downloader script for backward compatibility.
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    print("This script has been renamed. Redirecting to the T5 model downloader...")
    
    # Get the path to the new script
    script_path = Path(__file__).parent / "scripts" / "download_t5_model.py"
    
    if not script_path.exists():
        print(f"Error: Could not find the T5 model downloader at {script_path}")
        sys.exit(1)
    
    # Forward all arguments to the new script
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 