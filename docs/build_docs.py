#!/usr/bin/env python3
"""
Helper script to build documentation.
Adds parent directory to sys.path so modules can be imported.
"""
import subprocess
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Now run quartodoc build
result = subprocess.run(['quartodoc', 'build'], cwd=Path(__file__).parent)
sys.exit(result.returncode)
