"""
Configuration file for pytest.
"""
import os
import sys

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
