"""
blech_clust - Neural data processing and clustering pipeline

This package provides tools for processing and clustering neural recording data
from multi-electrode arrays. It includes functionality for spike sorting,
clustering, and analysis of neural data.

The package automatically checks for updates on initialization unless disabled
in the configuration.
"""

# Import the auto-update functionality
from utils.blech_utils import check_for_repo_updates

# Perform auto-update check on import
try:
    check_for_repo_updates()
except Exception as e:
    # Silently handle any errors during update check to avoid breaking imports
    pass

__version__ = "1.0.0"
__author__ = "Katz Lab, Brandeis University"