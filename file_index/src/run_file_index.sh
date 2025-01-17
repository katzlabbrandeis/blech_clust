#!/bin/bash

# Exit on error
set -e

echo "Starting file indexing pipeline..."

# 1. Generate basic summaries
echo "Generating basic summaries..."
python3 generate_basic_summaries.py
if [ $? -ne 0 ]; then
    echo "Error generating basic summaries"
    exit 1
fi

# 2. Update docstrings with LLM
echo "Updating docstrings with LLM..."
python3 llm_update_docstrings.py -y
if [ $? -ne 0 ]; then
    echo "Error updating docstrings"
    exit 1
fi

# 3. Merge docstrings
echo "Merging docstrings..."
python3 merge_docstrings.py
if [ $? -ne 0 ]; then
    echo "Error merging docstrings"
    exit 1
fi

echo "File indexing pipeline completed successfully!"
