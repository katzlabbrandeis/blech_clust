#!/bin/bash

# Script to run blech_autosort.sh on multiple directories listed in a file
# Usage: bash blech_autosort_batch.sh <file_path> [--force]

show_usage() {
    echo "Usage: bash blech_autosort_batch.sh <file_path> [--force]"
    echo ""
    echo "Arguments:"
    echo "  file_path    Path to a text file containing directory paths (one per line)"
    echo "  --force      Optional flag to force processing (passed to blech_autosort.sh)"
    echo ""
    echo "Example:"
    echo "  bash blech_autosort_batch.sh directories.txt"
    echo "  bash blech_autosort_batch.sh directories.txt --force"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    show_usage
fi

FILE_PATH=$1

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' does not exist"
    exit 1
fi

# Check if file is readable
if [ ! -r "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' is not readable"
    exit 1
fi

# Check if file is empty
if [ ! -s "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' is empty"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$0")

# Check if blech_autosort.sh exists
if [ ! -f "$SCRIPT_DIR/blech_autosort.sh" ]; then
    echo "Error: blech_autosort.sh not found in $SCRIPT_DIR"
    exit 1
fi

# Check for --force flag
FORCE_FLAG=""
if [ "$2" == "--force" ]; then
    FORCE_FLAG="--force"
    echo "Running in force mode"
fi

# Process each directory in the file
echo "Processing directories from: $FILE_PATH"
echo "=========================================="

LINE_NUM=0
SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r DIR || [ -n "$DIR" ]; do
    LINE_NUM=$((LINE_NUM + 1))
    
    # Skip empty lines
    if [ -z "$DIR" ]; then
        continue
    fi
    
    # Skip lines starting with # (comments)
    if [[ "$DIR" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Trim whitespace
    DIR=$(echo "$DIR" | xargs)
    
    echo ""
    echo "[$LINE_NUM] Processing: $DIR"
    echo "----------------------------------------"
    
    # Check if directory exists
    if [ ! -d "$DIR" ]; then
        echo "Warning: Directory '$DIR' does not exist, skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Run blech_autosort.sh
    if [ -n "$FORCE_FLAG" ]; then
        bash "$SCRIPT_DIR/blech_autosort.sh" "$DIR" --force
    else
        bash "$SCRIPT_DIR/blech_autosort.sh" "$DIR"
    fi
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $DIR"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "✗ Failed to process: $DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
done < "$FILE_PATH"

echo ""
echo "=========================================="
echo "Batch processing complete"
echo "Total directories processed: $((SUCCESS_COUNT + FAIL_COUNT))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "=========================================="

# Exit with error if any directory failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
