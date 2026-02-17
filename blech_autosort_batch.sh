#!/bin/bash

echo "=========================================="
echo "⚠️  WARNING: EXPERIMENTAL SCRIPT"
echo "=========================================="
echo "This script has not been fully tested."
echo "It automates the full spike sorting pipeline"
echo "with auto-clustering enabled."
echo ""
echo "Please review outputs carefully and report"
echo "any issues to the repository maintainers."
echo "=========================================="
echo ""

# Script to run blech_autosort.sh on multiple directories
# Usage: bash blech_autosort_batch.sh <file_path|directory1> [directory2 ...] [--force]

show_usage() {
    echo "Usage: bash blech_autosort_batch.sh <file_path|directory1> [directory2 ...] [--force]"
    echo ""
    echo "Arguments:"
    echo "  file_path    Path to a text file containing directory paths (one per line)"
    echo "  directory1   First directory path to process"
    echo "  directory2   Additional directory paths to process (optional)"
    echo "  --force      Optional flag to force processing (passed to blech_autosort.sh)"
    echo ""
    echo "Examples:"
    echo "  bash blech_autosort_batch.sh directories.txt"
    echo "  bash blech_autosort_batch.sh directories.txt --force"
    echo "  bash blech_autosort_batch.sh /path/to/dir1 /path/to/dir2"
    echo "  bash blech_autosort_batch.sh /path/to/dir1 /path/to/dir2 --force"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    show_usage
fi

# Determine input mode: file vs directories
FIRST_ARG=$1
INPUT_MODE="file"

# Check if first argument is a file (existing behavior)
if [ -f "$FIRST_ARG" ]; then
    FILE_PATH=$FIRST_ARG
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
else
    # Treat arguments as directories
    INPUT_MODE="directories"
    DIRECTORIES=()

    # Parse all arguments until we find --force or reach the end
    for arg in "$@"; do
        if [ "$arg" == "--force" ]; then
            break
        fi
        DIRECTORIES+=("$arg")
    done

    # Check if at least one directory was provided
    if [ ${#DIRECTORIES[@]} -eq 0 ]; then
        echo "Error: No directories provided"
        show_usage
    fi
fi

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$0")

# Check if blech_autosort.sh exists
if [ ! -f "$SCRIPT_DIR/blech_autosort.sh" ]; then
    echo "Error: blech_autosort.sh not found in $SCRIPT_DIR"
    exit 1
fi

# Function to process a single directory
process_directory() {
    local DIR="$1"
    local LINE_NUM="$2"

    echo ""
    echo "[$LINE_NUM] Processing: $DIR"
    echo "----------------------------------------"

    # Check if directory exists
    if [ ! -d "$DIR" ]; then
        echo "Warning: Directory '$DIR' does not exist, skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_DIRS+=("$DIR")
        return
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
        SUCCESS_DIRS+=("$DIR")
    else
        echo "✗ Failed to process: $DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_DIRS+=("$DIR")
    fi
}

# Check for --force flag
FORCE_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        FORCE_FLAG="--force"
        echo "Running in force mode"
        break
    fi
done

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0
SUCCESS_DIRS=()
FAIL_DIRS=()

# Process directories based on input mode
if [ "$INPUT_MODE" == "file" ]; then
    echo "Processing directories from: $FILE_PATH"
    echo "=========================================="

    LINE_NUM=0
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

        # Process this directory
        process_directory "$DIR" "$LINE_NUM"
    done < "$FILE_PATH"
else
    echo "Processing directories from command line arguments"
    echo "=========================================="

    for i in "${!DIRECTORIES[@]}"; do
        DIR="${DIRECTORIES[$i]}"
        LINE_NUM=$((i + 1))
        # Process this directory
        process_directory "$DIR" "$LINE_NUM"
    done
fi

echo ""
echo "=========================================="
echo "Batch processing complete"
echo "Total directories processed: $((SUCCESS_COUNT + FAIL_COUNT))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "=========================================="

# List successful directories
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "✓ Successfully processed directories:"
    for dir in "${SUCCESS_DIRS[@]}"; do
        echo "  $dir"
    done
fi

# List failed directories
if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "✗ Failed to process directories:"
    for dir in "${FAIL_DIRS[@]}"; do
        echo "  $dir"
    done
fi

echo ""
echo "=========================================="

# Exit with error if any directory failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
