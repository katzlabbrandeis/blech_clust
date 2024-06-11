#!/bin/bash

# Function to show folder selection dialog using zenity
choose_folder() {
    zenity --file-selection --directory --title="Select data folder" 2>/dev/null
}

DIR=$1

# Check if DIR exists and is a directory
if [ ! -d "$DIR" ]; then
    echo "Directory does not exist or was not provided. Please identify your data folder."
    DIR=$(choose_folder)
    if [ -z "$DIR" ]; then
        echo "No folder selected. Exiting."
        exit 1
    fi
fi

echo "Processing $DIR"
for i in {1..10}; do
    echo Retry $i
    bash "$DIR/temp/blech_process_parallel.sh"
done
