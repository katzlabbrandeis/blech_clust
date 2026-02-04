#!/bin/bash

# Function to show folder selection dialog using zenity
choose_folder() {
    zenity --file-selection --directory --title="Select data folder" 2>/dev/null
}

# Parse arguments
DELETE_LOG=false
DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --delete-log)
            DELETE_LOG=true
            shift
            ;;
        *)
            DIR="$1"
            shift
            ;;
    esac
done

# Check if the result log exists, handle based on delete-log flag
LOG_FILE="$DIR/results.log"
if [ -f "$LOG_FILE" ]; then
    if [ "$DELETE_LOG" = true ]; then
        echo "Forcing deletion of existing log"
        rm "$LOG_FILE"
    else
        while true; do
            read -p "results.log detected, overwrite existing log? ([y]/n) :: " yn
            yn=${yn:-y}  # Default to 'y' if no input is provided
            case $yn in
                [Yy]* ) echo "Overwriting existing log";rm "$LOG_FILE"; break;;
                [Nn]* ) echo "Using existing log"; break;;
                * ) echo "Please answer yes or no.";;
            esac
        done
    fi
fi


# Check if the required file exists
required_file="./params/waveform_classifier_params.json"
if [ ! -f "$required_file" ]; then
    echo "=== Waveform Classifier Params file not found. ==="
    echo "==> Please copy [[ ./params/_templates/waveform_classifier_params.json ]] to [[ $required_file ]] and update as needed."
    exit 1
fi

# Check if DIR exists and is a directory
if [ ! -d "$DIR" ]; then
    echo "Directory does not exist or was not provided. Please identify your data folder."
    DIR=$(choose_folder)
    if [ -z "$DIR" ]; then
        echo "No folder selected. Exiting."
        exit 1
    fi
fi

# Start RAM monitoring in background
python3 utils/ram_monitor.py "$DIR" &
RAM_MONITOR_PID=$!

echo "Processing $DIR"
for i in {1..10}; do
    echo Retry $i
    bash $DIR/temp/blech_process_parallel.sh
done

# Kill RAM monitor when done
kill $RAM_MONITOR_PID

# Check all logs for completion status
echo "Checking completion status of all electrodes..."
python3 - <<EOF
import json
import sys
import pathlib

log_path = pathlib.Path("$DIR") / 'blech_process.log'
if not log_path.exists():
    print("Error: blech_process.log not found")
    sys.exit(1)

with open(log_path) as f:
    process_log = json.load(f)

incomplete = [e for e, data in process_log.items() if data['status'] == 'attempted']

if incomplete:
    print(f"Error: The following electrodes did not complete successfully: {incomplete}")
    sys.exit(1)
else:
    print("All electrodes completed successfully")
EOF

# Generate rolling threshold grid plot
echo "Generating rolling threshold grid plot..."
cd "$DIR"
python3 - <<EOF
import sys
sys.path.insert(0, '$(dirname "$0")')
from utils.blech_process_utils import plot_rolling_threshold_grid
import os

rolling_thresh_dir = './QA_output/rolling_thresholds'
output_path = './QA_output/rolling_threshold_grid.png'

if os.path.isdir(rolling_thresh_dir):
    fig = plot_rolling_threshold_grid(rolling_thresh_dir, output_path)
    if fig:
        print(f"Rolling threshold grid plot saved to {output_path}")
    else:
        print("No rolling threshold data found")
else:
    print(f"Rolling threshold directory not found: {rolling_thresh_dir}")
EOF
