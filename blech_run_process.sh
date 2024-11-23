#!/bin/bash

# Function to show folder selection dialog using zenity
choose_folder() {
    zenity --file-selection --directory --title="Select data folder" 2>/dev/null
}

DIR=$1

# Check if the result log exists, ask whether to overwrite
LOG_FILE="$DIR/results.log"
if [ -f "$LOG_FILE" ]; then
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
