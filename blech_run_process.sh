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
print(f"Looking for log file at: {log_path}")

if not log_path.exists():
    print("Error: blech_process.log not found")
    print("Available files in directory:")
    try:
        for f in pathlib.Path("$DIR").iterdir():
            if f.name.endswith('.log'):
                print(f"  {f.name}")
    except Exception as e:
        print(f"  Could not list files: {e}")

print("Log file found, reading contents...")
try:
    with open(log_path) as f:
        process_log = json.load(f)
    print(f"Successfully loaded log with {len(process_log)} entries")
except Exception as e:
    print(f"Error reading log file: {e}")

incomplete = [e for e, data in process_log.items() if data['status'] == 'attempted']

if incomplete:
    print(f"Error: The following electrodes did not complete successfully: {incomplete}")
else:
    print("All electrodes completed successfully")
EOF

echo "Completion check passed, continuing..."

# Generate rolling threshold grid plot (only if rolling threshold is enabled)
echo "Checking if rolling threshold grid plot should be generated..."
cd "$DIR"
python3 - <<EOF

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_rolling_threshold_grid(rolling_thresh_dir, output_path):
    """Generate a grid plot of rolling thresholds for all electrodes.

    Parameters
    ----------
    rolling_thresh_dir : str
        Directory containing electrode*_rolling_threshold.npz files.
    output_path : str
        Path to save the output figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    import glob

    # Load all threshold files
    files = sorted(
        glob.glob(f'{rolling_thresh_dir}/electrode*_rolling_threshold.npz'))
    if not files:
        print(f"No rolling threshold files found in {rolling_thresh_dir}")
        return None

    # Load data
    data = []
    for f in files:
        npz = np.load(f)
        data.append({
            'times': npz['times'],
            'thresholds': npz['thresholds'],
            'electrode_num': int(npz['electrode_num']),
        })

    n_electrodes = len(data)
    if n_electrodes == 0:
        return None

    # Determine grid size
    n_cols = int(np.ceil(np.sqrt(n_electrodes)))
    n_rows = int(np.ceil(n_electrodes / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15,15),
        sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for i, d in enumerate(data):
        ax = axes[i]
        ax.plot(d['times'], d['thresholds'], linewidth=0.8)
        ax.set_title(f"Electrode {d['electrode_num']:02d}", fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel("Threshold (µV)")
        if i >= n_electrodes - n_cols:
            ax.set_xlabel("Time (s)")

    # Hide unused axes
    for i in range(n_electrodes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Rolling Spike Detection Thresholds", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    return fig

# Check if rolling threshold is enabled in params
# params_path = './params/sorting_params.json'
from glob import glob
params_path = glob('*.params')[0]
use_rolling = False
if os.path.exists(params_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    use_rolling = params.get('use_rolling_threshold', False)
else:
    print(f"Parameters file not found: {params_path}")

if not use_rolling:
    print("Rolling threshold is disabled, skipping grid plot generation")
else:
    rolling_thresh_dir = './QA_output/rolling_thresholds'
    output_path = './QA_output/rolling_threshold_grid.png'

    try:
        if os.path.isdir(rolling_thresh_dir):
            fig = plot_rolling_threshold_grid(rolling_thresh_dir, output_path)
            if fig:
                print(f"Rolling threshold grid plot saved to {output_path}")
            else:
                print("No rolling threshold data found")
        else:
            print(f"Rolling threshold directory not found: {rolling_thresh_dir}")

    except ImportError as e:
        print(f"Error importing plot_rolling_threshold_grid: {e}")
        print("Function may not exist in utils.blech_process_utils")
    except Exception as e:
        print(f"Error generating rolling threshold grid plot: {e}")
EOF
