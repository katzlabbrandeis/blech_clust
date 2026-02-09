#!/bin/bash
# Download test data from Google Drive using configuration from test_config.json

SCRIPT_PATH=$(realpath ${BASH_SOURCE[0]})
DIR_PATH=$(dirname ${SCRIPT_PATH})
CONFIG_PATH="${DIR_PATH}/../test_config.json"

# Check if jq is available, otherwise use python to parse JSON
if command -v jq &> /dev/null; then
    DATA_SUP_DIR=$(jq -r '.test_data_dir' "$CONFIG_PATH")
    # Get dataset names and gdrive IDs as arrays
    readarray -t DATA_DIR_NAMES < <(jq -r '.datasets[].name' "$CONFIG_PATH")
    readarray -t LINK_TO_DATA < <(jq -r '.datasets[].gdrive_id' "$CONFIG_PATH")
else
    # Fallback to python for JSON parsing
    DATA_SUP_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['test_data_dir'])")
    readarray -t DATA_DIR_NAMES < <(python3 -c "import json; [print(d['name']) for d in json.load(open('$CONFIG_PATH'))['datasets'].values()]")
    readarray -t LINK_TO_DATA < <(python3 -c "import json; [print(d['gdrive_id']) for d in json.load(open('$CONFIG_PATH'))['datasets'].values()]")
fi

# Expand ~ in path
DATA_SUP_DIR="${DATA_SUP_DIR/#\~/$HOME}"

# Create data directory if it doesn't exist
if [ ! -d "$DATA_SUP_DIR" ]; then
    mkdir -p $DATA_SUP_DIR
fi

# Download each dataset
for i in "${!DATA_DIR_NAMES[@]}"
do
    DATA_DIR=${DATA_SUP_DIR}/${DATA_DIR_NAMES[i]}
    if [ ! -d "$DATA_DIR" ]; then
        echo "Test dataset ${DATA_DIR_NAMES[i]} does not exist. Creating directory"
        mkdir $DATA_DIR
        echo "Downloading data to ${DATA_DIR}"
        gdown ${LINK_TO_DATA[i]} -O $DATA_DIR/

        # Unzip the downloaded file
        FILENAME=$(basename $(ls $DATA_DIR/*.zip))
        unzip $DATA_DIR/$FILENAME -d $DATA_DIR/
        rm $DATA_DIR/$FILENAME
        mv $DATA_DIR/*/* $DATA_DIR/
        rm -r $DATA_DIR/*/
    else
        echo "Test dataset ${DATA_DIR_NAMES[i]} already exists. Skipping download"
    fi
done
