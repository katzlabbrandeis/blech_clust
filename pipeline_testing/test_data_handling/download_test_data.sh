# Google Drive link : https://drive.google.com/file/d/1EcpUIqp81h3J89-6dEEueeULqBlKW5a7/view?usp=sharing

SCRIPT_PATH=$(realpath ${BASH_SOURCE[0]})
# Find path to neuRecommend as parent directory of test.sh
DIR_PATH=$(dirname ${SCRIPT_PATH})

#pip install gdown
LINK_TO_DATA=(
    1EcpUIqp81h3J89-6dEEueeULqBlKW5a7
    1aU2DWHhbVB3rDujbF4KRX9QLA1LlfpU3
)

DATA_DIR_NAMES=(
    KM45_5tastes_210620_113227_new
    eb24_behandephys_11_12_24_241112_114659_copy
)

DATA_SUP_DIR=${DIR_PATH}/test_data
# If DATA_DIR does not exist, create it
if [ ! -d "$DATA_SUP_DIR" ]; then
    mkdir $DATA_SUP_DIR
fi

for i in {0..1}
do
    DATA_DIR=${DATA_SUP_DIR}/${DATA_DIR_NAMES[i]}
    # If DATA_DIR does not exist, create it
    if [ ! -d "$DATA_DIR" ]; then
        echo Test dataset ${DATA_DIR_NAMES[i]} does not exist. Creating directory
        mkdir $DATA_DIR
        echo Downloading data to ${DATA_DIR}
        # Use gdown to download the model to DATA_DIR
        # gdown -O <output_file> <link_to_file>
        # -O option specifies the output file name
        gdown ${LINK_TO_DATA[i]} -O $DATA_DIR/ 

        # Unzip the downloaded file
        # unzip <zip_file> -d <destination_folder>
        FILENAME=$(basename $(ls $DATA_DIR/*.zip))
        # Remove the .zip extension
        unzip $DATA_DIR/$FILENAME -d $DATA_DIR/
        # Remove the .zip file
        rm $DATA_DIR/$FILENAME
        mv $DATA_DIR/*/* $DATA_DIR/
        rm -r $DATA_DIR/*/
    else
        echo Test dataset ${DATA_DIR_NAMES[i]} already exists. Skipping download
    fi
done
