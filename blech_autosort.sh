
if [ "$2" == "--force" ]; then
    FORCE=1
else
    FORCE=0
fi

DIR=$1
if [ ! -d $DIR ]; then
    echo "Directory $DIR does not exist"
    exit 1
fi

SCRIPT_DIR=$0
BLECH_DIR=$(dirname $SCRIPT_DIR)

# Check that sorting params in data_dir has:
# "clustering_params": {
#   "auto_params": {
#       "auto_cluster": true,
#       "auto_post_process": true
#       ...
# Look for .params file in data directory
sort_params=$(find $DIR -maxdepth 1 -name "*.params" | head -n 1)
if [ -z "$sort_params" ]; then
    echo "No .params file found in $DIR"
    exit 1
fi

# Check
AUTO_PARAMS=$(cat $sort_params | grep auto_cluster | grep true)
AUTO_POST_PROCESS=$(cat $sort_params | grep auto_post_process | grep true)

if [ -z "$AUTO_PARAMS" ]; then
    echo "auto_cluster is not set to true in $sort_params"
    exit 1
fi

if [ -z "$AUTO_POST_PROCESS" ]; then
    echo "auto_post_process is not set to true in $sort_params"
    exit 1
fi

# Check that waveform_classifier_params.json in data_dir has:
# 'use_neuRecommend': true
# 'use_classifier': true

waveform_params=$DIR/waveform_classifier_params.json
if [ ! -f $waveform_params ]; then
    echo "File $waveform_params does not exist in data directory"
    echo "Please ensure waveform_classifier_params.json is present in $DIR"
    exit 1
fi

USE_NEURECOMMEND=$(cat $waveform_params | grep use_neuRecommend | grep true)
USE_CLASSIFIER=$(cat $waveform_params | grep use_classifier | grep true)

if [ -z "$USE_NEURECOMMEND" ]; then
    echo "use_neuRecommend is not set to true in $waveform_params"
    exit 1
fi

if [ -z "$USE_CLASSIFIER" ]; then
    echo "use_classifier is not set to true in $waveform_params"
    exit 1
fi


#============================================================
if [ $FORCE -eq 1 ]; then
    bash blech_clust_pre.sh $DIR --force
else
    bash blech_clust_pre.sh $DIR
fi

echo === Post Process ===
if [ $FORCE -eq 1 ]; then
    python blech_post_process.py $DIR --delete-existing
else
    python blech_post_process.py $DIR
fi

bash blech_clust_post.sh $DIR
