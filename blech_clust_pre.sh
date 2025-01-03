# Detect --force flag for forcing blech_clust.py and blech_run_process.sh to run
#

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

echo Running Blech Clust 
if [ $FORCE -eq 1 ]; then
    echo "Forcing blech_clust.py to run"
    python blech_clust.py $DIR --force_run 
else
    python blech_clust.py $DIR
fi

echo Running Common Average Reference 
python blech_common_avg_reference.py $DIR &&

echo Running Jetstream Bash 
if [ $FORCE -eq 1 ]; then
    echo "Forcing blech_run_process.sh to run"
    bash blech_run_process.sh $DIR --delete-log
else
    bash blech_run_process.sh $DIR
fi
