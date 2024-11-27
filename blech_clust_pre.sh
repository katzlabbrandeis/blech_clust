DIR=$1
echo Running Blech Clust 
python blech_clust.py $DIR &&
echo Running Common Average Reference 
python blech_common_avg_reference.py $DIR &&
echo Running Jetstream Bash 
bash blech_run_process.sh $DIR
