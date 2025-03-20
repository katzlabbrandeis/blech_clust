SRC_DIR=$(dirname "$0")
cd $SRC_DIR
DIR=$1
echo === EMG Pipeline Starting ===
echo === EMG Filter ===
python emg_filter.py $DIR &&
echo === EMG Freq Setup ===
python emg_freq_setup.py $DIR &&
echo === Bash Parallel ====
bash blech_emg_jetstream_parallel.sh $DIR &&
echo === EMG Freq Post Process ===
python emg_freq_post_process.py $DIR &&
echo === EMG Freq Plot ===
python emg_freq_plot.py $DIR
echo === EMG Pipeline Done ===
