DIR=$1
if [ ! -d $DIR ]; then
    echo "Directory $DIR does not exist"
    exit 1
fi

SCRIPT_DIR=$0
BLECH_DIR=$(dirname $SCRIPT_DIR)
echo === Make Arrays ===
python blech_make_arrays.py $DIR &&
echo === Quality Assurance ===
bash blech_run_QA.sh $DIR &&
echo === Units Plot ===
python blech_units_plot.py $DIR &&
echo === Get unit characteristics ===
python blech_units_characteristics.py $DIR &&
echo === Done ===
