DIR=$1
BLECH_DIR=$HOME/Desktop/blech_clust
echo === Make Arrays ===
python $BLECH_DIR/blech_make_arrays.py $DIR &&
echo === Quality Assurance ===
bash $BLECH_DIR/blech_run_QA.sh $DIR &&
echo === Units Plot ===
python $BLECH_DIR/blech_units_plot.py $DIR &&
echo === Get unit characteristics ===
python $BLECH_DIR/blech_units_characteristics.py $DIR &&
echo === Done ===
