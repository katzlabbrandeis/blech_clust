DIR=$1
BLECH_DIR=$HOME/Desktop/blech_clust
echo Running Units Plot
python $BLECH_DIR/blech_units_plot.py $DIR;
echo Running Make Arrays
python $BLECH_DIR/blech_make_arrays.py $DIR;
echo Running Quality Assurance 
bash $BLECH_DIR/blech_run_QA.sh $DIR;
echo Running Make PSTHs
python $BLECH_DIR/blech_make_psth.py $DIR;
echo Running Palatability Identity Setup
python $BLECH_DIR/blech_palatability_identity_setup.py $DIR;
echo Running Overlay PSTH
python $BLECH_DIR/blech_overlay_psth.py $DIR;
