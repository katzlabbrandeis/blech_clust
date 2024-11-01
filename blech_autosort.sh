DIR=$1
echo === Blech Clust === 
python blech_clust.py $DIR &&
echo === Common Average Reference === 
python blech_common_avg_reference.py $DIR &&
echo === Blech Run Process === 
bash blech_run_process.sh $DIR &&
echo === Post Process ===
python blech_post_process.py $DIR &&
echo === Make Arrays ===
python blech_make_arrays.py $DIR &&
echo === Quality Assurance === 
bash blech_run_QA.sh $DIR &&
echo === Units Plot ===
python blech_units_plot.py $DIR &&
echo === Make PSTHs ===
python blech_make_psth.py $DIR &&
echo === Palatability Identity Setup ===
python blech_palatability_identity_setup.py $DIR &&
echo === Overlay PSTHs ===
python blech_overlay_psth.py $DIR &&
echo === Done ===
