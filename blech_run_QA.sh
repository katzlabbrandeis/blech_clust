# Runs set of QA tests on Blech data
DIR=$1
echo
echo "=============================="
echo "Running QA tests on Blech data"
echo "Directory: $DIR"
echo

echo "Running Similarity test"
python utils/qa_utils/unit_similarity.py $DIR

echo
echo "Running Drift test"
python utils/qa_utils/drift_check.py $DIR
python utils/qa_utils/elbo_drift.py $DIR

echo
echo "Finished QA tests"
echo "=============================="
