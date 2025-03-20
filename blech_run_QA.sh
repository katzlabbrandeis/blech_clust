# Runs set of QA tests on Blech data
DIR=$1
echo
echo "=============================="
echo "Running QA tests on Blech data"
echo "Directory: $DIR"
echo

echo "Running Similarity test"
python utils/qa_utils/unit_similarity.py $DIR || { echo "Similarity test failed"; exit 1; }

echo
echo "Running Drift test"
python utils/qa_utils/drift_check.py $DIR || { echo "Drift test failed"; exit 1; }
python utils/qa_utils/elbo_drift.py $DIR || { echo "ELBO drift test failed"; exit 1; }

echo
echo "Finished QA tests"
echo "=============================="
