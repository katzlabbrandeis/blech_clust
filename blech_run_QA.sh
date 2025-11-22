# Runs set of QA tests on Blech data
DIR=$1
SILENT_FLAG=""

# Check if --silent flag is passed
if [[ "$2" == "--silent" ]]; then
    SILENT_FLAG="--silent"
fi

echo
echo "=============================="
echo "Running QA tests on Blech data"
echo "Directory: $DIR"
echo

echo "Running Similarity test"
python utils/qa_utils/unit_similarity.py $DIR $SILENT_FLAG || { echo "Similarity test failed"; exit 1; }

echo
echo "Running Drift test"
python utils/qa_utils/drift_check.py $DIR $SILENT_FLAG || { echo "Drift test failed"; exit 1; }
python utils/qa_utils/elbo_drift.py $DIR $SILENT_FLAG || { echo "ELBO drift test failed"; exit 1; }

echo
echo "Finished QA tests"
echo "=============================="
