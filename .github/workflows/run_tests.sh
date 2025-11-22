#!/bin/bash

# Script to run blech_clust pipeline tests
# Usage: ./run_tests.sh <test_type>
# test_type: "spike-emg" or "emg-only"

set -e  # Exit on any error

TEST_TYPE="$1"
LOG_LOC="${GITHUB_WORKSPACE}/github.log"

if [ -z "$TEST_TYPE" ]; then
    echo "Error: Test type not specified"
    echo "Usage: $0 <spike-emg|emg-only>"
    exit 1
fi

case "$TEST_TYPE" in
    "spike-emg")
        echo "Running Prefect SPIKE then EMG test"
        conda run -n blech_clust python pipeline_testing/prefect_pipeline.py --spike-emg --silent 2>&1 | tee ${LOG_LOC}
        ;;
    "emg-only")
        echo "Running Prefect EMG only test"
        conda run -n blech_clust python pipeline_testing/prefect_pipeline.py -e --silent 2>&1 | tee ${LOG_LOC}
        ;;
    *)
        echo "Error: Unknown test type '$TEST_TYPE'"
        echo "Valid options: spike-emg, emg-only"
        exit 1
        ;;
esac

# Check for errors in the log
if grep -q "ERROR" ${LOG_LOC}; then
    echo "ERROR detected by bash"
    ./pipeline_testing/extract_traceback.sh ${LOG_LOC}
    exit 1
fi

echo "Test completed successfully"
