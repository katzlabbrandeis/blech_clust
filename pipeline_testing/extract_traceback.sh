#!/bin/bash

# Check for input argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <log_file_path>"
    exit 1
fi

LOG_FILE="$1"

awk '
{
    buffer[NR % 100] = $0
    if (found > 0) {
        print
        found++
        if (found > 50) exit
    }
}
/Traceback/ && !found {
    for (i = NR - 10; i < NR; i++) {
        idx = i % 100
        if (buffer[idx]) print buffer[idx]
    }
    print
    found = 1
}
' "$LOG_FILE"
