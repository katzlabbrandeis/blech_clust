"""
This module monitors RAM usage and logs it to a specified directory. It continuously records the used, total, and percentage of RAM usage every second until interrupted.

- `monitor_ram(output_dir)`: Monitors RAM usage and writes the data to a log file named `ram_usage.log` in the specified output directory. The log includes a timestamp, used RAM in GB, total RAM in GB, and RAM usage percentage. The function runs indefinitely until a KeyboardInterrupt is received.
- The script requires one command-line argument specifying the output directory for the log file. If the argument is not provided, it prints usage instructions and exits.
"""
#!/usr/bin/env python3
import psutil
import time
import sys
import os
from datetime import datetime


def monitor_ram(output_dir):
    """Monitor RAM usage and write to a log file"""
    log_file = os.path.join(output_dir, "ram_usage.log")

    with open(log_file, 'a') as f:
        f.write("Timestamp,RAM_Used_GB,RAM_Total_GB,RAM_Percent\n")

        while True:
            try:
                mem = psutil.virtual_memory()
                used_gb = mem.used / (1024 * 1024 * 1024)  # Convert to GB
                total_gb = mem.total / (1024 * 1024 * 1024)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                log_line = f"{timestamp},{used_gb:.2f},{total_gb:.2f},{mem.percent}\n"
                f.write(log_line)
                f.flush()

                time.sleep(1)  # Log every 1 seconds

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ram_monitor.py <output_directory>")
        sys.exit(1)

    monitor_ram(sys.argv[1])
