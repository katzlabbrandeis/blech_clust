#!/usr/bin/env python3
import psutil
import time
import sys
import os
from datetime import datetime

def monitor_ram(output_dir):
    """Monitor RAM usage and write to a log file"""
    log_file = os.path.join(output_dir, "ram_usage.log")
    
    with open(log_file, 'w') as f:
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
                
                time.sleep(5)  # Log every 5 seconds
                
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ram_monitor.py <output_directory>")
        sys.exit(1)
    
    monitor_ram(sys.argv[1])
