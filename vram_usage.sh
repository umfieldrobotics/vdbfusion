#!/bin/bash

# Log file path
LOG_FILE="gpu_python_memory_usage.log"

# Function to get GPU memory usage for Python processes
log_gpu_python_memory_usage() {
    # Get the current date and time
    echo "$(date):" >> "$LOG_FILE"
    
    # Parse nvidia-smi for python processes
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits | grep python >> "$LOG_FILE"
    
    # If no python processes are found, log that information
    if [ $? -ne 0 ]; then
        echo "No Python processes found using GPU" >> "$LOG_FILE"
    fi

    echo "" >> "$LOG_FILE"  # Add a new line for readability
}

# Infinite loop to log every second
while true; do
    log_gpu_python_memory_usage
    sleep 1
done
