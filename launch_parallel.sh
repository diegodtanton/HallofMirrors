#!/bin/bash

# ========================================================
# Parallel Launcher for Hall of Mirrors Experiment
# ========================================================

# 1. FORCE SINGLE-THREADED MODE
# This prevents PyTorch from spawning 32 threads per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# Create log directory
mkdir -p logs

echo "Starting 12 parallel experiments..."
echo "Configured for Single-Threaded PyTorch (prevents CPU thrashing)"

# List of configurations: "HiddenSize Layers"
configs=(
    "8 1" "16 1" "32 1" "64 1" "128 1" "256 1" "512 1" "1024 1"
    "8 2" "16 2" "32 2" "64 2" "128 2" "256 2" "512 2" "1024 2"
    "8 3" "16 3" "32 3" "64 3" "128 3" "256 3" "512 3" "1024 3"
)

for config in "${configs[@]}"; do
    # Split the string into variables
    set -- $config
    hs=$1
    layers=$2
    
    run_id="hs${hs}_l${layers}"
    echo "Launching process for: $run_id"
    
    # Run in background (&), redirect stdout and stderr to a log file
    # We use nohup so it survives if your SSH session disconnects
    nohup python3 run_pipeline.py --hs $hs --layers $layers > "logs/${run_id}.log" 2>&1 &
    
    # Sleep briefly to stagger startups
    sleep 1
done

echo "========================================================"
echo "All jobs launched in background."
echo "Check progress using 'htop' (Load avg should be ~12-15)"
echo "========================================================"