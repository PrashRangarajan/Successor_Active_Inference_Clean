#!/bin/bash
# Wait for a GPU to become free, then run the given script.
# Usage: nohup bash scripts/wait_and_run.sh scripts/run_exp_h.py &
#
# Checks every 30min if a GPU has <1GB memory used (= idle).
# Picks the first free GPU and launches the experiment on it.

SCRIPT="${1:-scripts/run_exp_h.py}"
SAVE_DIR="${2:-}"
THRESHOLD_MB=1000  # GPU is "free" if memory used < this

echo "[$(date)] Waiting for a free GPU to run: $SCRIPT"

while true; do
    # Check each GPU
    for GPU_ID in 0 1; do
        MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' ')
        if [ -n "$MEM_USED" ] && [ "$MEM_USED" -lt "$THRESHOLD_MB" ]; then
            echo "[$(date)] GPU $GPU_ID free (${MEM_USED}MB used). Launching $SCRIPT"
            if [ -n "$SAVE_DIR" ]; then
                CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n sai python "$SCRIPT" "$SAVE_DIR" cuda
            else
                CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n sai python "$SCRIPT"
            fi
            echo "[$(date)] Finished with exit code $?"
            exit $?
        fi
    done
    sleep 1800
done
