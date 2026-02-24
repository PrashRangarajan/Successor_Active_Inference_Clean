#!/bin/bash
#SBATCH --job-name=acrobot-sf
#SBATCH --output=logs/acrobot_%j.out
#SBATCH --error=logs/acrobot_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#
# For GPU, uncomment and set your partition:
# #SBATCH --partition=gpu-a40
# #SBATCH --gpus=1
#
# For CPU-only (works fine, ~60 min):
# #SBATCH --partition=compute
#
# Adjust partition to whatever you have access to.

module load anaconda3
conda activate neural_sr

mkdir -p logs

# Detect GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    DEVICE="cpu"
    echo "Running on CPU"
fi

echo "Starting acrobot training on $DEVICE"
python -u examples/run_neural_acrobot.py --device $DEVICE

echo "Starting multi-run evaluation"
python -u examples/run_eval_neural_acrobot.py --train --quick

echo "Done!"
