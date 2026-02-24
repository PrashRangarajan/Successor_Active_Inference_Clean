#!/bin/bash
#SBATCH --job-name=cheetah-sf
#SBATCH --output=logs/cheetah_%j.out
#SBATCH --error=logs/cheetah_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#
# For GPU, uncomment and set your partition:
# #SBATCH --partition=gpu-a40
# #SBATCH --gpus=1
#
# For CPU-only (works but slow, ~3-4 hours):
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

echo "Starting HalfCheetah training on $DEVICE"
python -u examples/run_neural_half_cheetah.py --train --device $DEVICE

echo "Done!"
