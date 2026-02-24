#!/bin/bash
# Setup conda environment on Hyak for neural SR training
# Run this once: bash hyak/setup_env.sh

module load anaconda3

# Create environment
conda create -n neural_sr python=3.10 -y
conda activate neural_sr

# Core dependencies
pip install torch numpy scipy matplotlib scikit-learn
pip install gymnasium imageio imageio-ffmpeg

# MuJoCo (for HalfCheetah, InvertedPendulum)
pip install mujoco gymnasium[mujoco]

# gymnasium-robotics (for PointMaze)
pip install gymnasium-robotics

echo "Environment 'neural_sr' ready. Activate with: conda activate neural_sr"
