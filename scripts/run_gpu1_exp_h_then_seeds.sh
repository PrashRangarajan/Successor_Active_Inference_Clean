#!/bin/bash
# GPU 1: Run Exp H (sf_dim=256) first, then seeds 1,3 of Exp I
set -e
cd /home/prashr/Successor_Active_Inference_Clean

echo "=== GPU 1: Starting Exp H (sf_dim=256) ==="
CUDA_VISIBLE_DEVICES=1 conda run -n sai python scripts/run_exp_h.py data/neural_point_maze_exp_h cuda

echo ""
echo "=== GPU 1: Exp H done, starting Exp I seeds 1,3 ==="
CUDA_VISIBLE_DEVICES=1 conda run -n sai python scripts/run_exp_i_seeds.py cuda 1 3

echo ""
echo "=== GPU 1: All done ==="
