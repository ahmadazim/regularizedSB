#!/bin/bash
#SBATCH -J figure7_rep          
#SBATCH -o logs/%x_%j.out     
#SBATCH -e logs/%x_%j.err     
#SBATCH -t 24:00:00           
#SBATCH -p gpu_quad                
#SBATCH -c 8                  
#SBATCH --account=park
#SBATCH --mem=80G             
#SBATCH --gres=gpu:2          

# Exit on error
set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=============================================="

# Activate conda environment
module load conda/miniforge3/24.11.3-0
conda activate isoformer-real

# Navigate to project root
cd /home/tig687/regularizedSB

# Print Python and GPU info
echo "Python: $(which python)"
python --version
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "=============================================="

# Run the Figure 7 replication experiment in parallel mode
# With 4 gammas and 2 GPUs, gammas will be distributed round-robin:
#   GPU 0: gamma[0], gamma[2]
#   GPU 1: gamma[1], gamma[3]
# Note: All 4 gamma workers run simultaneously, but share 2 GPUs
echo "Running parallel experiment with $(nvidia-smi -L | wc -l) GPUs..."
python -u richie_results/replicate_figure7_v2.py 2>&1 | tee "logs/figure7_replication_${SLURM_JOB_ID:-local}.log"

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
