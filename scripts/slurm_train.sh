#!/bin/bash
#SBATCH -J regSB_train
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 24:00:00
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 CONFIG_YAML"
  exit 1
fi

CONFIG="$1"
mkdir -p logs

echo "Running with config: ${CONFIG}"
python -u /n/home03/ahmadazim/SCRIPTS/regSB/main.py --config "${CONFIG}" 2>&1 | tee "logs/$(basename ${CONFIG%.*})_${SLURM_JOB_ID:-local}.log"


