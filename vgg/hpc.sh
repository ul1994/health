#!/bin/bash

#SBATCH --job-name=ulzeejob

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=16GB 
#SBATCH --gres=gpu:p40:1 
#SBATCH --time=12:00:00

echo "START"

module purge

module load python3/intel/3.6.3
module load cudnn/9.0v7.0.5
module load cuda/9.0.176
module load tensorflow/python3.6/1.5.0

echo "Loaded modules"
nvidia-smi
python3 train.py
python3 eval.py
echo "Ended"
