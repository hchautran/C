#!/bin/bash
#SBATCH --job-name="Lab1"
#SBATCH --output="lab1.out"
#SBATCH --error="lab1.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:01:00

module reset
module load cuda/12.4
./vecadd