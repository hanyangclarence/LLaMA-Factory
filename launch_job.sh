#!/bin/bash 
#SBATCH --job-name=yh
#SBATCH -o slurm_output/sft_%j.out
#SBATCH -e slurm_output/sft_%j.err
#SBATCH --mem=400G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8  # total number of tasks across all nodes
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8

source /gpfs/u/home/LMCG/LMCGhazh/scratch/miniconda3x86/etc/profile.d/conda.sh
conda activate vlm-r1

llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft.yaml