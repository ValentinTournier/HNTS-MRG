#!/bin/sh
#SBATCH --job-name="min_proj_medical_UNet_v1"
#SBATCH --account=share-ie-idi

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --mem=16G              

#SBATCH --time=0-12:00:00     

/cluster/home/vutourni/HNTS-MRG/.venv/bin/python /cluster/home/vutourni/HNTS-MRG/src/train.py