#!/bin/sh
#SBATCH --job-name="testing_for_min_proj_medical"
#SBATCH --account=share-ie-idi

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1

#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --mem=16G              

#SBATCH --time=0-12:00:00     

/cluster/home/vutourni/HNTS-MRG/.venv/bin/python /cluster/home/vutourni/HNTS-MRG/src/test.py