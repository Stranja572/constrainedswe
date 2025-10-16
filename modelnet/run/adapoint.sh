#!/bin/bash
#SBATCH --job-name=point
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2                     
#SBATCH --output=point_out/constrained_swe_wswe/out-%A-%a.log
#SBATCH --error=point_out/constrained_swe_wswe/err-%A-%a.log

SEEDS=(42 123 545)         

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate graphgps 

#pooling: swe, GAP, constrained_swe, constrained_esp, lot, fsw, fspool, covariance
python -u train.py --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" --pooling swe --projections 4