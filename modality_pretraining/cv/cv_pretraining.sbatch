#!/bin/bash
#SBATCH -J cv_pretraining
#SBATCH -o cv.out
#SBATCH -e cv.err
#SBATCH -c 1
#SBATCH -t 5:00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=6G

cd /path/to/Madrigal/modality_pretraining/cv
mamba activate madrigal_env

python cv_pretraining.py
