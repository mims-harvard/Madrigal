#!/bin/bash
#SBATCH -J structure_pretraining
#SBATCH -o str.out
#SBATCH -e str.err
#SBATCH -c 1
#SBATCH -t 3:00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=15G

mamba activate madrigal_env

cd /path/to/Madrigal/modality_pretraining/str/

python structure_pretraining_muv.py 
