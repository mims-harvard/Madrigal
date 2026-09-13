#!/bin/bash
#SBATCH -J kg_pretraining
#SBATCH -o kg.out
#SBATCH -e kg.err
#SBATCH -c 1
#SBATCH -t 3:00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

cd /path/to/Madrigal/modality_pretraining/kg
mamba activate madrigal_env

python kg_pretraining.py
