#!/bin/bash
#SBATCH -J normalize_scores
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p <partition>
#SBATCH --mem=160G

base="/path/to/Madrigal"

mamba activate madrigal_env
cd $base/notebooks

python normalize_scores.py
