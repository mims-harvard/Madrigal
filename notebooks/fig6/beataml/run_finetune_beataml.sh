#!/bin/bash
#SBATCH -J beataml
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

base="/path/to/Madrigal"
# One invocation = 5 Madrigal runs (frozen embeddings) x 1 training seed x 6 model types.
# Usage: sbatch run_finetune_beataml.sh [split_method] [seed]
SPLIT=${1:-patient}
SEED=${2:-42}
export OMP_NUM_THREADS=16

mamba activate madrigal_env
cd $base/notebooks/fig6/beataml

python -u finetune_beataml.py \
--split_method "$SPLIT" \
--seed "$SEED" \
--freeze_level "madrigal" \
--drug_combo_dim 128 \
--scorer_hidden_dims 256 128 \
--scorer_dropout 0.2 \
--lr 0.001 \
--use_rnaseq_profile \
--use_mut_profile \
--use_clin_feature
