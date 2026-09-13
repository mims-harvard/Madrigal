#!/bin/bash
#SBATCH -J finetune_twosides
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 16:00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

base="/path/to/Madrigal"
split_method=${1:-split_by_drugs_targets}   # split_by_drugs_targets | split_by_drugs_atc | split_by_drugs_random | split_by_pairs
seed=${2:-0}
repeat_num=None
config_file="configs/ddi_finetune/TWOSIDES/bottleneck_all_available.yaml"
checkpoint="TWOSIDES/checkpoint_1000.pt"          # relative to CL_CKPT_DIR
full_finetune_mode="str_random_sample"

mamba activate madrigal_env
cd $base

# Madrigal
python train_ddi_batch.py --checkpoint=$checkpoint --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file --run_name=madrigal_seed$seed
# Ablation: without modality alignment (no contrastive pretraining)
python train_ddi_batch.py --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file --run_name=no_pretrain_seed$seed
# Ablation: structure only, without modality alignment
python train_ddi_batch.py --finetune_mode="ablation_str_str" --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file --run_name=structure_only_no_pretrain_seed$seed
# Ablation: structure only, with modality alignment
python train_ddi_batch.py --checkpoint=$checkpoint --finetune_mode="ablation_str_str" --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file --run_name=structure_only_seed$seed
