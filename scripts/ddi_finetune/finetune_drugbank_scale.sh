#!/bin/bash
#SBATCH -J finetune
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 16:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --mem=16G

base="/path/to/Madrigal"
split_method="split_by_pairs"
repeat_num=None

config_file="configs/ddi_finetune/DrugBank/sweep_config_elated_sweep_163.yaml"
checkpoint="checkpoint_1000.pt"
full_finetune_mode="str_str+random_sample"

seed=42

source activate primekg
cd $base

# Full
python train_ddi_batch.py --checkpoint=$checkpoint --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# Full without finetuning
python train_ddi_batch.py --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# Structure ablation without finetuning
python train_ddi_batch.py --finetune_mode="ablation_str_str" --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# Structure ablation with finetuning
python train_ddi_batch.py --checkpoint=$checkpoint --finetune_mode="ablation_str_str" --split_method=$split_method --repeat $repeat_num --seed=$seed --from_yaml $config_file