#!/bin/bash
#SBATCH -J finetune_all_train
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH -p gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G

base="/home/yeh803/workspace/DDI/NovelDDI"
split_method="split_by_pairs"
repeat_num=None

config_file="configs/ddi_finetune/DrugBank/sweep_config_elated_sweep_163.yaml"
checkpoint="2024-02-06_18:12_helpful-field-81/checkpoint_1000.pt"  # elated-sweep-163 (new corresponding checkpoint)
full_finetune_mode="str_str+random_sample"  # elated-sweep-163

seed=42

source activate primekg
cd $base

# Full
python train_ddi_batch_all_train.py --checkpoint=$checkpoint --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# -p gpu_quad,gpu_requeue
# --qos=gpuquad_qos
# --gres=gpu:a100:1
# --requeue