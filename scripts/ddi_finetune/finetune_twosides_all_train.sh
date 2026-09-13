#!/bin/bash
#SBATCH -J finetune_twosides_all_train
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

base="/path/to/Madrigal"
seed=${1:-0}                                 # the released checkpoints use seeds 0, 1, 2, 42, 99
config_file="configs/ddi_finetune/TWOSIDES/bottleneck_all_available.yaml"
checkpoint="TWOSIDES/checkpoint_1000.pt"          # relative to CL_CKPT_DIR
full_finetune_mode="str_random_sample"

mamba activate madrigal_env
cd $base

python train_ddi_batch_all_train.py --checkpoint=$checkpoint --finetune_mode=$full_finetune_mode --split_method=split_by_pairs --repeat=None --seed=$seed --from_yaml=$config_file --run_name=all_train_seed$seed
