#!/bin/bash
#SBATCH -J pretrain_drugbank
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

base="/path/to/Madrigal"

mamba activate madrigal_env
cd $base

python pretrain.py --from_yaml configs/cl_pretrain/pretrain_drugbank_shared_basal.yaml --run_name pretrain_drugbank
