#!/bin/bash
#SBATCH -J gen_emb_scores
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

base="/path/to/Madrigal"
# GPU for embeddings and raw scores; the rank normalization is CPU multiprocessing over the allocated cores.
data_source="DrugBank"
eval_type="full_full"
finetune_mode="str_str+random_sample"
epoch="700"
stop_after="all"       # "embeddings" or "all"
checkpoints=("$@")     # e.g. sbatch run_generate_embeddings_and_scores.sh all_train_seed0 all_train_seed1
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "Usage: sbatch $0 <checkpoint_dir_name> [...]" >&2
    exit 1
fi

mamba activate madrigal_env
cd $base

python notebooks/generate_embeddings_and_scores/generate_embeddings_and_scores.py \
    --data_source=$data_source \
    --eval_type=$eval_type \
    --finetune_mode=$finetune_mode \
    --epoch=$epoch \
    --stop_after=$stop_after \
    --checkpoints "${checkpoints[@]}"
