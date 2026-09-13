#!/bin/bash
#SBATCH -J agg_gmean_ranks
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p <partition>
#SBATCH --mem=192G

base="/path/to/Madrigal"
# CPU-only. Memory scales with --interval x number of runs.
data_source="DrugBank"
epoch="700"
interval=10
checkpoints=("$@")     # e.g. sbatch run_aggregate_gmean_ranks.sh all_train_seed0 all_train_seed1 all_train_seed2 all_train_seed42 all_train_seed99
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "Usage: sbatch $0 <checkpoint_dir_name> [...]" >&2
    exit 1
fi

mamba activate madrigal_env
cd $base

python notebooks/generate_embeddings_and_scores/aggregate_gmean_ranks.py \
    --data_source=$data_source \
    --epoch=$epoch \
    --interval=$interval \
    --checkpoints "${checkpoints[@]}"
