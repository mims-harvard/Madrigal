"""Geometric-mean-aggregate normalized ranks across a list of model runs, then re-normalize.

Script version of the "Across runs" section of notebooks/generate_embeddings.ipynb.

Inputs (per run):  {data_source}_drugs_normalized_ranks_{epoch}{suffix}.npy
Outputs (in the split-level model_output dir):
  {data_source}_drugs_normalized_ranks_{epoch}{suffix}_gmean.npy   (gmean across runs)
  {data_source}_drugs_normalized_ranks{suffix}.npy                 (gmean re-normalized to [0, 1])

Example:
  python aggregate_gmean_ranks.py \
      --data_source DrugBank \
      --epoch 700 \
      --interval 10 \
      --checkpoints all_train_seed0 all_train_seed1 all_train_seed2 all_train_seed42 all_train_seed99
"""
import os
import argparse
from time import time
from multiprocessing import Pool

import numpy as np
from numpy.lib.format import open_memmap
from scipy.stats.mstats import gmean

from madrigal.utils import BASE_DIR


# Module-level globals consumed by run_slice() in the re-normalization Pool (set before Pool creation).
gmean_ranks = None
gmean_ranks_norm = None
mask_indices = None


def n_workers():
    """Worker count capped to the CPUs allocated to this job."""
    slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if slurm > 0:
        return slurm
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def classwise_normalized_rank_3d_numpy(tensor):
    flat_tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.shape[0] > 1:
        flat_rank = flat_tensor.argsort(axis=1).argsort(axis=1) + 1
    else:
        temp = flat_tensor.argsort(axis=1)
        flat_rank = np.empty_like(temp)
        flat_rank[0, temp] = np.arange(flat_rank.shape[1]) + 1
        del temp
    normalized_rank = flat_rank / (tensor.shape[1] * (tensor.shape[2] - 1) / 2)
    return normalized_rank.reshape(tensor.shape)


def run_slice(tup):
    start, end = tup
    gmean_ranks_slice = gmean_ranks[start:end, :, :]
    gmean_ranks_slice = gmean_ranks_slice.copy()
    gmean_ranks_slice[:, mask_indices[0], mask_indices[1]] = 1e7
    gmean_ranks_slice_norm = classwise_normalized_rank_3d_numpy(gmean_ranks_slice)
    gmean_ranks_slice_norm[:, mask_indices[0], mask_indices[1]] = 0
    gmean_ranks_slice_norm = gmean_ranks_slice_norm + gmean_ranks_slice_norm.swapaxes(1, 2)
    gmean_ranks_norm[start:end, :, :] = gmean_ranks_slice_norm


def eval_type_suffix(eval_type):
    return "" if eval_type == "full_full" else "_" + eval_type.split("_")[0]


def resolve_epoch(split_output_dir, checkpoint, epoch_arg):
    """Return the integer epoch used in the rank filenames. If 'best'/'none', read it from best_model.pt."""
    if str(epoch_arg).lower() not in {"best", "none", ""}:
        return int(epoch_arg)
    import torch  # only needed for the 'best' case
    ckpt_path = split_output_dir + f"{checkpoint}/best_model.pt"
    return torch.load(ckpt_path, map_location="cpu")["epoch"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_source", required=True, choices=["DrugBank", "TWOSIDES"])
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="List of run checkpoint directory names to aggregate")
    p.add_argument("--epoch", default="best",
                   help="Epoch number used in the per-run rank filenames, or 'best'/'none' (read from best_model.pt of the first run)")
    p.add_argument("--interval", type=int, default=10,
                   help="Number of label rows processed per chunk during gmean")
    p.add_argument("--split_method", default="split_by_pairs")
    p.add_argument("--eval_type", default="full_full")
    p.add_argument("--example", action="store_true",
                   help="Aggregate the '_example' outputs of generate_embeddings_and_scores.py --example")
    return p.parse_args()


def main():
    global gmean_ranks, gmean_ranks_norm, mask_indices
    args = parse_args()
    suffix = eval_type_suffix(args.eval_type)
    if args.example:
        suffix += "_example"
    split_output_dir = BASE_DIR + f"model_output/{args.data_source}/{args.split_method}/"

    # All runs are assumed to share one epoch; resolve from the first run.
    epoch = resolve_epoch(split_output_dir, args.checkpoints[0], args.epoch)
    print(f"Aggregating {len(args.checkpoints)} runs at epoch={epoch} (suffix={suffix!r})", flush=True)

    # --- load per-run normalized ranks --------------------------------------
    normalized_ranks_list = []
    for checkpoint in args.checkpoints:
        checkpoint_dir = split_output_dir + f"{checkpoint}/"
        path = f"{checkpoint_dir}/{args.data_source}_drugs_normalized_ranks_{epoch}{suffix}.npy"
        normalized_ranks_list.append(np.load(path, mmap_mode="r"))
        print(f"  loaded {path}", flush=True)
    normalized_ranks = normalized_ranks_list[0]

    # --- geometric mean across runs -----------------------------------------
    gmean_npy_path = split_output_dir + f"{args.data_source}_drugs_normalized_ranks_{epoch}{suffix}_gmean.npy"
    gmean_fp = open_memmap(gmean_npy_path, mode="w+", dtype=np.float32,
                           shape=(normalized_ranks.shape[0], normalized_ranks.shape[1], normalized_ranks.shape[2]))

    st = time()
    for start, end in zip(
        np.arange(0, normalized_ranks.shape[0], args.interval),
        np.arange(0, normalized_ranks.shape[0], args.interval)[1:].tolist() + [normalized_ranks.shape[0]],
    ):
        print(f"[gmean] labels {start}", flush=True)
        gmean_fp[start:end, :, :] = gmean(
            np.stack([ranks[start:end, :, :] for ranks in normalized_ranks_list], axis=-1), axis=-1
        )
    print(f"[gmean] took {(time() - st):.4f} s", flush=True)
    gmean_fp.flush()
    del gmean_fp
    print(f"[gmean] saved {gmean_npy_path}", flush=True)

    # --- re-normalize the aggregated ranks ----------------------------------
    gmean_ranks = np.load(gmean_npy_path, mmap_mode="r")
    final_npy_path = split_output_dir + f"{args.data_source}_drugs_normalized_ranks{suffix}.npy"
    gmean_ranks_norm = open_memmap(final_npy_path, mode="w+", dtype=np.float32, shape=gmean_ranks.shape)
    mask_indices = np.vstack(np.triu_indices(gmean_ranks.shape[1], k=0, m=gmean_ranks.shape[2]))

    st = time()
    nproc = n_workers()
    print(f"[re-normalize] normalizing with {nproc} workers...", flush=True)
    with Pool(processes=nproc) as pool:
        pool.map(
            run_slice,
            zip(
                np.arange(0, gmean_ranks.shape[0], 1),
                np.arange(0, gmean_ranks.shape[0], 1)[1:].tolist() + [gmean_ranks.shape[0]],
            ),
        )
    print(f"[re-normalize] took {(time() - st):.4f} s", flush=True)
    gmean_ranks_norm.flush()
    print(f"[re-normalize] saved {final_npy_path}", flush=True)
    del gmean_ranks_norm


if __name__ == "__main__":
    main()
