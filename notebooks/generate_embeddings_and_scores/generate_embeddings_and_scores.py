"""Generate drug embeddings, raw DDI scores, and normalized ranks for individual model runs.

Script version of the "For each run" section of notebooks/generate_embeddings.ipynb.

For each provided checkpoint this script will:
  1. Build the all-drugs batch and load the model.
  2. Compute and save full drug embeddings              -> all_drug_embeddings_full_{epoch}{suffix}.pt
  3. (unless --stop_after embeddings)
     Compute and save raw decoder scores               -> {data_source}_drugs_raw_scores_{epoch}{suffix}.npy
     Convert scores to normalized ranks in [0, 1]      -> {data_source}_drugs_normalized_ranks_{epoch}{suffix}.npy

All outputs are written into each checkpoint's own model_output directory.

Example:
  python generate_embeddings_and_scores.py \
      --data_source DrugBank \
      --eval_type full_full \
      --finetune_mode str_str+random_sample \
      --epoch 700 \
      --stop_after all \
      --checkpoints all_train_seed0 all_train_seed1 all_train_seed2 all_train_seed42 all_train_seed99
"""
import os
import argparse
from time import time
from multiprocessing import Pool

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import torch
import torch.nn.functional as F

from madrigal.utils import DATA_DIR, BASE_DIR, to_device
from madrigal.evaluate.predict import get_data_for_analysis_all_drugs
from madrigal.models.models import MadrigalEncoder, MadrigalMultilabel
from madrigal.evaluate.eval_utils import get_evaluate_masks


# Number of label (DDI type) rows decoded per GPU pass when generating raw scores.
LABEL_CHUNK = 30

# Module-level globals consumed by run_slice() in the multiprocessing Pool. They are set
# (per checkpoint) just before the Pool is created so the forked workers inherit them.
raw_scores = None
raw_scores_norm = None
mask_indices = None


def n_workers():
    """Number of worker processes to use, capped to the CPUs actually allocated to this job."""
    slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if slurm > 0:
        return slurm
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def classwise_normalized_rank_3d_numpy(tensor):
    # flatten the tensor while maintaining the class dimension
    flat_tensor = tensor.reshape(tensor.shape[0], -1)

    # compute the ranks
    if tensor.shape[0] > 1:
        flat_rank = flat_tensor.argsort(axis=1).argsort(axis=1) + 1
    else:
        temp = flat_tensor.argsort(axis=1)
        flat_rank = np.empty_like(temp)
        flat_rank[0, temp] = np.arange(flat_rank.shape[1]) + 1
        del temp

    # normalize the ranks
    normalized_rank = flat_rank / (tensor.shape[1] * (tensor.shape[2] - 1) / 2)

    # reshape back to the original shape
    return normalized_rank.reshape(tensor.shape)


def run_slice(tup):
    start, end = tup
    raw_scores_slice = raw_scores[start:end, :, :]
    raw_scores_slice = raw_scores_slice.copy()
    raw_scores_slice[:, mask_indices[0], mask_indices[1]] = 1e7
    raw_scores_slice_norm = classwise_normalized_rank_3d_numpy(raw_scores_slice)
    raw_scores_slice_norm[:, mask_indices[0], mask_indices[1]] = 0
    raw_scores_slice_norm = raw_scores_slice_norm + raw_scores_slice_norm.swapaxes(1, 2)
    raw_scores_norm[start:end, :, :] = raw_scores_slice_norm


def eval_type_suffix(eval_type):
    """Filename suffix for non-default eval types (full_full -> "", str+kg_str+kg -> "_str+kg", str_str -> "_str")."""
    return "" if eval_type == "full_full" else "_" + eval_type.split("_")[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_source", required=True, choices=["DrugBank", "TWOSIDES"])
    p.add_argument("--eval_type", default="full_full",
                   help="e.g. full_full, str+kg_str+kg, str_str")
    p.add_argument("--finetune_mode", default="str_str+random_sample",
                   help="e.g. str_str+random_sample, str_full, str_random_sample")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint directory names under model_output/{data_source}/{split_method}/")
    p.add_argument("--epoch", default="best",
                   help="Checkpoint epoch number, or 'best'/'none' to use best_model.pt")
    p.add_argument("--stop_after", default="all", choices=["embeddings", "all"],
                   help="'embeddings' stops after saving embeddings; 'all' also generates raw scores + normalized ranks")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Device for embeddings + raw-score generation (the rank-normalization step is always CPU)")
    p.add_argument("--split_method", default="split_by_pairs")
    p.add_argument("--repeat", default=None)
    p.add_argument("--kg_encoder", default="hgt")
    p.add_argument("--example", action="store_true",
                   help="Only use the first 200 drugs; outputs get an '_example' suffix "
                        "(normalization then differs from the full dataset)")
    p.add_argument("--drug_set", default="nash", choices=["nash", "ddi"],
                   help="Which drug table to embed. 'nash' is the case-study build "
                        "(drug_metadata_nash.pkl + nash_molecules_torchdrug.pt; 11,607 drugs, "
                        "case-study drugs appended at the end). 'ddi' is the FULL master table "
                        "(drug_metadata_ddi.pkl + all_molecules_torchdrug.pt; 21,842 drugs, no "
                        "case-study drugs) and writes {data_source}_all_metadata_drug_embeddings_full.pt.")
    return p.parse_args()


def generate_for_checkpoint(checkpoint, args, n_drugs, device, suffix, use_best):
    global raw_scores, raw_scores_norm, mask_indices

    checkpoint_dir = BASE_DIR + f"model_output/{args.data_source}/{args.split_method}/{checkpoint}/"
    if use_best:
        ckpt_path = checkpoint_dir + "best_model.pt"
    else:
        ckpt_path = checkpoint_dir + f"checkpoint_{int(args.epoch)}.pt"

    print(f"\n=== {checkpoint} ===\nckpt: {ckpt_path}", flush=True)

    # --- data ---------------------------------------------------------------
    _, _, batch, label_map = get_data_for_analysis_all_drugs(
        data_source=args.data_source,
        kg_encoder=args.kg_encoder,
        split_method=args.split_method,
        repeat=args.repeat,
        path_base=DATA_DIR,
        checkpoint=ckpt_path,
        first_num_drugs=200 if args.example else n_drugs,
        # add_specific_drugs=None takes get_all_drugs_data's "full master table" branch
        # (drug_metadata_ddi.pkl + all_molecules_torchdrug.pt); otherwise the case-study build.
        add_specific_drugs=None if args.drug_set == "ddi" else args.drug_set,
    )

    # --- model --------------------------------------------------------------
    checkpoint_dict = torch.load(ckpt_path, map_location="cpu")
    epoch = checkpoint_dict["epoch"] if use_best else int(args.epoch)

    encoder = MadrigalEncoder(**checkpoint_dict["encoder_configs"])
    model = MadrigalMultilabel(encoder, **checkpoint_dict["model_configs"])
    model.load_state_dict(checkpoint_dict["state_dict"])
    model.eval()
    model.to(device)

    batch_head = to_device(batch["head"], device)
    batch_tail = to_device(batch["tail"], device)
    batch_kg = to_device(batch["kg"], device)
    head_masks_base = batch["head"]["masks"]
    tail_masks_base = batch["tail"]["masks"]
    masks_head, masks_tail = get_evaluate_masks(head_masks_base, tail_masks_base, args.eval_type, args.finetune_mode, device)

    # --- (1) embeddings -----------------------------------------------------
    with torch.no_grad():
        z_full = model.encoder(
            batch_head["drugs"], masks_head, batch_head["strs"], batch_kg, batch_head["cv"], batch_head["tx"]
        )
        if model.normalize:
            z_full = F.normalize(z_full)
    if args.drug_set == "ddi":
        # Full-table embeddings use the file name that predict.py loads (no epoch in the name).
        emb_path = f"{checkpoint_dir}/{args.data_source.lower()}_all_metadata_drug_embeddings_full{suffix}.pt"
    else:
        emb_path = f"{checkpoint_dir}/all_drug_embeddings_full_{epoch}{suffix}.pt"
    torch.save(z_full.detach().cpu(), emb_path)
    print(f"[embeddings] saved {emb_path}  shape={tuple(z_full.shape)}", flush=True)

    if args.stop_after == "embeddings":
        print("[stop_after=embeddings] skipping raw scores and normalized ranks.", flush=True)
        return

    # --- (2) raw scores -----------------------------------------------------
    scores_npy_path = f"{checkpoint_dir}/{args.data_source}_drugs_raw_scores_{epoch}{suffix}.npy"
    fp = open_memmap(scores_npy_path, mode="w+", dtype=np.float32,
                     shape=(label_map.shape[0], z_full.shape[0], z_full.shape[0]))
    for start, end in zip(
        np.arange(0, len(label_map), LABEL_CHUNK),
        np.arange(0, len(label_map), LABEL_CHUNK)[1:].tolist() + [len(label_map)],
    ):
        print(f"[raw scores] labels {start}", flush=True)
        with torch.no_grad():
            pred_scores = model.decoder(z_full, z_full, (start, end)).detach().cpu().numpy()
        fp[start:end, :, :] = pred_scores
    fp.flush()
    del fp
    print(f"[raw scores] saved {scores_npy_path}", flush=True)

    # --- (3) normalized ranks ----------------------------------------------
    raw_scores = np.load(scores_npy_path, mmap_mode="r")
    ranks_npy_path = f"{checkpoint_dir}/{args.data_source}_drugs_normalized_ranks_{epoch}{suffix}.npy"
    raw_scores_norm = open_memmap(ranks_npy_path, mode="w+", dtype=np.float32, shape=raw_scores.shape)
    mask_indices = np.vstack(np.triu_indices(raw_scores.shape[1], k=0, m=raw_scores.shape[2]))

    assert np.isnan(raw_scores[:1]).sum() == 0, "NaNs found in raw scores (checked first slice)"

    st = time()
    nproc = n_workers()
    print(f"[ranks] normalizing with {nproc} workers...", flush=True)
    with Pool(processes=nproc) as pool:
        pool.map(
            run_slice,
            zip(
                np.arange(0, raw_scores.shape[0], 1),
                np.arange(0, raw_scores.shape[0], 1)[1:].tolist() + [raw_scores.shape[0]],
            ),
        )
    print(f"[ranks] normalization took {(time() - st):.4f} s", flush=True)
    raw_scores_norm.flush()
    print(f"[ranks] saved {ranks_npy_path}", flush=True)
    del raw_scores_norm


def main():
    args = parse_args()
    suffix = eval_type_suffix(args.eval_type)
    if args.example:
        suffix += "_example"
    use_best = str(args.epoch).lower() in {"best", "none", ""}

    if args.drug_set != "nash" and args.stop_after != "embeddings":
        raise SystemExit(f"--drug_set={args.drug_set} supports --stop_after=embeddings only.")

    drug_metadata = pd.read_pickle(BASE_DIR + f"processed_data/drug_features/drug_metadata_{args.drug_set}.pkl")
    n_drugs = drug_metadata.shape[0]
    print(f"drug metadata ({args.drug_set}): {drug_metadata.shape}", flush=True)

    device = torch.device(args.device)

    for checkpoint in args.checkpoints:
        generate_for_checkpoint(checkpoint, args, n_drugs, device, suffix, use_best)


if __name__ == "__main__":
    main()
