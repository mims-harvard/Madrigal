from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import HeteroData
from torchdrug.data import PackedMolecule
from madrigal.utils import CELL_LINES
from madrigal.data.data_utils import (
    structured_negative_sampling_multilabel,
    sample_kg_data,
    remove_edges_attached_to_nodes,
)


def load_sentence_embedding(index, batch_size=256, folder_path="./"):
    # Calculate which batch file the index is in and the index within the batch
    batch_file_number = index // batch_size
    index_within_batch = index % batch_size

    # Construct the file name for the batch
    batch_file_path = f"{folder_path}/embeddings_{batch_file_number}.pt"

    # Load the batch of embeddings
    embeddings_batch = torch.load(batch_file_path)

    # Extract the specific embedding
    sentence_embedding = embeddings_batch[index_within_batch]

    return sentence_embedding


class BatchedDDIDataset(Dataset):
    """ Load positive pairs from a long DDI table
    Long so that each row is a pair of drug indices.
    """

    def __init__(self, split, paraphrase, use_label, embedding_dir, embedding_file):
        
        self.paraphrase = paraphrase
        
        df = pd.read_csv(f"{embedding_dir}/{split}_df.csv")
        if paraphrase:
            self.descriptions = [df[f"descriptions_{i}"].values for i in range(10)]
        elif use_label:
            self.descriptions = df["label_descriptions"].values            
        else:
            self.descriptions = df["descriptions"].values
            
        if paraphrase:
            self.edgelist = np.stack([df["head"].values, df["tail"].values], axis=1)
            self.labels = df["labels"].values
            self.pos_neg = df["pos_neg"].values
            self.all_label_indices = []
            self.all_embeddings = []
            
            for i in range(10):
                label_descriptions = np.unique(self.descriptions[i])
                label_mapping = {k: i for i, k in enumerate(label_descriptions)}
                
                self.all_label_indices.append(np.array([label_mapping[desc] for desc in self.descriptions[i]]))
                self.all_embeddings.append(torch.load(embedding_file.replace('_0', '_' + str(i)), map_location="cpu"))
            
            self.all_label_indices = np.array(self.all_label_indices) # 10, N
            self.all_embeddings = torch.stack(self.all_embeddings) # 10, L, 4096
            
        else:
            self.edgelist = np.stack([df["head"].values, df["tail"].values], axis=1)
            self.labels = df["labels"].values
            self.pos_neg = df["pos_neg"].values

            label_descriptions = np.unique(self.descriptions)
            label_mapping = {k: i for i, k in enumerate(label_descriptions)}
            self.label_indices = np.array(
                [label_mapping[desc] for desc in self.descriptions]
            )
            self.full_embeddings = torch.load(embedding_file, map_location="cpu")
    
    def __getitem__(self, index: int):
        if self.paraphrase:
            embeddings = torch.stack([self.all_embeddings[i][self.all_label_indices[i, index]] for i in range(10)]) # 10, 4096
            return (
                self.edgelist[index],
                self.labels[index],
                self.pos_neg[index],
                embeddings,
            )
            
        else:
            return (
                self.edgelist[index],
                self.labels[index],
                self.pos_neg[index],
                self.full_embeddings[self.label_indices[index]],
            )

    def __len__(self):
        return len(self.edgelist)


class BatchedDDIDataCollator:
    def __init__(
        self,
        data_source: str,
        split: str,
        split_method: str,
        repeat: str,
        num_labels: int,
        drug_metadata: pd.DataFrame,
        all_molecules: PackedMolecule,
        all_kg_data: HeteroData,
        kg_sampling_num_neighbors: int,
        kg_sampling_num_layers: int,
        cv_df: pd.DataFrame,
        tx_df: pd.DataFrame,
    ):
        """
        Args:
            path_base: Path to the directory containing the edgelist and adjmat.
            split: One of ["train", "val", "test", "val_within", "test_within", "val_between", "test_between"].
            split_method: One of ["ddi_centric", "drug_centric_hard", "drug_centric_easy"]. 
            num_negative_samples_per_pair: If None, don't sample negatives on-the-fly. Otherwise, sample this many negatives for each positive pair for the training set (still NO sampling for evaluation sets!). Only implemented for num = 1 and 2. Default is 1 since now we are **not** condensing the undirected edgelist.
            negative_sampling_probs_type: The type of probability distribution to use for sampling. Choose from ['uniform', 'degree', 'degree_w2v']. (default: `uniform`)
            device: Device to load the data to.
        """
        
        self.split = split
        self.num_labels = num_labels
        self.all_molecules = all_molecules
        self.all_kg_data = all_kg_data
        self.kg_sampling_num_neighbors = kg_sampling_num_neighbors
        self.kg_sampling_num_layers = kg_sampling_num_layers
        self.str_node_feat_dim = self.all_molecules.node_feature.shape[1]

        # load perturbation data
        self.cv_df = cv_df
        self.tx_df = tx_df

        # prepare modality masks
        self.drug_metadata = drug_metadata

        # if needs negative sampling on-the-fly
        # NOTE: In this case, split == 'train'. Also note that for train set, we theoretically shouldn't know the val/test edges, so we cannot filter them out when negative sampling, i.e. valid indices and ground truth edgelist should both come from only the train edgelist

    def __call__(self, batch: List[Tuple]):
        """
        Args:
            batch: A list of tuples of (positive_pair, label, neg_head, neg_tail).
        """

        edgelist, labels, pos_neg, embs = zip(*batch)
 
        embs = torch.stack(embs, dim=0) 

        edgelist = torch.tensor(np.array(edgelist), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        heads = edgelist[:, 0]
        tails = edgelist[:, 1]

        # extract modality availability
        unique_head_mod_avail = self.drug_metadata.loc[heads][
            ["view_str", "view_kg", "view_cv"]
            + [f"view_tx_{cell_line}" for cell_line in CELL_LINES]
        ]
        unique_tail_mod_avail = self.drug_metadata.loc[tails][
            ["view_str", "view_kg", "view_cv"]
            + [f"view_tx_{cell_line}" for cell_line in CELL_LINES]
        ]

        # extract structure modality (torchdrug mol graphs)
        # NOTE: KG modality is computed via HAN during encoding, by doing message passing across the KG and then
        unique_head_mol_strs = self.all_molecules[heads.tolist()]
        unique_tail_mol_strs = self.all_molecules[tails.tolist()]

        # Subgraphs are generated with NeighborLoader when sampling is requested. By default there is no sampling (`kg_sampling_num_neighbors=None`)
        # NOTE: To ensure KG sampling is identical for the overlapping heads and tails, we first extract the unique indices, then sample the KG. The same subgraph will be fed into the KG encoder for both heads and tails, and the relevant embeddings will be extracted after encoding for heads and tails, respectively.
        unique_drug_indices = torch.unique(torch.cat([heads, tails], dim=0))
        kg_dict = sample_kg_data(
            self.all_kg_data,
            unique_drug_indices,
            self.kg_sampling_num_neighbors,
            "neighborloader",
            self.kg_sampling_num_layers,
            drug_only=True,
        )

        # extract Cv, Tx signatures, fill dummies for unavailable ones
        def get_signatures_and_fill_dummy(
            unique_indices, sig_df, unique_mod_avail, sig, unique_sig_ids
        ):
            unique_sig_output = torch.randn((unique_indices.shape[0], sig_df.shape[0]))
            unique_sig_avail_indices = unique_mod_avail[sig].values == 1
            unique_sig_output[unique_sig_avail_indices, :] = torch.from_numpy(
                sig_df[unique_sig_ids[unique_sig_avail_indices]].values.T
            ).float()
            unique_sig_output[~unique_sig_avail_indices, :] = 0
            return unique_sig_output

        unique_head_cv_sig_ids = self.drug_metadata.loc[
            heads.tolist(), "cv_sig_id"
        ].values
        unique_tail_cv_sig_ids = self.drug_metadata.loc[
            tails.tolist(), "cv_sig_id"
        ].values
        unique_head_cv = get_signatures_and_fill_dummy(
            heads, self.cv_df, unique_head_mod_avail, "view_cv", unique_head_cv_sig_ids
        )
        unique_tail_cv = get_signatures_and_fill_dummy(
            tails, self.cv_df, unique_tail_mod_avail, "view_cv", unique_tail_cv_sig_ids
        )

        unique_head_tx_dict = {}
        unique_tail_tx_dict = {}
        for cell_line in CELL_LINES:
            unique_head_tx_dict[cell_line] = {}
            unique_tail_tx_dict[cell_line] = {}

            # sigs
            unique_head_tx_cell_line_sig_ids = self.drug_metadata.loc[
                heads.tolist(), f"{cell_line}_max_dose_averaged_sig_id"
            ].values
            unique_tail_tx_cell_line_sig_ids = self.drug_metadata.loc[
                tails.tolist(), f"{cell_line}_max_dose_averaged_sig_id"
            ].values
            unique_head_tx_dict[cell_line]["sigs"] = get_signatures_and_fill_dummy(
                heads,
                self.tx_df,
                unique_head_mod_avail,
                f"view_tx_{cell_line}",
                unique_head_tx_cell_line_sig_ids,
            )
            unique_tail_tx_dict[cell_line]["sigs"] = get_signatures_and_fill_dummy(
                tails,
                self.tx_df,
                unique_tail_mod_avail,
                f"view_tx_{cell_line}",
                unique_tail_tx_cell_line_sig_ids,
            )

            # drugs
            unique_head_tx_dict[cell_line]["drugs"] = heads.long()
            unique_tail_tx_dict[cell_line]["drugs"] = tails.long()

            # dosages
            unique_head_tx_dosages = (
                self.drug_metadata.loc[heads.tolist(), f"{cell_line}_pert_dose"]
                .fillna(0)
                .values
            )
            unique_tail_tx_dosages = (
                self.drug_metadata.loc[tails.tolist(), f"{cell_line}_pert_dose"]
                .fillna(0)
                .values
            )
            unique_head_tx_dict[cell_line]["dosages"] = torch.from_numpy(
                unique_head_tx_dosages
            ).float()
            unique_tail_tx_dict[cell_line]["dosages"] = torch.from_numpy(
                unique_tail_tx_dosages
            ).float()

            # cell_lines (in str, converted to one-hot )
            unique_head_tx_dict[cell_line]["cell_lines"] = np.array(
                [cell_line] * len(heads)
            )
            unique_tail_tx_dict[cell_line]["cell_lines"] = np.array(
                [cell_line] * len(tails)
            )

        # NOTE: Instead of using a label matrix, we put all label-specific bilinear decoder weights together into a 3D matrix, and compute for each label all drugs-drugs predictions together (N_drug x feat, feat x feat x num_labels, N_drug x feat), masking the unwanted ones thereafter. This is memory efficient and can be done in a single forward pass.
        label_matrix = None

        # finally, construct the input drug's modality masks: reverse to mod avail, here 1 means NOT having the modality, 0 means having the modality (for the sake of key_padding_mask, thus inverse)
        unique_head_masks = torch.from_numpy(1 - unique_head_mod_avail.values).bool()
        unique_tail_masks = torch.from_numpy(1 - unique_tail_mod_avail.values).bool()

        return {
            # unique heads
            "head": {
                "drugs": heads,
                "strs": unique_head_mol_strs,
                "cv": unique_head_cv,
                "tx": unique_head_tx_dict,
                "masks": unique_head_masks,
            },
            # unique tails
            "tail": {
                "drugs": tails,
                "strs": unique_tail_mol_strs,
                "cv": unique_tail_cv,
                "tx": unique_tail_tx_dict,
                "masks": unique_tail_masks,
            },
            # kg
            "kg": kg_dict,
            # used for indexing the label matrix
            "edge_indices": {
                "head": heads,
                "tail": tails,
                "label": labels,
                "pos_neg": pos_neg,
            },
            "text_embeddings": embs,
        }


def get_datasets(paraphrase, use_label, embedding_dir, train_embedding_file, eval_embedding_file):
    """
    Get datasets for DDI training
    """
    train_dataset = BatchedDDIDataset("train", paraphrase, use_label, embedding_dir, train_embedding_file)
    eval_dataset = BatchedDDIDataset("eval", paraphrase, use_label, embedding_dir, eval_embedding_file)

    return train_dataset, eval_dataset


def get_collators(
    drug_metadata, all_molecules, all_kg_data, train_kg_data, cv_df, tx_df, num_labels
):
    kg_sampling_num_neighbors = None
    kg_sampling_num_layers = None
    """
    Get collators for DDI training
    """
    train_collator = BatchedDDIDataCollator(
        data_source='DrugBank',
        split="train",
        split_method='split_by_classes',
        repeat=None,
        num_labels=num_labels,
        drug_metadata=drug_metadata,
        all_molecules=all_molecules,
        all_kg_data=train_kg_data,
        kg_sampling_num_neighbors=kg_sampling_num_neighbors,
        kg_sampling_num_layers=kg_sampling_num_layers,
        cv_df=cv_df,
        tx_df=tx_df,
    )

    eval_collator = BatchedDDIDataCollator(
        data_source='DrugBank',
        split="eval",
        split_method='split_by_classes',
        repeat=None,
        num_labels=num_labels,
        drug_metadata=drug_metadata,
        all_molecules=all_molecules,
        all_kg_data=train_kg_data,
        kg_sampling_num_neighbors=kg_sampling_num_neighbors,
        kg_sampling_num_layers=kg_sampling_num_layers,
        cv_df=cv_df,
        tx_df=tx_df,
    )

    return train_collator, eval_collator


def get_dataloaders(
    train_dataset,
    eval_dataset,
    train_collator,
    eval_collator,
    train_batch_size,
    eval_batch_size,
    weighted_sampling
):
    if weighted_sampling:
        pos_neg = torch.from_numpy(train_dataset.pos_neg)
        num_zeros = (pos_neg == 0).sum().item()
        num_ones = (pos_neg == 1).sum().item()
        
        weights = torch.tensor([num_zeros, num_ones])
        
        weights = 1/weights
        samples_weight = torch.tensor([weights[t] for t in pos_neg.int()]).double()
        
        num_to_draw = 2*(len(samples_weight) // 3)
        sampler = WeightedRandomSampler(samples_weight, num_to_draw, replacement=False)
        
    else:
        sampler = None
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        collate_fn=train_collator,
        num_workers=0,
        sampler=sampler,
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, eval_loader
