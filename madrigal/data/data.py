import os, pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.data import HeteroData
from torchdrug.data import PackedMolecule

from .data_utils import (
    structured_negative_sampling_multilabel,
    sample_kg_data,
    remove_edges_attached_to_nodes,
)
from ..utils import (
    CELL_LINES, 
    NUM_NON_TX_MODALITIES,
    NON_TX_MODALITIES,
    SEED, 
    CL_CKPT_DIR,
    get_pretrain_masks,
)


######
# Data utils for training
######
def get_datasets(args):
    """
    Get datasets for DDI training
    """
    train_dataset = LongDDIDataset(args.path_base, data_source=args.data_source, split='train', split_method=args.split_method, repeat=args.repeat)
    if args.split_method.startswith('split_by_drugs'):
        val_datasets = [
            LongDDIDataset(args.path_base, data_source=args.data_source, split='val_between', split_method=args.split_method, repeat=args.repeat), 
            LongDDIDataset(args.path_base, data_source=args.data_source, split='val_within', split_method=args.split_method, repeat=args.repeat)
        ]
        test_datasets = [
            LongDDIDataset(args.path_base, data_source=args.data_source, split='test_between', split_method=args.split_method, repeat=args.repeat), 
            LongDDIDataset(args.path_base, data_source=args.data_source, split='test_within', split_method=args.split_method, repeat=args.repeat)
        ]
    else:
        val_datasets = [
            LongDDIDataset(args.path_base, data_source=args.data_source, split='val', split_method=args.split_method, repeat=args.repeat)
        ]
        test_datasets = [
            LongDDIDataset(args.path_base, data_source=args.data_source, split='test', split_method=args.split_method, repeat=args.repeat)
        ]

    return train_dataset, val_datasets, test_datasets


def get_datasets_all_train(args):
    """
    Get datasets for final DDI score generation
    """
    train_dataset = LongDDIDatasetAllTrain(args.path_base, data_source=args.data_source)
    return train_dataset


def get_collators(args, drug_metadata, all_molecules, all_kg_data, train_kg_data, tabular_mod_dfs, tx_df, num_labels):
    """
    Get collators for DDI training
    """        
    train_collator = LongDDIDataCollator(
        path_base=args.path_base, 
        data_source=args.data_source,
        split='train', 
        split_method=args.split_method, 
        repeat=args.repeat,
        num_labels=num_labels,
        drug_metadata=drug_metadata,
        all_molecules=all_molecules,
        all_kg_data=train_kg_data,
        kg_sampling_num_neighbors=args.kg_sampling_num_neighbors,
        kg_sampling_num_layers=args.kg_sampling_num_layers,
        tabular_mod_dfs=tabular_mod_dfs,
        tx_df=tx_df,
        num_negative_samples_per_pair=args.num_negative_samples_per_pair, 
        negative_sampling_probs_type=args.negative_sampling_probs_type, 
    )
    if args.split_method.startswith('split_by_drugs'):
        val_collators = [
            LongDDIDataCollator(
                path_base=args.path_base, 
                data_source=args.data_source,
                split='val_between', 
                split_method=args.split_method, 
                repeat=args.repeat,
                num_labels=num_labels,
                drug_metadata=drug_metadata,
                all_molecules=all_molecules,
                all_kg_data=train_kg_data,
                kg_sampling_num_neighbors=None,
                kg_sampling_num_layers=None,
                tabular_mod_dfs=tabular_mod_dfs,
                tx_df=tx_df,
                num_negative_samples_per_pair=None, 
                negative_sampling_probs_type=None, 
            ), 
            LongDDIDataCollator(
                path_base=args.path_base, 
                data_source=args.data_source,
                split='val_within', 
                split_method=args.split_method, 
                repeat=args.repeat,
                num_labels=num_labels,
                drug_metadata=drug_metadata,
                all_molecules=all_molecules,
                all_kg_data=train_kg_data,
                kg_sampling_num_neighbors=None,
                kg_sampling_num_layers=None,
                tabular_mod_dfs=tabular_mod_dfs,
                tx_df=tx_df,
                num_negative_samples_per_pair=None, 
                negative_sampling_probs_type=None, 
            )
        ]
        test_collators = [
            LongDDIDataCollator(
                path_base=args.path_base, 
                data_source=args.data_source,
                split='test_between', 
                split_method=args.split_method, 
                repeat=args.repeat,
                num_labels=num_labels,
                drug_metadata=drug_metadata,
                all_molecules=all_molecules, 
                all_kg_data=all_kg_data,  # NOTE: This is different from train/val
                kg_sampling_num_neighbors=None,
                kg_sampling_num_layers=None,
                tabular_mod_dfs=tabular_mod_dfs,
                tx_df=tx_df,
                num_negative_samples_per_pair=None, 
                negative_sampling_probs_type=None, 
            ), 
            LongDDIDataCollator(
                path_base=args.path_base, 
                data_source=args.data_source,
                split='test_within', 
                split_method=args.split_method,
                repeat=args.repeat, 
                num_labels=num_labels,
                drug_metadata=drug_metadata,
                all_molecules=all_molecules, 
                all_kg_data=all_kg_data,  # NOTE: This is different from train/val
                kg_sampling_num_neighbors=None,
                kg_sampling_num_layers=None,
                tabular_mod_dfs=tabular_mod_dfs,
                tx_df=tx_df,
                num_negative_samples_per_pair=None, 
                negative_sampling_probs_type=None, 
            )
        ]
    else:
        val_collators = [
            LongDDIDataCollator(
                path_base=args.path_base,
                data_source=args.data_source,
                split='val', 
                split_method=args.split_method, 
                repeat=args.repeat,
                num_labels=num_labels,
                drug_metadata=drug_metadata,
                all_molecules=all_molecules,
                all_kg_data=train_kg_data, 
                kg_sampling_num_neighbors=None,
                kg_sampling_num_layers=None,
                tabular_mod_dfs=tabular_mod_dfs,
                tx_df=tx_df,
                num_negative_samples_per_pair=None, 
                negative_sampling_probs_type=None, 
            )
        ]
        test_collators = [
            LongDDIDataCollator(
                path_base=args.path_base,
                data_source=args.data_source,
                split='test', 
                split_method=args.split_method, 
                repeat=args.repeat,
                num_labels=num_labels,
                drug_metadata=drug_metadata,
                all_molecules=all_molecules, 
                all_kg_data=all_kg_data,
                kg_sampling_num_neighbors=None,
                kg_sampling_num_layers=None,
                tabular_mod_dfs=tabular_mod_dfs,
                tx_df=tx_df,
                num_negative_samples_per_pair=None, 
                negative_sampling_probs_type=None, 
            )
        ]
        
    return train_collator, val_collators, test_collators


def get_collators_all_train(args, drug_metadata, all_molecules, all_kg_data, train_kg_data, tabular_mod_dfs, tx_df, num_labels):
    """
    Get collators for DDI training
    """        
    train_collator = LongDDIDataCollator(
        path_base=args.path_base, 
        data_source=args.data_source,
        split="train", 
        split_method="split_by_pairs", 
        repeat=None,
        num_labels=num_labels,
        drug_metadata=drug_metadata,
        all_molecules=all_molecules,
        all_kg_data=train_kg_data,
        kg_sampling_num_neighbors=args.kg_sampling_num_neighbors,
        kg_sampling_num_layers=args.kg_sampling_num_layers,
        tabular_mod_dfs=tabular_mod_dfs,
        tx_df=tx_df,
        num_negative_samples_per_pair=args.num_negative_samples_per_pair, 
        negative_sampling_probs_type=args.negative_sampling_probs_type, 
    )        
    return train_collator


def get_dataloaders(args, train_dataset, val_datasets, test_datasets, train_collator, val_collators, test_collators, train_batch_size, val_batch_sizes, test_batch_sizes):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loaders = [DataLoader(
        dataset=val_datasets[i], 
        batch_size=val_batch_sizes[i], 
        shuffle=False, 
        collate_fn=val_collators[i], 
        num_workers=args.num_workers, 
        pin_memory=True
    ) for i in range(len(val_datasets))]
    test_loaders = [DataLoader(
        dataset=test_datasets[i], 
        batch_size=test_batch_sizes[i], 
        shuffle=False, 
        collate_fn=test_collators[i], 
        num_workers=args.num_workers, 
        pin_memory=True
    ) for i in range(len(test_datasets))]
    
    return train_loader, val_loaders, test_loaders


def get_dataloaders_all_train(args, train_dataset, train_collator, train_batch_size):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader


def get_pretrain_data(args, hparams):
    """ Get all pretraining data
    NOTE: Pretraining should always be based on split-by-drugs (but random, easy, and hard should be different) -- we use only drugs not in the val/test set (technically we can also include val drugs, but during model development, we don't want to overfit to the val set).  
    """    
    # load drug metadata, get pretraining drugs (drugs with more than one views available)
    drug_metadata = pd.read_pickle(os.path.join(args.path_base, 'drug_features/drug_metadata_ddi.pkl'))
    drug_metadata['view_str'] = 1  # all drugs must have structure (filtered already during preprocessing)
    mod_avail_df = drug_metadata[[f'view_{mod}' for mod in NON_TX_MODALITIES] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
    pretrain_drugs = mod_avail_df[mod_avail_df.sum(axis=1) >= 2].index.values  # only drugs with at least 2 views are used for pretraining
    
    if 'drugs' in args.split_method:
        # NOTE: ddi-val and ddi-test drugs are treated as val drugs for CL, 
        initial_val_between_drugs = pd.read_csv(os.path.join(args.path_base, f"drug_combination_data/{args.data_source}/{args.split_method}/val_between_df.csv"))['head'].unique()
        initial_val_within_drugs = np.unique(pd.read_csv(os.path.join(args.path_base, f"drug_combination_data/{args.data_source}/{args.split_method}/val_within_df.csv"))[['head', 'tail']].values)
        initial_test_between_drugs = pd.read_csv(os.path.join(args.path_base, f"drug_combination_data/{args.data_source}/{args.split_method}/test_between_df.csv"))['head'].unique()
        initial_test_within_drugs = np.unique(pd.read_csv(os.path.join(args.path_base, f"drug_combination_data/{args.data_source}/{args.split_method}/test_within_df.csv"))[['head', 'tail']].values)
        initial_eval_drugs = np.unique(np.concatenate([initial_val_between_drugs, initial_val_within_drugs, initial_test_between_drugs, initial_test_within_drugs]))
        test_drugs = np.unique(np.concatenate([initial_test_between_drugs, initial_test_within_drugs]))
        
        train_drugs = pretrain_drugs[~np.isin(pretrain_drugs, initial_eval_drugs)]
        val_drugs = pretrain_drugs[np.isin(pretrain_drugs, initial_eval_drugs) & (~np.isin(pretrain_drugs, test_drugs))]
        if val_drugs.shape[0] == 0:
            train_drugs, val_drugs = train_test_split(pretrain_drugs, test_size=0.1, random_state=SEED)
        
    else:
        train_drugs, val_drugs = train_test_split(pretrain_drugs, test_size=0.1, random_state=SEED)  # 10% of pretraining drugs are left out for validation
        test_drugs = None
    
    if False:
        pretrain_compound_sims_str = np.load(args.path_base + '/pretrain_compound_str_morgan_tanimoto.npy')
        pretrain_compound_sims_kg = np.load(args.path_base + '/pretrain_compound_kg_jaccard.npy')
        pretrain_compound_sims_cv = np.load(args.path_base + '/pretrain_compound_cv_pearson.npy')
        
        pretrain_compound_sims_tx_all_cell_lines = {}
        for cell_line in CELL_LINES:
            pretrain_compound_sims_tx_all_cell_lines[cell_line] = np.load(args.path_base + f'/pretrain_compound_tx_{cell_line}_pearson.npy')
    else:
        pretrain_compound_sims_str = np.zeros((drug_metadata.shape[0], drug_metadata.shape[0]))
        pretrain_compound_sims_kg = np.zeros((drug_metadata.shape[0], drug_metadata.shape[0]))
        pretrain_compound_sims_cv = np.zeros((drug_metadata.shape[0], drug_metadata.shape[0]))
        
        pretrain_compound_sims_tx_all_cell_lines = {}
        for cell_line in CELL_LINES:
            pretrain_compound_sims_tx_all_cell_lines[cell_line] = np.zeros((drug_metadata.shape[0], drug_metadata.shape[0]))
    
    
    
    
    hard_negative_mask = None

    # Get possible modality subset masks for all drugs (depending on their own modality availability)
    masks = 1 - mod_avail_df.values  # NOTE: convert to mask for Transformer, where 1 = NOT available (masked), 0 = available (as usual)
    all_train_subset_masks = get_pretrain_masks(train_drugs, masks[train_drugs], args.pretrain_mode, args.pretrain_unbalanced, args.pretrain_tx_downsample_ratio)  # NOTE: returns a dictionary of possible subset masks for each drug
    
    # Get extra negative molecules for pretraining
    if args.extra_str_neg_mol_num > 0:
        all_extra_molecules = torch.load(os.path.join(args.path_base, 'all_ChEMBL_molecules_torchdrug.pt'))
        extra_mol_str_masks = torch.ones(masks.shape[1]).repeat(args.extra_str_neg_mol_num, 1).bool()
        extra_mol_str_masks[:, 0] = False
    else:
        all_extra_molecules = None
        extra_mol_str_masks = None
        
    train_dataset = TensorDataset(torch.from_numpy(train_drugs))
    val_dataset = TensorDataset(torch.from_numpy(val_drugs))
    all_dataset = TensorDataset(torch.from_numpy(np.concatenate([train_drugs, val_drugs])))
    
    collator = DrugCLCollator(args.path_base, mod_avail_df, args.kg_encoder, args.kg_sampling_num_neighbors, args.kg_sampling_num_layers, test_drugs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=args.drop_last, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.pretrain_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=args.drop_last, collate_fn=collator)
    pretrain_loader = DataLoader(all_dataset, batch_size=args.pretrain_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=args.drop_last, collate_fn=collator)
    
    kg_args = {
        'kg_encoder': args.kg_encoder,
        'kg_sampling_num_neighbors': args.kg_sampling_num_neighbors,
        'kg_sampling_num_layers': args.kg_sampling_num_layers,
    }

    return train_drugs, val_drugs, pretrain_drugs, train_loader, val_loader, pretrain_loader, collator, masks, all_train_subset_masks, hard_negative_mask, all_extra_molecules, extra_mol_str_masks, kg_args


def get_train_data(args, logger=None, eval_mode=False):
    # Ensure KG input args in finetune are the same as in the pretrained checkpoint
    if args.checkpoint is not None:
        if eval_mode:
            full_ckpt_path = args.checkpoint
        else:
            full_ckpt_path = CL_CKPT_DIR + args.checkpoint
        checkpoint = torch.load(full_ckpt_path, map_location='cpu')
        if 'kg_args' in checkpoint.keys():    
            args.kg_encoder = checkpoint['kg_args']['kg_encoder']
            args.kg_sampling_num_neighbors = checkpoint['kg_args']['kg_sampling_num_neighbors']
            args.kg_sampling_num_layers = checkpoint['kg_args']['kg_sampling_num_layers']
        else:
            if logger is not None:
                logger.info(f'No kg_args in checkpoint {full_ckpt_path}, using default or provided args')
        if logger is not None:
            logger.info(f'KG input params\nkg_encoder: {full_ckpt_path}\nkg_sampling_num_neighbors: {args.kg_sampling_num_neighbors}\nkg_sampling_num_layers: {args.kg_sampling_num_layers}\n')
    
    # load drug metadata
    drug_metadata = pd.read_pickle(os.path.join(args.path_base, 'drug_features/drug_metadata_ddi.pkl'))
    drug_metadata['view_str'] = 1  # all drugs must have structure (filtered already during preprocessing)
    
    # load all structure modality data
    all_molecules = torch.load(os.path.join(args.path_base, 'drug_features/str/all_molecules_torchdrug.pt'), map_location='cpu')
    
    # load all KG modality data
    all_kg_data = torch.load(os.path.join(args.path_base, f'drug_features/kg/KG_data_{args.kg_encoder}.pt'), map_location='cpu')
    
    # load tabular drug data
    assert NON_TX_MODALITIES[0] == "str"
    assert NON_TX_MODALITIES[1] == "kg"
    tabular_mod_dfs = {}
    for mod in NON_TX_MODALITIES[2:]:
        tabular_mod_dfs[mod] = pd.read_csv(args.path_base + f"drug_features/{mod}/{mod}_cp_data.csv", index_col=0)
    
    tx_df = pd.read_csv(args.path_base + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)
    
    # load label map
    with open(args.path_base + f'drug_combination_data/{args.data_source}/{args.data_source.lower()}_ddi_directed_label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    # load datasets
    train_dataset, val_datasets, test_datasets = get_datasets(args)
    
    if 'drugs' in args.split_method:
        test_drugs = np.unique(np.concatenate([
            test_datasets[0].edge_df['head'].unique(),
            np.unique(test_datasets[1].edge_df[['head', 'tail']].values)
        ]))
        train_kg_data = remove_edges_attached_to_nodes(all_kg_data, torch.from_numpy(test_drugs).long())
    else:
        train_kg_data = all_kg_data
    
    # load collators
    train_collator, val_collators, test_collators = get_collators(args, drug_metadata, all_molecules, all_kg_data, train_kg_data, tabular_mod_dfs, tx_df, train_dataset.num_labels)
    
    train_batch_size = len(train_dataset) if args.batch_size is None else args.batch_size  # full batch training; if batched, need to implement negative sampling
    val_batch_sizes = [len(val_dataset) if args.batch_size is None else args.batch_size for val_dataset in val_datasets]  # use -1 so that we can use the same code for both split_by_drugs and split_by_pairs
    test_batch_sizes = [len(test_dataset) if args.batch_size is None else args.batch_size for test_dataset in test_datasets]
    
    train_loader, val_loaders, test_loaders = get_dataloaders(args, train_dataset, val_datasets, test_datasets, train_collator, val_collators, test_collators, train_batch_size, val_batch_sizes, test_batch_sizes)
    
    return all_kg_data, train_loader, val_loaders, test_loaders, train_collator, train_dataset, val_datasets, test_datasets, label_map


def get_all_drugs_data(args, add_specific_drugs=None):
    # Ensure KG input args in finetune are the same as in the pretrained checkpoint
    if args.checkpoint is not None:
        full_ckpt_path = args.checkpoint  # NOTE: not using CL_CKPT_DIR here because here we are not loading CL-pretrained model
        checkpoint = torch.load(full_ckpt_path, map_location='cpu')
        if "kg_args" in checkpoint.keys():
            args.kg_encoder = checkpoint['kg_args']['kg_encoder']
            args.kg_sampling_num_neighbors = checkpoint['kg_args']['kg_sampling_num_neighbors']
            args.kg_sampling_num_layers = checkpoint['kg_args']['kg_sampling_num_layers']
        else:
            print(f'No kg_args in checkpoint {full_ckpt_path}, using default or provided args')
    
    # load drug metadata and structure modality data
    if add_specific_drugs is None:
        drug_metadata = pd.read_pickle(os.path.join(args.path_base, "drug_features/drug_metadata_ddi.pkl"))
        assert drug_metadata.shape[0] >= args.first_num_drugs
        drug_metadata = drug_metadata.iloc[:args.first_num_drugs, :]
        all_molecules = torch.load(os.path.join(args.path_base, "drug_features/str/all_molecules_torchdrug.pt"), map_location="cpu")[:args.first_num_drugs]
    else:
        drug_metadata = pd.read_pickle(os.path.join(args.path_base, f"drug_features/drug_metadata_{add_specific_drugs}.pkl"))
        assert drug_metadata.shape[0] >= args.first_num_drugs
        drug_metadata = drug_metadata.iloc[:args.first_num_drugs, :]
        all_molecules = torch.load(os.path.join(args.path_base, f"drug_features/str/{add_specific_drugs}_molecules_torchdrug.pt"), map_location="cpu")[:args.first_num_drugs]
        print(f"Using DrugBank + {add_specific_drugs} metadata")
    
    drug_metadata["view_str"] = 1  # all drugs must have structure (filtered already during preprocessing)
    
    # load all KG modality data
    all_kg_data = torch.load(os.path.join(args.path_base, f"drug_features/kg/KG_data_{args.kg_encoder}.pt"), map_location="cpu")
    
    # load tabular drug data
    assert NON_TX_MODALITIES[0] == "str"
    assert NON_TX_MODALITIES[1] == "kg"
    tabular_mod_dfs = {}
    for mod in NON_TX_MODALITIES[2:]:
        tabular_mod_dfs[mod] = pd.read_csv(args.path_base + f"drug_features/{mod}/{mod}_cp_data.csv", index_col=0)
    tx_df = pd.read_csv(args.path_base + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)
    
    # load label map
    with open(args.path_base + f'drug_combination_data/{args.data_source}/{args.data_source.lower()}_ddi_directed_label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    # load datasets
    all_drugs_dataset = EvalDDIDataset(args.path_base, data_source=args.data_source, split='test', split_method=args.split_method, repeat=args.repeat, first_num_drugs=args.first_num_drugs)
    train_kg_data = all_kg_data
    
    # load collators
    test_collator = get_collators_all_train(args, drug_metadata, all_molecules, all_kg_data, train_kg_data, tabular_mod_dfs, tx_df, all_drugs_dataset.num_labels)
    test_batch_size = len(all_drugs_dataset) if args.batch_size is None else args.batch_size
    test_loader = get_dataloaders_all_train(args, all_drugs_dataset, test_collator, test_batch_size)
    
    return all_kg_data, test_loader, test_collator, all_drugs_dataset, label_map


def get_train_data_for_all_train(args, logger=None, eval_mode=False):
    # Ensure KG input args in finetune are the same as in the pretrained checkpoint
    if args.checkpoint is not None:
        if eval_mode:
            full_ckpt_path = args.checkpoint
        else:
            full_ckpt_path = CL_CKPT_DIR + args.checkpoint
        checkpoint = torch.load(full_ckpt_path, map_location='cpu')
        if 'kg_args' in checkpoint.keys():    
            args.kg_encoder = checkpoint['kg_args']['kg_encoder']
            args.kg_sampling_num_neighbors = checkpoint['kg_args']['kg_sampling_num_neighbors']
            args.kg_sampling_num_layers = checkpoint['kg_args']['kg_sampling_num_layers']
        else:
            if logger is not None:
                logger.info(f'No kg_args in checkpoint {full_ckpt_path}, using default or provided args')
        if logger is not None:
            logger.info(f'KG input params\nkg_encoder: {full_ckpt_path}\nkg_sampling_num_neighbors: {args.kg_sampling_num_neighbors}\nkg_sampling_num_layers: {args.kg_sampling_num_layers}\n')
    
    # load drug metadata
    drug_metadata = pd.read_pickle(os.path.join(args.path_base, 'drug_features/drug_metadata_ddi.pkl'))
    drug_metadata['view_str'] = 1  # all drugs must have structure (filtered already during preprocessing)
    
    # load all structure modality data
    all_molecules = torch.load(os.path.join(args.path_base, 'drug_features/str/all_molecules_torchdrug.pt'), map_location='cpu')
    
    # load all KG modality data
    all_kg_data = torch.load(os.path.join(args.path_base, f'drug_features/kg/KG_data_{args.kg_encoder}.pt'), map_location='cpu')
    
    # load tabular drug data
    assert NON_TX_MODALITIES[0] == "str"
    assert NON_TX_MODALITIES[1] == "kg"
    tabular_mod_dfs = {}
    for mod in NON_TX_MODALITIES[2:]:
        tabular_mod_dfs[mod] = pd.read_csv(args.path_base + f"drug_features/{mod}/{mod}_cp_data.csv", index_col=0)
    tx_df = pd.read_csv(args.path_base + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)
    
    # load label map
    with open(args.path_base + f'drug_combination_data/{args.data_source}/{args.data_source.lower()}_ddi_directed_label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    # load datasets
    train_dataset = get_datasets_all_train(args)
    train_kg_data = all_kg_data

    # load collators
    train_collator = get_collators_all_train(args, drug_metadata, all_molecules, all_kg_data, train_kg_data, tabular_mod_dfs, tx_df, train_dataset.num_labels)
    train_batch_size = len(train_dataset) if args.batch_size is None else args.batch_size  # full batch training; if batched, need to implement negative sampling
    
    train_loader = get_dataloaders_all_train(args, train_dataset, train_collator, train_batch_size)
    
    return all_kg_data, train_loader, train_collator, train_dataset, label_map


############################################
# DDI datasets & data-collators
############################################
class LongDDIDataset(Dataset):
    """ Load positive pairs from a long DDI table
    Long so that each row is a pair of drug indices.
    """
    def __init__(
        self, 
        path_base: str, 
        data_source: str,
        split: str,
        split_method: str, 
        repeat: str = None,
        num_base_labels: int = 0,
    ):
        """
        Args:
            path_base: Path to the directory containing the edgelist.
            split: One of "train", "val", "test", "val_within", "test_within", "val_between", "test_between".
            split_method: One of "ddi_centric", "drug_centric_hard", "drug_centric_easy". 
        """
        self.split = split
        if repeat is not None:
            edgelist_df_fname = f"drug_combination_data/{data_source}/{split_method}/{repeat}/{split}_df.csv"
        else:
            edgelist_df_fname = f"drug_combination_data/{data_source}/{split_method}/{split}_df.csv"
        self.edge_df = pd.read_csv(os.path.join(path_base, edgelist_df_fname))
        self.edgelist = self.edge_df[['head', 'tail']].values.tolist()
        self.labels = self.edge_df['label_indexed'].values
        self.num_labels = max(self.labels) + 1 if max(self.labels) > 1 else 1
        if self.split in {'val_between', 'test_between'}:
            self.neg_tails_1 = self.edge_df['neg_tail_1'].values.tolist()
            self.neg_tails_2 = self.edge_df['neg_tail_2'].values.tolist()
        else:
            self.neg_heads = self.edge_df['neg_head'].values.tolist()
            self.neg_tails = self.edge_df['neg_tail'].values.tolist()
            
        if num_base_labels > 0:
            self.labels = self.labels + num_base_labels
        
        # NOTE: All those edgelists should be directed by design.
        temp = self.edge_df[['head', 'tail', 'label_indexed']]
        assert pd.concat([temp, temp.rename(columns={'head':'tail', 'tail':'head'})]).drop_duplicates().shape[0] == 2 * temp.shape[0]  # check if strctly directed
    
    def __getitem__(self, index: int):
        positive_pair = self.edgelist[index]
        label = self.labels[index]
        if self.split in {'val_between', 'test_between'}:
            neg_tail_1 = self.neg_tails_1[index]
            neg_tail_2 = self.neg_tails_2[index]
            return positive_pair, label, neg_tail_1, neg_tail_2
        else:
            neg_head = self.neg_heads[index]
            neg_tail = self.neg_tails[index]
            return positive_pair, label, neg_head, neg_tail
    
    def __len__(self):
        return len(self.edgelist)


class SingleDrugDataset(Dataset):
    def __init__(
        self, 
        path_base: str, 
        data_source: str,
        split: str,
        split_method: str, 
        repeat: str = None,
    ):
        """
        Args:
            `path_base`: Path to the directory containing the edgelist. 
        """
        single_sides_df = pd.read_csv(os.path.join(path_base, f"single_drug/{data_source}/{split_method}/{split}_df.csv"))
        
        # Concatenate the datasets with the specified ratios, ensuring that TWOSIDES is the first one and is 1
        self.edge_df = single_sides_df.query("label_indexed < 100")
        self.edgelist = self.edge_df[['head', 'tail']].values.tolist()
        self.labels = self.edge_df['label_indexed'].tolist()
        self.num_labels = max(self.labels) + 1 if max(self.labels) > 1 else 1
        
        self.neg_heads = self.edge_df['neg_head'].tolist()
        self.neg_tails = self.edge_df['neg_tail'].tolist()
        
        # NOTE: There is no "direction" for single drugs
        temp = self.edge_df[['head', 'tail', 'label_indexed']]
        assert pd.concat([temp, temp.rename(columns={'head':'tail', 'tail':'head'})]).drop_duplicates().shape[0] == temp.shape[0]
    
    def __getitem__(self, index: int):
        positive_pair = self.edgelist[index]
        label = self.labels[index]
        
        neg_head = self.neg_heads[index]
        neg_tail = self.neg_tails[index]
        return positive_pair, label, neg_head, neg_tail
    
    def __len__(self):
        return len(self.edgelist)


class LongDDIDatasetAllTrain(Dataset):
    """ Load positive pairs from a long DDI table
    Long so that each row is a pair of drug indices.
    """
    def __init__(
        self, 
        path_base: str, 
        data_source: str,
    ):
        """
        Args:
            path_base: Path to the directory containing the edgelist.
        """
        # NOTE: Using split-by-pairs because the negative sampling strategy for val and test is the same as train there
        train_df_fname = f"drug_combination_data/{data_source}/split_by_pairs/train_df.csv"
        val_df_fname = f"drug_combination_data/{data_source}/split_by_pairs/val_df.csv"
        test_df_fname = f"drug_combination_data/{data_source}/split_by_pairs/test_df.csv"
        train_df = pd.read_csv(os.path.join(path_base, train_df_fname))
        val_df = pd.read_csv(os.path.join(path_base, val_df_fname))
        test_df = pd.read_csv(os.path.join(path_base, test_df_fname))
        
        self.edge_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True).reset_index(drop=True)
        self.edgelist = self.edge_df[['head', 'tail']].values.tolist()
        self.labels = self.edge_df['label_indexed'].values.tolist()
        self.num_labels = max(self.labels) + 1 if max(self.labels) > 1 else 1
        self.neg_heads = self.edge_df['neg_head'].values.tolist()
        self.neg_tails = self.edge_df['neg_tail'].values.tolist()
        
        # NOTE: All those edgelists should be directed by design.
        temp = self.edge_df[['head', 'tail', 'label_indexed']]
        assert pd.concat([temp, temp.rename(columns={'head':'tail', 'tail':'head'})]).drop_duplicates().shape[0] == 2 * temp.shape[0]  # check if strctly directed
    
    def __getitem__(self, index: int):
        positive_pair = self.edgelist[index]
        label = self.labels[index]
        neg_head = self.neg_heads[index]
        neg_tail = self.neg_tails[index]
        return positive_pair, label, neg_head, neg_tail
    
    def __len__(self):
        return len(self.edgelist)
    
    
class EvalDDIDataset(Dataset):
    """ Load positive pairs from a long DDI table
    Long so that each row is a pair of drug indices.
    """
    def __init__(
        self, 
        path_base: str, 
        data_source: str,
        split: str,
        split_method: str, 
        repeat: str = None,
        first_num_drugs: int = 6843,
        label_range: tuple = None,
    ):
        """
        Args:
            path_base: Path to the directory containing the edgelist.
            split: One of "train", "val", "test", "val_within", "test_within", "val_between", "test_between".
            split_method: One of "ddi_centric", "drug_centric_hard", "drug_centric_easy". 
        """
        if repeat is not None:
            train_ddi_df = pd.read_csv(os.path.join(path_base, f"drug_combination_data/{data_source}/{split_method}/{repeat}/train_df.csv"))
        else:
            train_ddi_df = pd.read_csv(os.path.join(path_base, f"drug_combination_data/{data_source}/{split_method}/train_df.csv"))
        all_unique_labels = train_ddi_df['label_indexed'].unique() if label_range is None else train_ddi_df['label_indexed'].unique()[label_range[0], label_range[1]]
        self.edge_df = pd.DataFrame({
            'head': np.arange(first_num_drugs),
            'tail': np.arange(first_num_drugs),
            'label_indexed': np.array(
                all_unique_labels.tolist() + \
                np.random.choice(
                    all_unique_labels, 
                    first_num_drugs - all_unique_labels.shape[0], 
                    replace=True
                ).tolist()
            ),
            'neg_head': np.arange(first_num_drugs),
            'neg_tail': np.arange(first_num_drugs),
        })
        self.edgelist = self.edge_df[['head', 'tail']].values.tolist()
        self.labels = self.edge_df['label_indexed'].values.tolist()
        self.num_labels = max(self.labels) + 1 if max(self.labels) > 1 else 1
        self.neg_heads = self.edge_df['neg_head'].values.tolist()
        self.neg_tails = self.edge_df['neg_tail'].values.tolist()
        
        # NOTE: All those edgelists should be directed by design.
        temp = self.edge_df[['head', 'tail', 'label_indexed']]
        print(temp)

    def __getitem__(self, index: int):
        positive_pair = self.edgelist[index]
        label = self.labels[index]
        
        neg_head = self.neg_heads[index]
        neg_tail = self.neg_tails[index]
        return positive_pair, label, neg_head, neg_tail
    
    def __len__(self):
        return len(self.edgelist)


class LongDDIDataCollator:
    def __init__(
        self, 
        path_base: str, 
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
        tabular_mod_dfs: pd.DataFrame,
        tx_df: pd.DataFrame,
        num_negative_samples_per_pair: int = None,
        negative_sampling_probs_type: str = 'uniform',
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
        self.num_negative_samples_per_pair = num_negative_samples_per_pair
        assert num_negative_samples_per_pair is None or num_negative_samples_per_pair <= 2  # only None or 1/2
        assert (split == 'train') or (num_negative_samples_per_pair is None)  # must load fixed negatives for val and test
        self.split = split
        self.num_labels = num_labels
        self.all_molecules = all_molecules
        self.all_kg_data = all_kg_data
        self.kg_sampling_num_neighbors = kg_sampling_num_neighbors
        self.kg_sampling_num_layers = kg_sampling_num_layers
        self.str_node_feat_dim = self.all_molecules.node_feature.shape[1]
        
        # load perturbation data
        self.tabular_mod_dfs = tabular_mod_dfs
        self.tx_df = tx_df
        
        # prepare modality masks
        self.drug_metadata = drug_metadata
        
        # if needs negative sampling on-the-fly
        # NOTE: In this case, split == 'train'. Also note that for train set, we theoretically shouldn't know the val/test edges, so we cannot filter them out when negative sampling, i.e. valid indices and ground truth edgelist should both come from only the train edgelist
        if self.num_negative_samples_per_pair:
            if repeat is not None:
                train_edge_df_fname = f"drug_combination_data/{data_source}/{split_method}/{repeat}/train_df.csv"
            else:
                train_edge_df_fname = f"drug_combination_data/{data_source}/{split_method}/train_df.csv"
            train_edge_df = pd.read_csv(os.path.join(path_base, train_edge_df_fname))
            self.valid_indices = np.unique(train_edge_df[['head', 'tail']])
            self.other_ground_truth_edgelist = train_edge_df[['head', 'tail']].values
            self.other_ground_truth_labels = train_edge_df['label_indexed'].values
            
            # get probability distribution of nodes for negative sampling during training
            if negative_sampling_probs_type == 'uniform':
                self.negative_sampling_probs = None
            elif negative_sampling_probs_type in {'degree', 'degree_w2v'}:
                self.negative_sampling_probs = np.bincount(train_edge_df[['head', 'tail']].values.flatten(), minlength=self.valid_indices.max())  # NOTE: this probability distribution aligns with the indices in self.valid_indices
                if negative_sampling_probs_type == 'degree_w2v':
                    self.negative_sampling_probs = self.negative_sampling_probs ** 0.75
                self.negative_sampling_probs /= self.negative_sampling_probs.sum()
        else:
            pass
    
    def __call__(
        self, 
        batch: List[Tuple]
    ):
        """
        Args:
            batch: A list of tuples of (positive_pair, label, neg_head, neg_tail).
        """
        if self.split in {'val_between', 'test_between'}:
            positive_edgelist, positive_labels, neg_tails_1, neg_tails_2 = zip(*batch)
        else:
            positive_edgelist, positive_labels, neg_heads, neg_tails = zip(*batch)
        positive_edgelist = torch.tensor(positive_edgelist, dtype=torch.long)
        positive_labels = torch.tensor(positive_labels, dtype=torch.long)
        
        positive_heads = positive_edgelist[:, 0]
        positive_tails = positive_edgelist[:, 1]
        
        # first get negatives
        if self.num_negative_samples_per_pair:
            negative_edgelist, negative_labels = self._negative_sampling(positive_edgelist.numpy(), positive_labels.numpy(), self.valid_indices, self.other_ground_truth_edgelist, self.other_ground_truth_labels, self.num_negative_samples_per_pair, self.negative_sampling_probs)
        else:
            if self.split in {'val_between', 'test_between'}:
                negative_edgelist = torch.cat([
                    torch.stack([positive_heads, torch.tensor(neg_tails_1, dtype=torch.long)]),
                    torch.stack([positive_heads, torch.tensor(neg_tails_2, dtype=torch.long)]), 
                ], dim=1).T
            else:
                negative_edgelist = torch.cat([
                    torch.stack([positive_heads, torch.tensor(neg_tails, dtype=torch.long)]),
                    torch.stack([torch.tensor(neg_heads, dtype=torch.long), positive_tails]),
                ], dim=1).T
            negative_labels = positive_labels.repeat(2)
        
        # NOTE: Make the edges undirected for training
        if self.split == 'train':
            positive_edgelist = torch.cat([positive_edgelist, positive_edgelist.flip(1)], dim=0)
            negative_edgelist = torch.cat([negative_edgelist, negative_edgelist.flip(1)], dim=0)
            positive_labels = positive_labels.repeat(2)
            negative_labels = negative_labels.repeat(2)
        
        positive_heads = positive_edgelist[:, 0]
        positive_tails = positive_edgelist[:, 1]
        negative_heads = negative_edgelist[:, 0]
        negative_tails = negative_edgelist[:, 1]
        
        # then extract unique indices -- NOTE: We separate out head and tail because we might give head and tail different modality compositions during both training and evaluation. See below for procedure that ensures KG sampling is identical for heads and tails.
        unique_head_indices, all_heads_new = torch.unique(torch.cat([positive_heads, negative_heads], dim=0), return_inverse=True)
        unique_tail_indices, all_tails_new = torch.unique(torch.cat([positive_tails, negative_tails], dim=0), return_inverse=True)
        positive_heads_new, negative_heads_new = torch.split(all_heads_new, [len(positive_heads), len(negative_heads)])
        positive_tails_new, negative_tails_new = torch.split(all_tails_new, [len(positive_tails), len(negative_tails)])
        all_labels_new = torch.cat([positive_labels, negative_labels])
        all_pos_neg_new = torch.cat([torch.ones_like(positive_labels), torch.zeros_like(negative_labels)])
        
        # extract modality availability
        unique_head_mod_avail = self.drug_metadata.loc[unique_head_indices][['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        unique_tail_mod_avail = self.drug_metadata.loc[unique_tail_indices][['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        
        # extract structure modality (torchdrug mol graphs)
        # NOTE: KG modality is computed via HAN during encoding, by doing message passing across the KG and then         
        unique_head_mol_strs = self.all_molecules[unique_head_indices.tolist()]
        unique_tail_mol_strs = self.all_molecules[unique_tail_indices.tolist()]
        
        # NOTE: To ensure KG sampling is identical for the overlapping heads and tails, we first extract the unique indices, then sample the KG. The same subgraph will be fed into the KG encoder for both heads and tails, and the relevant embeddings will be extracted after encoding for heads and tails, respectively.
        unique_drug_indices = torch.unique(torch.cat([unique_head_indices, unique_tail_indices], dim=0))
        kg_dict = sample_kg_data(self.all_kg_data, unique_drug_indices, self.kg_sampling_num_neighbors, 'neighborloader', self.kg_sampling_num_layers, drug_only=True)
        
        # extract Cv, Tx signatures, fill dummies for unavailable ones
        def get_signatures_and_fill_dummy(unique_indices, sig_df, unique_mod_avail, sig, unique_sig_ids):
            unique_sig_output = torch.randn((unique_indices.shape[0], sig_df.shape[0]))
            unique_sig_avail_indices = unique_mod_avail[sig].values == 1
            unique_sig_output[unique_sig_avail_indices, :] = torch.from_numpy(sig_df[unique_sig_ids[unique_sig_avail_indices]].values.T).float()
            unique_sig_output[~unique_sig_avail_indices, :] = 0
            return unique_sig_output
        
        unique_head_tabular_mods = {}
        unique_tail_tabular_mods = {}
        for mod in NON_TX_MODALITIES[2:]:
            unique_head_mod_sig_ids = self.drug_metadata.loc[unique_head_indices.tolist(), f"{mod}_sig_id"].values
            unique_tail_mod_sig_ids = self.drug_metadata.loc[unique_tail_indices.tolist(), f"{mod}_sig_id"].values
            unique_head_tabular_mods[mod] = get_signatures_and_fill_dummy(unique_head_indices, self.tabular_mod_dfs[mod], unique_head_mod_avail, f"view_{mod}", unique_head_mod_sig_ids)
            unique_tail_tabular_mods[mod] = get_signatures_and_fill_dummy(unique_tail_indices, self.tabular_mod_dfs[mod], unique_tail_mod_avail, f"view_{mod}", unique_tail_mod_sig_ids)
        
        unique_head_tx_dict = {}
        unique_tail_tx_dict = {}
        for cell_line in CELL_LINES:
            unique_head_tx_dict[cell_line] = {}
            unique_tail_tx_dict[cell_line] = {}
            
            # sigs
            unique_head_tx_cell_line_sig_ids = self.drug_metadata.loc[unique_head_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_tail_tx_cell_line_sig_ids = self.drug_metadata.loc[unique_tail_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_head_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(unique_head_indices, self.tx_df, unique_head_mod_avail, f'view_tx_{cell_line}', unique_head_tx_cell_line_sig_ids)
            unique_tail_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(unique_tail_indices, self.tx_df, unique_tail_mod_avail, f'view_tx_{cell_line}', unique_tail_tx_cell_line_sig_ids)
            
            # drugs
            unique_head_tx_dict[cell_line]['drugs'] = unique_head_indices.long()
            unique_tail_tx_dict[cell_line]['drugs'] = unique_tail_indices.long()
            
            # dosages
            unique_head_tx_dosages = self.drug_metadata.loc[unique_head_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_tail_tx_dosages = self.drug_metadata.loc[unique_tail_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_head_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_head_tx_dosages).float()
            unique_tail_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_tail_tx_dosages).float()
            
            # cell_lines (in str, converted to one-hot )
            unique_head_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_head_indices))
            unique_tail_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_tail_indices))
        
        # NOTE: Instead of using a label matrix, we put all label-specific bilinear decoder weights together into a 3D matrix, and compute for each label all drugs-drugs predictions together (N_drug x feat, feat x feat x num_labels, N_drug x feat), masking the unwanted ones thereafter. This is memory efficient and can be done in a single forward pass.
        label_matrix = None
        
        # finally, construct the input drug's modality masks: reverse to mod avail, here 1 means NOT having the modality, 0 means having the modality (for the sake of key_padding_mask, thus inverse)
        unique_head_masks = torch.from_numpy(1 - unique_head_mod_avail.values).bool()
        unique_tail_masks = torch.from_numpy(1 - unique_tail_mod_avail.values).bool()
        
        return {
            # unique heads
            'head':{
                'drugs': unique_head_indices,
                'strs': unique_head_mol_strs,
                **unique_head_tabular_mods,
                'tx': unique_head_tx_dict,
                'masks': unique_head_masks,
            },
            # unique tails
            'tail':{
                'drugs': unique_tail_indices,
                'strs': unique_tail_mol_strs,
                **unique_tail_tabular_mods,
                'tx': unique_tail_tx_dict,
                'masks': unique_tail_masks,
            },
            # kg
            'kg': kg_dict,
            # used for indexing the label matrix
            'edge_indices':{
                'head': all_heads_new,
                'tail': all_tails_new,
                'label': all_labels_new,
                'pos_neg': all_pos_neg_new,
            },
        }
    
    def _negative_sampling(
        self, 
        positive_edgelist: np.ndarray, 
        positive_labels: np.ndarray,
        valid_indices: np.ndarray,
        other_ground_truth_edgelist: Optional[np.ndarray] = None, 
        other_ground_truth_labels: Optional[np.ndarray] = None,
        num_negative_samples_per_pair: Optional[int] = None,
        negative_sampling_probs: np.ndarray = None
    ):
        """ Sample negatives on-the-fly for the training set."""
        # NOTE: If only generate one negative sample, permutate tail.  Since this is essentially for training only, and the training edges are kept undirected (i.e. bidirectional), we are only sampling 1, not 2, negative pairs for each positive pair, and then **flip** each negative to still keep the pos:neg = 1:2 (and keep the data consistent for different encodings used for head and tail during training).
        if num_negative_samples_per_pair == 2:
            heads, tails, neg_heads, neg_tails = structured_negative_sampling_multilabel(
                edge_index=positive_edgelist.T, 
                label=positive_labels, 
                valid_negative_nodes=valid_indices, 
                other_ground_truth_edge_index=other_ground_truth_edgelist.T, 
                other_ground_truth_label=other_ground_truth_labels, 
                num_nodes=None, 
                contains_neg_self_loops=False, 
                two_sided=True,
                probs=negative_sampling_probs
            )
            positive_heads = positive_edgelist[:, 0]
            positive_tails = positive_edgelist[:, 1]
            negative_edgelist = torch.cat([
                torch.stack([torch.tensor(positive_heads, dtype=torch.long), torch.from_numpy(neg_tails).long()]),
                torch.stack([torch.from_numpy(neg_heads).long(), torch.tensor(positive_tails, dtype=torch.long)]),
            ], dim=1).T
            negative_labels = torch.from_numpy(positive_labels).long().repeat(2)
        elif num_negative_samples_per_pair == 1:
            raise NotImplementedError  # not supported
        else:
            raise NotImplementedError
        
        return negative_edgelist, negative_labels


class LongDDIAllDatasetsDataCollator:
    def __init__(
        self, 
        num_labels: int,
        drug_metadata: pd.DataFrame,
        all_molecules: PackedMolecule,
        all_kg_data: HeteroData,
        kg_sampling_num_neighbors: int,
        kg_sampling_num_layers: int,
        tabular_mod_dfs: Dict[str, pd.DataFrame],
        tx_df: pd.DataFrame,
        use_single_drug: bool = False,
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
        self.num_labels = num_labels
        self.all_molecules = all_molecules
        self.all_kg_data = all_kg_data
        self.kg_sampling_num_neighbors = kg_sampling_num_neighbors
        self.kg_sampling_num_layers = kg_sampling_num_layers
        self.str_node_feat_dim = self.all_molecules.node_feature.shape[1]
        
        # load perturbation data
        self.tabular_mod_dfs = tabular_mod_dfs
        self.tx_df = tx_df
        
        # prepare modality masks
        self.drug_metadata = drug_metadata
        
        self.use_single_drug = use_single_drug
    
    def __call__(
        self, 
        batch: List[Tuple]
    ):
        """
        Args:
            batch: A list of tuples of (positive_pair, label, neg_head, neg_tail).
        """
        positive_edgelist_ori, positive_labels_ori, neg_heads_ori, neg_tails_ori, edge_sources_ori = zip(*batch)
        
        pair_drugs_positive_edgelist = torch.from_numpy(np.array(positive_edgelist_ori)[np.array(edge_sources_ori) != "ONSIDES_OFFSIDES_MERGED"]).long()
        pair_drugs_positive_labels = torch.from_numpy(np.array(positive_labels_ori)[np.array(edge_sources_ori) != "ONSIDES_OFFSIDES_MERGED"]).long()
        single_drug_positive_edgelist = torch.from_numpy(np.array(positive_edgelist_ori)[np.array(edge_sources_ori) == "ONSIDES_OFFSIDES_MERGED"]).long()
        single_drug_positive_labels = torch.from_numpy(np.array(positive_labels_ori)[np.array(edge_sources_ori) == "ONSIDES_OFFSIDES_MERGED"]).long()
        if not self.use_single_drug:
            assert single_drug_positive_edgelist.shape[0] == 0
            assert single_drug_positive_labels.shape[0] == 0
        
        pair_drugs_neg_heads = torch.from_numpy(np.array(neg_heads_ori)[np.array(edge_sources_ori) != "ONSIDES_OFFSIDES_MERGED"]).long()
        pair_drugs_neg_tails = torch.from_numpy(np.array(neg_tails_ori)[np.array(edge_sources_ori) != "ONSIDES_OFFSIDES_MERGED"]).long()
        if self.use_single_drug:
            single_drug_neg_heads = torch.from_numpy(np.array(neg_heads_ori)[np.array(edge_sources_ori) == "ONSIDES_OFFSIDES_MERGED"]).long()
            single_drug_neg_tails = torch.from_numpy(np.array(neg_tails_ori)[np.array(edge_sources_ori) == "ONSIDES_OFFSIDES_MERGED"]).long()
        
        pair_drugs_positive_heads = pair_drugs_positive_edgelist[:, 0]
        pair_drugs_positive_tails = pair_drugs_positive_edgelist[:, 1]
        
        # first get negatives
        pair_drugs_negative_edgelist = torch.cat([
            torch.stack([pair_drugs_positive_heads, pair_drugs_neg_tails]),
            torch.stack([pair_drugs_neg_heads, pair_drugs_positive_tails]),
        ], dim=1).T
        pair_drugs_negative_labels = pair_drugs_positive_labels.repeat(2)
        if self.use_single_drug:
            single_drug_negative_edgelist = torch.cat([
                torch.stack([single_drug_neg_heads, single_drug_neg_heads]),
                torch.stack([single_drug_neg_tails, single_drug_neg_tails]),
            ], dim=1).T  # NOTE: Duplicate drugs for single drugs
            single_drug_negative_labels = single_drug_positive_labels.repeat(2)

        # NOTE: Concat drug pair and single drug samples
        positive_edgelist = torch.cat([pair_drugs_positive_edgelist, single_drug_positive_edgelist], dim=0)
        positive_labels = torch.cat([pair_drugs_positive_labels, single_drug_positive_labels], dim=0)
        if self.use_single_drug:
            negative_edgelist = torch.cat([pair_drugs_negative_edgelist, single_drug_negative_edgelist], dim=0)
            negative_labels = torch.cat([pair_drugs_negative_labels, single_drug_negative_labels], dim=0)
        else:
            negative_edgelist = pair_drugs_negative_edgelist
            negative_labels = pair_drugs_negative_labels
        
        pair_drug_edge_sources = np.array(edge_sources_ori)[np.array(edge_sources_ori) != "ONSIDES_OFFSIDES_MERGED"].tolist()
        if self.use_single_drug:
            single_drug_edge_sources = np.array(edge_sources_ori)[np.array(edge_sources_ori) == "ONSIDES_OFFSIDES_MERGED"].tolist()
            positive_edge_sources = pair_drug_edge_sources + single_drug_edge_sources
            negative_edge_sources = pair_drug_edge_sources * 2 + single_drug_edge_sources * 2
        else:
            positive_edge_sources = pair_drug_edge_sources
            negative_edge_sources = pair_drug_edge_sources * 2
        
        # NOTE: Make the edges undirected for training
        positive_edgelist = torch.cat([positive_edgelist, positive_edgelist.flip(1)], dim=0)
        negative_edgelist = torch.cat([negative_edgelist, negative_edgelist.flip(1)], dim=0)
        positive_labels = positive_labels.repeat(2)
        negative_labels = negative_labels.repeat(2)
        
        positive_edge_sources = positive_edge_sources * 2
        negative_edge_sources = negative_edge_sources * 2
        edge_sources = positive_edge_sources + negative_edge_sources
        
        positive_heads = positive_edgelist[:, 0]
        positive_tails = positive_edgelist[:, 1]
        negative_heads = negative_edgelist[:, 0]
        negative_tails = negative_edgelist[:, 1]
        
        # then extract unique indices -- NOTE: We separate out head and tail because we might give head and tail different modality compositions during both training and evaluation. See below for procedure that ensures KG sampling is identical for heads and tails.
        unique_head_indices, all_heads_new = torch.unique(torch.cat([positive_heads, negative_heads], dim=0), return_inverse=True)
        unique_tail_indices, all_tails_new = torch.unique(torch.cat([positive_tails, negative_tails], dim=0), return_inverse=True)
        all_labels_new = torch.cat([positive_labels, negative_labels])
        all_pos_neg_new = torch.cat([torch.ones_like(positive_labels), torch.zeros_like(negative_labels)])
        
        # extract modality availability
        unique_head_mod_avail = self.drug_metadata.loc[unique_head_indices][['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        unique_tail_mod_avail = self.drug_metadata.loc[unique_tail_indices][['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        
        # extract structure modality (torchdrug mol graphs)
        # NOTE: KG modality is computed via HAN during encoding, by doing message passing across the KG and then         
        unique_head_mol_strs = self.all_molecules[unique_head_indices.tolist()]
        unique_tail_mol_strs = self.all_molecules[unique_tail_indices.tolist()]
        
        # NOTE: To ensure KG sampling is identical for the overlapping heads and tails, we first extract the unique indices, then sample the KG. The same subgraph will be fed into the KG encoder for both heads and tails, and the relevant embeddings will be extracted after encoding for heads and tails, respectively.
        unique_drug_indices = torch.unique(torch.cat([unique_head_indices, unique_tail_indices], dim=0))
        kg_dict = sample_kg_data(self.all_kg_data, unique_drug_indices, self.kg_sampling_num_neighbors, 'neighborloader', self.kg_sampling_num_layers, drug_only=True)
        
        # extract Cv, Tx signatures, fill dummies for unavailable ones
        def get_signatures_and_fill_dummy(unique_indices, sig_df, unique_mod_avail, sig, unique_sig_ids):
            unique_sig_output = torch.randn((unique_indices.shape[0], sig_df.shape[0]))
            unique_sig_avail_indices = unique_mod_avail[sig].values == 1
            unique_sig_output[unique_sig_avail_indices, :] = torch.from_numpy(sig_df[unique_sig_ids[unique_sig_avail_indices]].values.T).float()
            unique_sig_output[~unique_sig_avail_indices, :] = 0
            return unique_sig_output

        unique_head_tabular_mods = {}
        unique_tail_tabular_mods = {}
        for mod in NON_TX_MODALITIES[2:]:
            unique_head_mod_sig_ids = self.drug_metadata.loc[unique_head_indices.tolist(), f"{mod}_sig_id"].values
            unique_tail_mod_sig_ids = self.drug_metadata.loc[unique_tail_indices.tolist(), f"{mod}_sig_id"].values
            unique_head_tabular_mods[mod] = get_signatures_and_fill_dummy(unique_head_indices, self.tabular_mod_dfs[mod], unique_head_mod_avail, f"view_{mod}", unique_head_mod_sig_ids)
            unique_tail_tabular_mods[mod] = get_signatures_and_fill_dummy(unique_tail_indices, self.tabular_mod_dfs[mod], unique_tail_mod_avail, f"view_{mod}", unique_tail_mod_sig_ids)
        
        unique_head_tx_dict = {}
        unique_tail_tx_dict = {}
        for cell_line in CELL_LINES:
            unique_head_tx_dict[cell_line] = {}
            unique_tail_tx_dict[cell_line] = {}
            
            # sigs
            unique_head_tx_cell_line_sig_ids = self.drug_metadata.loc[unique_head_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_tail_tx_cell_line_sig_ids = self.drug_metadata.loc[unique_tail_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_head_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(unique_head_indices, self.tx_df, unique_head_mod_avail, f'view_tx_{cell_line}', unique_head_tx_cell_line_sig_ids)
            unique_tail_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(unique_tail_indices, self.tx_df, unique_tail_mod_avail, f'view_tx_{cell_line}', unique_tail_tx_cell_line_sig_ids)
            
            # drugs
            unique_head_tx_dict[cell_line]['drugs'] = unique_head_indices.long()
            unique_tail_tx_dict[cell_line]['drugs'] = unique_tail_indices.long()

            # dosages
            unique_head_tx_dosages = self.drug_metadata.loc[unique_head_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_tail_tx_dosages = self.drug_metadata.loc[unique_tail_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_head_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_head_tx_dosages).float()
            unique_tail_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_tail_tx_dosages).float()
            
            # cell_lines (in str, converted to one-hot )
            unique_head_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_head_indices))
            unique_tail_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_tail_indices))
        
        # NOTE: Instead of using a label matrix, we put all label-specific bilinear decoder weights together into a 3D matrix, and compute for each label all drugs-drugs predictions together (N_drug x feat, feat x feat x num_labels, N_drug x feat), masking the unwanted ones thereafter. This is memory efficient and can be done in a single forward pass.
        label_matrix = None
        
        # finally, construct the input drug's modality masks: reverse to mod avail, here 1 means NOT having the modality, 0 means having the modality (for the sake of key_padding_mask, thus inverse)
        unique_head_masks = torch.from_numpy(1 - unique_head_mod_avail.values).bool()
        unique_tail_masks = torch.from_numpy(1 - unique_tail_mod_avail.values).bool()
        
        return {
            # unique heads
            'head':{
                'drugs': unique_head_indices,
                'strs': unique_head_mol_strs,
                **unique_head_tabular_mods,
                'tx': unique_head_tx_dict,
                'masks': unique_head_masks,
            },
            # unique tails
            'tail':{
                'drugs': unique_tail_indices,
                'strs': unique_tail_mol_strs,
                **unique_tail_tabular_mods,
                'tx': unique_tail_tx_dict,
                'masks': unique_tail_masks,
            },
            # kg
            'kg': kg_dict,
            # used for indexing the label matrix
            'edge_indices':{
                'head': all_heads_new,
                'tail': all_tails_new,
                'label': all_labels_new,
                'pos_neg': all_pos_neg_new,
                'edge_source': edge_sources,
            },
        }
        

class SingleDrugDataCollator:
    def __init__(
        self, 
        num_labels: int,
        drug_metadata: pd.DataFrame,
        all_molecules: PackedMolecule,
        all_kg_data: HeteroData,
        kg_sampling_num_neighbors: int,
        kg_sampling_num_layers: int,
        tabular_mod_dfs: Dict[str, pd.DataFrame],
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
        self.num_labels = num_labels
        self.all_molecules = all_molecules
        self.all_kg_data = all_kg_data
        self.kg_sampling_num_neighbors = kg_sampling_num_neighbors
        self.kg_sampling_num_layers = kg_sampling_num_layers
        self.str_node_feat_dim = self.all_molecules.node_feature.shape[1]
        
        # load perturbation data
        self.tabular_mod_dfs = tabular_mod_dfs
        self.tx_df = tx_df
        
        # prepare modality masks
        self.drug_metadata = drug_metadata
    
    def __call__(
        self, 
        batch: List[Tuple]
    ):
        """
        Args:
            batch: A list of tuples of (positive_pair, label, neg_head, neg_tail).
        """
        single_drug_positive_edgelist, single_drug_positive_labels, single_drug_neg_heads, single_drug_neg_tails = zip(*batch)
        single_drug_positive_edgelist = torch.tensor(single_drug_positive_edgelist, dtype=torch.long)
        single_drug_positive_labels = torch.tensor(single_drug_positive_labels, dtype=torch.long)
        
        single_drug_negative_edgelist = torch.cat([
            torch.stack([torch.tensor(single_drug_neg_heads, dtype=torch.long), torch.tensor(single_drug_neg_heads, dtype=torch.long)]),
            torch.stack([torch.tensor(single_drug_neg_tails, dtype=torch.long), torch.tensor(single_drug_neg_tails, dtype=torch.long)]),
        ], dim=1).T  # NOTE: Duplicate drugs for single drugs
        single_drug_negative_labels = single_drug_positive_labels.repeat(2)
        
        positive_edgelist = single_drug_positive_edgelist
        positive_labels = single_drug_positive_labels
        negative_edgelist = single_drug_negative_edgelist
        negative_labels = single_drug_negative_labels
        
        positive_heads = positive_edgelist[:, 0]
        positive_tails = positive_edgelist[:, 1]
        negative_heads = negative_edgelist[:, 0]
        negative_tails = negative_edgelist[:, 1]
        
        # then extract unique indices -- NOTE: We separate out head and tail because we might give head and tail different modality compositions during both training and evaluation. See below for procedure that ensures KG sampling is identical for heads and tails.
        unique_head_indices, all_heads_new = torch.unique(torch.cat([positive_heads, negative_heads], dim=0), return_inverse=True)
        unique_tail_indices, all_tails_new = torch.unique(torch.cat([positive_tails, negative_tails], dim=0), return_inverse=True)
        all_labels_new = torch.cat([positive_labels, negative_labels])
        all_pos_neg_new = torch.cat([torch.ones_like(positive_labels), torch.zeros_like(negative_labels)])
        
        # extract modality availability
        unique_head_mod_avail = self.drug_metadata.loc[unique_head_indices][['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        unique_tail_mod_avail = self.drug_metadata.loc[unique_tail_indices][['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        
        # extract structure modality (torchdrug mol graphs)
        # NOTE: KG modality is computed via HAN during encoding, by doing message passing across the KG and then         
        unique_head_mol_strs = self.all_molecules[unique_head_indices.tolist()]
        unique_tail_mol_strs = self.all_molecules[unique_tail_indices.tolist()]
        
        # NOTE: To ensure KG sampling is identical for the overlapping heads and tails, we first extract the unique indices, then sample the KG. The same subgraph will be fed into the KG encoder for both heads and tails, and the relevant embeddings will be extracted after encoding for heads and tails, respectively.
        unique_drug_indices = torch.unique(torch.cat([unique_head_indices, unique_tail_indices], dim=0))
        kg_dict = sample_kg_data(self.all_kg_data, unique_drug_indices, self.kg_sampling_num_neighbors, 'neighborloader', self.kg_sampling_num_layers, drug_only=True)
        
        # extract Cv, Tx signatures, fill dummies for unavailable ones
        def get_signatures_and_fill_dummy(unique_indices, sig_df, unique_mod_avail, sig, unique_sig_ids):
            unique_sig_output = torch.randn((unique_indices.shape[0], sig_df.shape[0]))
            unique_sig_avail_indices = unique_mod_avail[sig].values == 1
            unique_sig_output[unique_sig_avail_indices, :] = torch.from_numpy(sig_df[unique_sig_ids[unique_sig_avail_indices]].values.T).float()
            unique_sig_output[~unique_sig_avail_indices, :] = 0
            return unique_sig_output

        unique_head_tabular_mods = {}
        unique_tail_tabular_mods = {}
        for mod in NON_TX_MODALITIES[2:]:
            unique_head_mod_sig_ids = self.drug_metadata.loc[unique_head_indices.tolist(), f"{mod}_sig_id"].values
            unique_tail_mod_sig_ids = self.drug_metadata.loc[unique_tail_indices.tolist(), f"{mod}_sig_id"].values
            unique_head_tabular_mods[mod] = get_signatures_and_fill_dummy(unique_head_indices, self.tabular_mod_dfs[mod], unique_head_mod_avail, f"view_{mod}", unique_head_mod_sig_ids)
            unique_tail_tabular_mods[mod] = get_signatures_and_fill_dummy(unique_tail_indices, self.tabular_mod_dfs[mod], unique_tail_mod_avail, f"view_{mod}", unique_tail_mod_sig_ids)
        
        unique_head_tx_dict = {}
        unique_tail_tx_dict = {}
        for cell_line in CELL_LINES:
            unique_head_tx_dict[cell_line] = {}
            unique_tail_tx_dict[cell_line] = {}
            
            # sigs
            unique_head_tx_cell_line_sig_ids = self.drug_metadata.loc[unique_head_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_tail_tx_cell_line_sig_ids = self.drug_metadata.loc[unique_tail_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_head_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(unique_head_indices, self.tx_df, unique_head_mod_avail, f'view_tx_{cell_line}', unique_head_tx_cell_line_sig_ids)
            unique_tail_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(unique_tail_indices, self.tx_df, unique_tail_mod_avail, f'view_tx_{cell_line}', unique_tail_tx_cell_line_sig_ids)
            
            # drugs
            unique_head_tx_dict[cell_line]['drugs'] = unique_head_indices.long()
            unique_tail_tx_dict[cell_line]['drugs'] = unique_tail_indices.long()
            
            # dosages
            unique_head_tx_dosages = self.drug_metadata.loc[unique_head_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_tail_tx_dosages = self.drug_metadata.loc[unique_tail_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_head_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_head_tx_dosages).float()
            unique_tail_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_tail_tx_dosages).float()
            
            # cell_lines (in str, converted to one-hot )
            unique_head_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_head_indices))
            unique_tail_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_tail_indices))
        
        # NOTE: Instead of using a label matrix, we put all label-specific bilinear decoder weights together into a 3D matrix, and compute for each label all drugs-drugs predictions together (N_drug x feat, feat x feat x num_labels, N_drug x feat), masking the unwanted ones thereafter. This is memory efficient and can be done in a single forward pass.
        label_matrix = None
        
        # finally, construct the input drug's modality masks: reverse to mod avail, here 1 means NOT having the modality, 0 means having the modality (for the sake of key_padding_mask, thus inverse)
        unique_head_masks = torch.from_numpy(1 - unique_head_mod_avail.values).bool()
        unique_tail_masks = torch.from_numpy(1 - unique_tail_mod_avail.values).bool()
        
        return {
            # unique heads
            'head':{
                'drugs': unique_head_indices,
                'strs': unique_head_mol_strs,
                **unique_head_tabular_mods,
                'tx': unique_head_tx_dict,
                'masks': unique_head_masks,
            },
            # unique tails
            'tail':{
                'drugs': unique_tail_indices,
                'strs': unique_tail_mol_strs,
                **unique_tail_tabular_mods,
                'tx': unique_tail_tx_dict,
                'masks': unique_tail_masks,
            },
            # kg
            'kg': kg_dict,
            # used for indexing the label matrix
            'edge_indices':{
                'head': all_heads_new,
                'tail': all_tails_new,
                'label': all_labels_new,
                'pos_neg': all_pos_neg_new,
            },
        }
    

# Drug CL Dataset & Collator
class DrugCLDataset(Dataset):  # just a wrapper for the data list/np.ndarray
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


class DrugCLCollator:
    def __init__(
        self, 
        path_base: str, 
        mod_avail_df: pd.DataFrame, 
        kg_encoder: str,
        kg_sampling_num_neighbors: int = None,
        kg_sampling_num_layers: int = None,
        test_drugs: np.ndarray = None,
    ):
        self.kg_sampling_num_neighbors = kg_sampling_num_neighbors
        self.kg_sampling_num_layers = kg_sampling_num_layers
      
        drug_metadata = pd.read_pickle(os.path.join(path_base, 'drug_features/drug_metadata_ddi.pkl'))
        
        # Get KG and structure data
        all_kg_data = torch.load(os.path.join(path_base, f"drug_features/kg/KG_data_{kg_encoder}.pt"), map_location="cpu")
        self.all_molecules = torch.load(os.path.join(path_base, "drug_features/str/all_molecules_torchdrug.pt"), map_location="cpu")
        self.str_node_feat_dim = self.all_molecules.node_feature.shape[1]
        
        if test_drugs is not None:
            self.kg_data = remove_edges_attached_to_nodes(all_kg_data, torch.from_numpy(test_drugs).long())
        else:
            self.kg_data = all_kg_data
        
        # load tabular drug data
        assert NON_TX_MODALITIES[0] == "str"
        assert NON_TX_MODALITIES[1] == "kg"
        tabular_mod_dfs = {}
        for mod in NON_TX_MODALITIES[2:]:
            tabular_mod_dfs[mod] = pd.read_csv(path_base + f"drug_features/{mod}/{mod}_cp_data.csv", index_col=0)

        tx_df = pd.read_csv(path_base + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)
        
        # create tensors for each perturbation modality, filling in missing samples with dummy 0 (should be masked by masks in the model)
        
        def get_signatures_and_fill_dummy(unique_indices, sig_df, unique_mod_avail, sig, unique_sig_ids):
            unique_sig_output = torch.randn((unique_indices.shape[0], sig_df.shape[0]))
            unique_sig_avail_indices = unique_mod_avail[sig].values == 1
            unique_sig_output[unique_sig_avail_indices, :] = torch.from_numpy(sig_df[unique_sig_ids[unique_sig_avail_indices]].values.T).float()
            unique_sig_output[~unique_sig_avail_indices, :] = 0
            return unique_sig_output
        
        self.tabular_mod_stores = {}
        for mod in NON_TX_MODALITIES[2:]:
            mod_sig_ids = drug_metadata[f"{mod}_sig_id"].values
            self.tabular_mod_stores[mod] = get_signatures_and_fill_dummy(drug_metadata, tabular_mod_dfs[mod], mod_avail_df, f'view_{mod}', mod_sig_ids)
        
        self.tx_store_dict = {}
        for cell_line in CELL_LINES:
            self.tx_store_dict[cell_line] = {}
            
            # sigs
            tx_cell_line_sig_ids = drug_metadata[f'{cell_line}_max_dose_averaged_sig_id'].values
            self.tx_store_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(drug_metadata, tx_df, mod_avail_df, f'view_tx_{cell_line}', tx_cell_line_sig_ids)
            
            # drugs
            self.tx_store_dict[cell_line]['drugs'] = torch.arange(drug_metadata.shape[0]).long()
            
            # dosages
            tx_dosages = drug_metadata[f'{cell_line}_pert_dose'].fillna(0).values
            self.tx_store_dict[cell_line]['dosages'] = torch.from_numpy(tx_dosages).float()
            
            # cell_lines (in str, converted to one-hot )
            self.tx_store_dict[cell_line]['cell_lines'] = np.array([cell_line] * drug_metadata.shape[0])
    
    def __call__(self, batch):
        if isinstance(batch[0], torch.Tensor):  # i.e. call this func directly
            batch_drug_indices = batch[0]  # NOTE: Because of the way TensorDataset loads data, [tensor([xxx])]
        elif isinstance(batch[0], np.ndarray):  # i.e. call this func directly
            batch_drug_indices = torch.from_numpy(batch[0])
        elif isinstance(batch[0], tuple):  # i.e. call this func via DataLoader and the dataset element is a tuple (torch int, )
            batch_drug_indices = torch.LongTensor(batch).flatten()
        elif isinstance(batch[0], int):  # i.e. call this func via DataLoader and the dataset element is an int
            batch_drug_indices = torch.LongTensor(batch)
        else:
            raise NotImplementedError(f"Unknown batch elem type: {type(batch[0])} for element {batch[0]}")
        
        kg_dict = sample_kg_data(self.kg_data, batch_drug_indices, self.kg_sampling_num_neighbors, 'neighborloader', self.kg_sampling_num_layers, drug_only=True)
        
        out_views = (self.all_molecules[batch_drug_indices.tolist()], kg_dict, )
        for mod in NON_TX_MODALITIES[2:]:
            out_views += (self.tabular_mod_stores[mod][batch_drug_indices.tolist()], )
        out_views += ({cell_line : {k: v[batch_drug_indices.tolist()] for k, v in tx_store.items()} for cell_line, tx_store in self.tx_store_dict.items()}, )
        
        return (
            batch_drug_indices,
            out_views,
        )

