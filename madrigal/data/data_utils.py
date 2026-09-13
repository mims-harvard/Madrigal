from typing import Optional
import numpy as np
import torch
from torch_geometric.utils import is_undirected
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


# negative sampling torch version
def structured_negative_sampling_binary_torch(
    edge_index: torch.Tensor, 
    valid_indices: torch.Tensor,
    ground_truth_edge_index: Optional[torch.LongTensor] = None, 
    contains_neg_self_loops: bool = False,
    probs: Optional[torch.Tensor] = None,
):
    r"""Adapted from Pytorch Geometric's `structured_negative_sampling`. Samples a negative edge :obj:`(i,k)` for every positive edge :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a tuple of the form :obj:`(i,j,k)`. This is used for binary DDI case only.

    Args:
        edge_index (LongTensor): The edge indices.
        valid_indices (LongTensor): The indices of nodes that are valid for negative sampling (for tails).
        ground_truth_edge_index (LongTensor): Ground truth edge indices (used for excluding false negatives), usually contains `edge_index`. 
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        probs (torch.Tensor, optional): The probability distribution of nodes for negative sampling.

    :rtype: (LongTensor, LongTensor, LongTensor)

    """
    assert edge_index.numel() > 0
    assert is_undirected(edge_index)
    assert is_undirected(ground_truth_edge_index) if ground_truth_edge_index is not None else True
    
    # get number of nodes and valid indices for sampling
    num_nodes = int(ground_truth_edge_index.max()) + 1
    valid_indices = valid_indices.numpy() if valid_indices is not None else torch.unique(edge_index, sorted=True).numpy()
    
    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)
    
    # add ground truth edges to pos_idx
    if ground_truth_edge_index is not None:
        valid_ground_truth_edge_index = ground_truth_edge_index[
            :, 
            (torch.isin(ground_truth_edge_index[0], edge_index[0])) | (torch.isin(ground_truth_edge_index[1], valid_indices))
        ]  # record those ground truth edges whose heads are also in edge_index's heads and whose tails are in valid_indices for exclusion during sampling later
        ground_truth_row, ground_truth_col = valid_ground_truth_edge_index.cpu()
        ground_truth_idx = ground_truth_row * num_nodes + ground_truth_col
        pos_idx = torch.cat([pos_idx, ground_truth_idx], dim=0)
    
    rand = torch.from_numpy(np.random.choice(valid_indices, (row.size(0), ), replace=True, p=probs)).long()
    neg_idx = row * num_nodes + rand
    
    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.from_numpy(np.random.choice(valid_indices, (rest.size(0), ), replace=True, p=probs)).long()
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)


# negative sampling np version
def structured_negative_sampling_binary(edge_index, valid_negative_nodes: Optional[np.ndarray] = None, other_ground_truth_edge_index: Optional[np.ndarray] = None, num_nodes: Optional[int] = None, contains_neg_self_loops: bool = False, two_sided: bool = True, probs: Optional[np.ndarray] = None):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index: The edge indices.
        other_ground_truth_edge_index: Other ground truth edge indices (used for excluding false negatives).
        valid_negative_nodes: If not None, use this as the source for sampling negative nodes. If None, use all nodes that exist in edge_index.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        two_sided: If True, permutating both src and dst. Else, the src and dst mean different things (e.g. val and train drugs), and we permutate only tail as is convention (used for negative sampling of evaluation triplets in the drug-centric splits, assuming tail is `train` drugs).

    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    
    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1
    base = num_nodes + 1
        
    unique_nodes = np.unique(edge_index)
    if valid_negative_nodes is not None:
        pass
    elif not two_sided:
        valid_negative_nodes = np.unique(edge_index[1])  # By default, we only permutate the tail (expected to be `train`)
    else:
        valid_negative_nodes = np.unique(edge_index)
    
    # we use base systems to easily avoid false negatives
    head, tail = edge_index
    pos_idx = head*base + tail
    
    if two_sided:  # NOTE: even if it is not two sided we can also add rev into pos_idx
        pos_idx_rev = tail*base + head  # also take in the other direction since DDI is undirected
        pos_idx = np.concatenate([pos_idx, pos_idx_rev], axis=0)
    
    if not contains_neg_self_loops:
        # put all (node, node) triplets in pos_idx: node*(base+1), Var node
        loop_idx = np.arange(num_nodes) * (base + 1)
        assert len(loop_idx) == len(set(loop_idx))
        pos_idx = np.concatenate([pos_idx, loop_idx], axis=0)
    
    if other_ground_truth_edge_index is not None:
        bool_indices = (np.isin(other_ground_truth_edge_index[0], unique_nodes) & np.isin(other_ground_truth_edge_index[1], unique_nodes))
        valid_other_ground_truth_edge_index = other_ground_truth_edge_index[:, bool_indices]
        other_ground_truth_head, other_ground_truth_tail = valid_other_ground_truth_edge_index
        other_ground_truth_idx = other_ground_truth_head * base + other_ground_truth_tail
        if two_sided:
            other_ground_truth_idx_rev = other_ground_truth_tail * base + other_ground_truth_head
            other_ground_truth_idx = np.concatenate([other_ground_truth_idx, other_ground_truth_idx_rev], axis=0)
        pos_idx = np.concatenate([pos_idx, other_ground_truth_idx], axis=0)
    
    # sample random negatives for tail
    rand_tail = np.random.choice(valid_negative_nodes, size=(tail.shape[0], ), replace=True, p=probs)
    neg_idx_tail = head * base + rand_tail
    mask_tail = np.isin(neg_idx_tail, pos_idx)
    rest_tail = mask_tail.nonzero()[0].reshape(-1)
    while rest_tail.size > 0:  # pragma: no cover
        tmp_tail = np.random.choice(valid_negative_nodes, size=(rest_tail.shape[0], ), replace=True, p=probs)
        rand_tail[rest_tail] = tmp_tail
        neg_idx_tail = head[rest_tail] * base + tmp_tail

        mask_tail = np.isin(neg_idx_tail, pos_idx)
        rest_tail = rest_tail[mask_tail]
    
    rand_head = None
    if two_sided:
        # sample random negatives for tail
        rand_head = np.random.choice(valid_negative_nodes, size=(head.shape[0], ), replace=True, p=probs)
        neg_idx_head = rand_head * base + tail
        mask_head = np.isin(neg_idx_head, pos_idx)
        rest_head = mask_head.nonzero()[0].reshape(-1)
        while rest_head.size > 0:  # pragma: no cover
            tmp_head = np.random.choice(valid_negative_nodes, size=(rest_head.shape[0], ), replace=True, p=probs)
            rand_head[rest_head] = tmp_head
            neg_idx_head = tmp_head * base + tail[rest_head]

            mask_head = np.isin(neg_idx_head, pos_idx)
            rest_head = rest_head[mask_head]

    return edge_index[0], edge_index[1], rand_head, rand_tail


# np+multilabel version
def structured_negative_sampling_multilabel(
    edge_index, 
    label, 
    valid_negative_nodes: Optional[np.ndarray] = None, 
    other_ground_truth_edge_index: Optional[np.ndarray] = None, 
    other_ground_truth_label: Optional[np.ndarray] = None, 
    num_nodes: Optional[int] = None, 
    contains_neg_self_loops: bool = False, 
    two_sided: bool = True,
    probs: Optional[np.ndarray] = None,
):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index: The edge indices.
        label: Labels of the edges.
        other_ground_truth_edge_index: Other ground truth edge indices (used for excluding false negatives).
        other_ground_truth_label: Labels of other ground truth edges.
        valid_negative_nodes: If not None, use this as the source for sampling negative nodes. If None, use all nodes that exist in edge_index.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        two_sided: If True, permutating both src and dst. Else, the src and dst mean different things (e.g. val and train drugs), and we permutate only tail as is convention (used for negative sampling of evaluation triplets in the drug-centric splits, assuming tail is `train` drugs).

    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    assert edge_index.ndim == 2
    assert label.ndim == 1
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == label.shape[0]
    assert ((other_ground_truth_label is not None) and (other_ground_truth_edge_index is not None) and other_ground_truth_label.shape[0] == other_ground_truth_edge_index.shape[1]) or ((other_ground_truth_label is None) and (other_ground_truth_edge_index is None))
    
    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1
    num_labels = np.max(label) + 1
    if num_labels < num_nodes:
        base = num_labels + 1
    else:
        base = num_nodes + 1
        
    unique_labels = np.unique(label)
    unique_nodes = np.unique(edge_index)
    if valid_negative_nodes is not None:
        pass
    elif not two_sided:
        valid_negative_nodes = np.unique(edge_index[1])  # By default, we only permutate the tail (expected to be `train`)
    else:
        valid_negative_nodes = np.unique(edge_index)
    
    # we use base systems to easily avoid false negatives
    head, tail = edge_index
    pos_idx = label*base*base + head*base + tail  # put label_index at the front because max label_index would usually be smaller than max src/dst
    if two_sided:  # NOTE: even if it is not two sided we can also add rev into pos_idx
        pos_idx_rev = label*base*base + tail*base + head  # also take in the other direction since DDI is undirected
        pos_idx = np.concatenate([pos_idx, pos_idx_rev], axis=0)
    
    if not contains_neg_self_loops:
        # put all (label, node, node) triplets in pos_idx: label*base*base + node*(base+1), Var label, Var node
        loop_idx = np.tile(np.arange(num_nodes)*(base+1), (num_labels, 1)) + np.tile(np.arange(num_labels)*base*base, (num_nodes, 1)).T
        loop_idx = loop_idx.flatten()
        assert len(loop_idx) == len(set(loop_idx))
        pos_idx = np.concatenate([pos_idx, loop_idx], axis=0)
    
    if other_ground_truth_edge_index is not None:
        bool_indices = (np.isin(other_ground_truth_edge_index[0], unique_nodes) & np.isin(other_ground_truth_edge_index[1], valid_negative_nodes))
        valid_other_ground_truth_edge_index = other_ground_truth_edge_index[:, bool_indices]
        valid_other_ground_truth_label = other_ground_truth_label[bool_indices]
        other_ground_truth_head, other_ground_truth_tail = valid_other_ground_truth_edge_index
        other_ground_truth_idx = valid_other_ground_truth_label*base*base + other_ground_truth_head*base + other_ground_truth_tail
        if two_sided:
            other_ground_truth_idx_rev = valid_other_ground_truth_label*base*base + other_ground_truth_tail*base + other_ground_truth_head
            other_ground_truth_idx = np.concatenate([other_ground_truth_idx, other_ground_truth_idx_rev], axis=0)
        pos_idx = np.concatenate([pos_idx, other_ground_truth_idx], axis=0)
    
    # sample random negatives for tail
    rand_tail = np.random.choice(valid_negative_nodes, size=(tail.shape[0], ), replace=True, p=probs)
    neg_idx_tail = label*base*base + head*base + rand_tail
    mask_tail = np.isin(neg_idx_tail, pos_idx)
    rest_tail = mask_tail.nonzero()[0].reshape(-1)
    while rest_tail.size > 0:  # pragma: no cover
        tmp_tail = np.random.choice(valid_negative_nodes, size=(rest_tail.shape[0], ), replace=True, p=probs)
        rand_tail[rest_tail] = tmp_tail
        neg_idx_tail = label[rest_tail]*base*base + head[rest_tail]*base + tmp_tail

        mask_tail = np.isin(neg_idx_tail, pos_idx)
        rest_tail = rest_tail[mask_tail]
    
    rand_head = None
    if two_sided:
        # sample random negatives for tail
        rand_head = np.random.choice(valid_negative_nodes, size=(head.shape[0], ), replace=True, p=probs)
        neg_idx_head = label*base*base + rand_head*base + tail
        mask_head = np.isin(neg_idx_tail, pos_idx)
        rest_head = mask_head.nonzero()[0].reshape(-1)
        while rest_head.size > 0:  # pragma: no cover
            tmp_head = np.random.choice(valid_negative_nodes, size=(rest_head.shape[0], ), replace=True, p=probs)
            rand_head[rest_head] = tmp_head
            neg_idx_head = label[rest_head]*base*base + tmp_head*base + tail[rest_head]

            mask_head = np.isin(neg_idx_head, pos_idx)
            rest_head = rest_head[mask_head]

    return edge_index[0], edge_index[1], rand_head, rand_tail


def remove_edges_attached_to_nodes(kg_data: HeteroData, test_drugs_to_remove_edges: torch.LongTensor, node_type: str = 'drug'):
    kg_data_filtered = kg_data.clone()
    for edge_type in kg_data_filtered.edge_types:
        if edge_type[0] == 'drug' and edge_type[-1] == 'drug':
            temp = kg_data_filtered[edge_type]['edge_index']
            kg_data_filtered[edge_type]['edge_index'] = temp[:, (~torch.isin(temp[0], test_drugs_to_remove_edges)) & (~torch.isin(temp[1], test_drugs_to_remove_edges))]
        elif edge_type[0] == 'drug':
            temp = kg_data_filtered[edge_type]['edge_index']
            kg_data_filtered[edge_type]['edge_index'] = temp[:, (~torch.isin(temp[0], test_drugs_to_remove_edges))]
        elif edge_type[-1] == 'drug':
            temp = kg_data_filtered[edge_type]['edge_index']
            kg_data_filtered[edge_type]['edge_index'] = temp[:, (~torch.isin(temp[1], test_drugs_to_remove_edges))]
        else:
            pass
    return kg_data_filtered


def sample_kg_data(all_kg_data: HeteroData, batch_drugs: torch.LongTensor, kg_sampling_num_neighbors: int = None, sampler: str = 'neighborloader', num_layers: int = 2, drug_only: bool = True):
    unique_kg_avail_bool = torch.zeros(all_kg_data.x_dict['drug'].shape[0]).bool()  # NOTE: here we use the fact that drugs are sorted so that those with KG are at the front
    unique_kg_avail_bool[batch_drugs[batch_drugs < unique_kg_avail_bool.shape[0]]] = True
    assert (batch_drugs < unique_kg_avail_bool.shape[0]).sum() == unique_kg_avail_bool.sum()
    
    # remove edges attached to nodes, but keep the original nodes (so as not to reindex everything)
    if kg_sampling_num_neighbors is not None and sampler is not None:
        assert num_layers is not None
        if sampler == 'neighborloader':
            if drug_only:
                kg_loader = NeighborLoader(
                    all_kg_data,
                    # Sample `num_neighbors` for each node and each edge type for `num_layers` iterations
                    num_neighbors=[kg_sampling_num_neighbors] * num_layers,
                    # Use full batch for "drug"
                    batch_size=(batch_drugs < unique_kg_avail_bool.shape[0]).sum().item(),
                    input_nodes=('drug', unique_kg_avail_bool),
                )
                kg_batch_data = next(iter(kg_loader))
                kg_drug_index_map = kg_batch_data['drug']['n_id']
            else:
                raise NotImplementedError
                kg_loader = NeighborLoader(
                    all_kg_data,
                    # Sample `num_neighbors` for each node and each edge type for `num_layers` iterations
                    num_neighbors=[kg_sampling_num_neighbors] * num_layers,
                    # Use full batch for "drug"
                    batch_size=(batch_drugs < unique_kg_avail_bool.shape[0]).sum().item(),
                    input_nodes=('drug', unique_kg_avail_bool),
                )
                kg_batch_data = next(iter(kg_loader))
                kg_drug_index_map = kg_batch_data['drug']['n_id']
        else:
            raise NotImplementedError
    else:  # for the cases where we don't want to sample the KG (and the GPU can hold)
        kg_batch_data = all_kg_data.clone()
        kg_drug_index_map = torch.arange(kg_batch_data.x_dict['drug'].shape[0])
    
    return {
        'data': kg_batch_data,
        'drug_index_map': kg_drug_index_map,
    }
