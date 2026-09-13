import os
import shutil
import logging
import random
import math
from dotenv import load_dotenv
import numpy as np
from typing import Iterable, Union
from itertools import chain, combinations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import HeteroData, Data
from torchdrug import models
from torchdrug.data import PackedMolecule

from .chemcpa.chemcpa_config_utils import generate_configs, read_config

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


SEED = 42
MOL_DIM = 67  # NOTE: torchdrug default
MAX_DRUGS = 25000
CELL_LINES = ['a375', 'a549', 'asc', 'ha1e', 'hcc515', 'hec108', 'hela', 'hepg2', 'ht29', 'huvec', 'mcf7', 'npc', 'pc3', 'thp1', 'vcap', 'yapc']  # NOTE: This list is ORDERED
CELL_LINES_CAPITALIZED = [cell_line.upper() for cell_line in CELL_LINES]
NON_TX_MODALITIES = ["str", "kg", "cv"]
NUM_NON_TX_MODALITIES = len(NON_TX_MODALITIES)
NUM_MODALITIES = NUM_NON_TX_MODALITIES + len(CELL_LINES)


load_dotenv()
PROJECT_DIR = os.getenv("PROJECT_DIR", "")
DATA_DIR = os.getenv("DATA_DIR", "")
BASE_DIR = os.getenv("BASE_DIR", "")
ENCODER_CKPT_DIR = os.getenv("ENCODER_CKPT_DIR", "")
CL_CKPT_DIR = os.getenv("CL_CKPT_DIR", "")


######
# Masking
######
def get_pretrain_masks(drugs, masks, pretrain_mode, pretrain_unbalanced, pretrain_tx_downsample_ratio):
    """ Get all train subset masks for pretraining and finetuning
    """
    # NOTE: In all cases, we first need to construct a dictionary of all possible subsets of all possible masks of drugs
    unique_subset_masks = {}  # {tuple(modalities):(subsets, subset_probs)}

    # compute sample-balanced modality sampling probabilities
    if not pretrain_unbalanced:
        mod_probs = (1 / (1 - masks).sum(axis=0))
        assert pretrain_tx_downsample_ratio <= 1
        mod_probs[-len(CELL_LINES):] = pretrain_tx_downsample_ratio * mod_probs[-len(CELL_LINES):]
        mod_probs = np.array(mod_probs / mod_probs.sum())
        mod_probs = np.clip(mod_probs, 1e-6, 1.0)  # avoid zero probs
    
    # get all possible subset masks for all samples
    if pretrain_mode == 'double_random' or pretrain_mode == 'str_kg':
        for mask in np.unique(masks, axis=0):
            # The first subset in the output of powerset is always (), so remove it
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor(list(indices), masks.shape[1]) for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:]])  # in the final mask, 0 means not masked, 1 means masked
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}  # in the final mask, 0 means not masked, 1 means masked
    
    elif pretrain_mode == 'str_center' and pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            mask[0] = 1  # NOTE: we don't want str to appear in the other branch as that would lead to shortcut solution, so set str (0) always to True (masked)
            # The first subset in the output of powerset is always (), so remove it
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor(list(indices), masks.shape[1]) for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:]])  # in the final mask, 0 means not masked, 1 means masked
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}  # in the final mask, 0 means not masked, 1 means masked
    
    elif pretrain_mode == 'str_center' and not pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            # same as above, set str always to masked and remove the first subset in the output of powerset, which is always ()
            mask[0] = 1
            all_subsets = [from_indices_to_tensor(list(indices), masks.shape[1]).numpy() for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:]]
            
            # for each of the subset, get the probability of it getting sampled
            # P({m1, ..., not n1, ...}) = P(m1) * ... * (1-P(n1)) * ... * (#all choose #m)
            subset_probs = []
            for subset in all_subsets:
                subset_probs.append(np.concatenate([mod_probs[np.where(subset==0)[0]], (1-mod_probs)[np.where(subset==1)[0]]]).prod() * math.comb((1-mask).sum(), (1-subset).sum())) # type: ignore
            subset_probs = np.array(subset_probs) / sum(subset_probs)
            
            unique_subset_masks[tuple(mask)] = (all_subsets, subset_probs)
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}

    elif pretrain_mode == 'str_center_uni' and pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):    
            # Different from all above, here we only want unimodal masks.  The first subset in the output of torch.where is always 0 (structure), so remove it.
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor([mod_index], masks.shape[1]) for mod_index in np.where(mask==0)[0][1:]])
            
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}
        
    elif pretrain_mode == 'str_center_uni' and not pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            # Same as above, but we also need to compute the sampling probabilities
            all_subsets = [from_indices_to_tensor([mod_index], masks.shape[1]).numpy() for mod_index in np.where(mask==0)[0][1:]]
            
            subset_probs = []
            for subset in all_subsets:
                subset_probs.extend(mod_probs[np.where(subset==0)[0]])  # type: ignore # because each time we only sample one modality
            subset_probs = np.array(subset_probs) / sum(subset_probs)
            
            unique_subset_masks[tuple(mask)] = (all_subsets, subset_probs)
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}
    
    elif pretrain_mode == 'str_center_comb' and pretrain_unbalanced:  # in this case, the other branch always contains more than one modalities
        for mask in np.unique(masks, axis=0):
            mask[0] = 1  # NOTE: we don't want str to appear in the other branch as that would lead to shortcut solution, so set str (0) always to True (masked)
            # The first subset in the output of powerset is always (), so remove it
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor(list(indices), masks.shape[1]) for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:] if len(indices) > 1])  # in the final mask, 0 means not masked, 1 means masked
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}  # in the final mask, 0 means not masked, 1 means masked

    elif pretrain_mode == 'str_center_comb' and not pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            # same as above, set str always to masked and remove the first subset in the output of powerset, which is always ()
            mask[0] = 1
            all_subsets = [from_indices_to_tensor(list(indices), masks.shape[1]).numpy() for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:] if len(indices) > 1]
            
            # for each of the subset, get the probability of it getting sampled
            subset_probs = []
            for subset in all_subsets:
                subset_probs.append(np.concatenate([mod_probs[np.where(subset==0)[0]], (1-mod_probs)[np.where(subset==1)[0]]]).prod()) # type: ignore
            subset_probs = np.array(subset_probs) / sum(subset_probs)
            
            unique_subset_masks[tuple(mask)] = (all_subsets, subset_probs)
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}
    
    else:
        raise NotImplementedError
    
    return all_subset_masks


def get_train_masks(finetune_mode, train_batch_drugs, masks):
    pass


####
# Model utils
####
def get_model(
    all_kg_data, 
    feature_dim,
    prediction_dim,
    str_encoder_name, 
    str_encoder_hparams, 
    kg_encoder_name, 
    kg_encoder_hparams, 
    cv_encoder_name,
    cv_encoder_hparams,
    tx_encoder_name,
    tx_encoder_hparams,
    num_attention_bottlenecks,
    pos_emb_type,
    pos_emb_dropout,
    transformer_fusion_hparams,
    proj_hparams, 
    fusion,
    normalize,
    decoder_normalize,
    checkpoint_path,
    frozen,
    device,
    modality_pretrain_path=None,
    encoder_only=False,
    finetune_mode=None,
    str_node_feat_dim=MOL_DIM,
    logger=None,
    use_modality_pretrain=True,
    use_tx_basal=False,
    adapt_before_fusion=False,
    use_pretrained_adaptor=False,
    use_single_drug=False,
    prediction_dim_single_drug=None,
    tab_mod_encoder_hparams_dict=None,
):
    from .models.models import MadrigalEncoder, MadrigalMultilabel
    
    # encoder
    if checkpoint_path is None:
        encoder = MadrigalEncoder(
            all_kg_data=all_kg_data, 
            feat_dim=feature_dim, 
            str_encoder_name=str_encoder_name, 
            str_encoder_hparams=str_encoder_hparams, 
            kg_encoder_name=kg_encoder_name,
            kg_encoder_hparams=kg_encoder_hparams, 
            cv_encoder_name=cv_encoder_name, 
            cv_encoder_hparams=cv_encoder_hparams, 
            tx_encoder_name=tx_encoder_name, 
            tx_encoder_hparams=tx_encoder_hparams, 
            num_tx_bottlenecks=num_attention_bottlenecks,
            pos_emb_dropout=pos_emb_dropout, 
            transformer_fusion_hparams=transformer_fusion_hparams, 
            proj_hparams=proj_hparams,
            fusion=fusion,
            str_node_feat_dim=str_node_feat_dim,
            use_modality_pretrain=use_modality_pretrain,
            normalize=normalize,
            pos_emb_type=pos_emb_type,
            use_tx_basal=use_tx_basal,
            adapt_before_fusion=adapt_before_fusion,
            tab_mod_encoder_hparams_dict=tab_mod_encoder_hparams_dict,
        )
        
        # encoder configs
        encoder_configs = {
            'all_kg_data': all_kg_data, 
            'feat_dim': feature_dim, 
            'str_encoder_name': str_encoder_name, 
            'str_encoder_hparams': str_encoder_hparams, 
            'kg_encoder_name': kg_encoder_name,
            'kg_encoder_hparams': kg_encoder_hparams, 
            'cv_encoder_name': cv_encoder_name, 
            'cv_encoder_hparams': cv_encoder_hparams, 
            'tx_encoder_name': tx_encoder_name, 
            'tx_encoder_hparams': tx_encoder_hparams,
            'num_tx_bottlenecks': num_attention_bottlenecks,
            'pos_emb_type': pos_emb_type,
            'pos_emb_dropout': pos_emb_dropout, 
            'transformer_fusion_hparams': transformer_fusion_hparams, 
            'proj_hparams': proj_hparams,
            'fusion': fusion,
            'str_node_feat_dim': str_node_feat_dim,
            'use_modality_pretrain': use_modality_pretrain,
            'normalize': normalize,
            'adapt_before_fusion': adapt_before_fusion,
            "tab_mod_encoder_hparams_dict": tab_mod_encoder_hparams_dict,
        }
        
    else:
        full_ckpt_path = CL_CKPT_DIR+checkpoint_path
        if os.path.isfile(full_ckpt_path):
            if logger is not None:
                logger.info("Loading checkpoint at '{}'".format(full_ckpt_path))
            else:
                print("Loading checkpoint at '{}'".format(full_ckpt_path))
            
            checkpoint = torch.load(full_ckpt_path, map_location="cpu")
            encoder_configs = checkpoint['encoder_configs']
            state_dict = checkpoint['state_dict']
            
            print(full_ckpt_path)
            
            # replace the hyperparameters other than encoder-related ones in encoder_configs with the desired ones
            for k, new_v in zip(
                ['num_tx_bottlenecks', 'pos_emb_type', 'pos_emb_dropout', 'transformer_fusion_hparams', 'proj_hparams', 'fusion', 'normalize', 'use_tx_basal', 'adapt_before_fusion'],
                [num_attention_bottlenecks, pos_emb_type, pos_emb_dropout, transformer_fusion_hparams, proj_hparams, fusion, normalize, use_tx_basal, adapt_before_fusion],
            ):
                if new_v is not None:
                    encoder_configs[k] = new_v
            
            # NOTE: Since we still want to do ablation studies with pretrained models, we need to revise the encoder_configs according to specified `finetune_mode`
            encoder = MadrigalEncoder(**encoder_configs)
            
            if 'epoch' in checkpoint.keys():
                pass
            
            # process state_dict
            for k in list(state_dict.keys()):
                # NOTE: retain only base_encoder up to before the embedding layer, excluding fusion-related modules
                if (
                    k.startswith('base_encoder') and
                    not k.startswith('base_encoder.head') and 
                    k not in {'base_encoder.tx_bottleneck_tokens', 'base_encoder.cls'} and
                    not k.startswith('base_encoder.pos_encoder') and
                    not k.startswith('base_encoder.transformer')
                ):
                    if (not use_pretrained_adaptor) and k.startswith('base_encoder.uni_projector'):
                        continue
                    # remove prefix
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
                
            msg = encoder.load_state_dict(state_dict, strict=False)
            if logger is not None:
                logger.info("=> Missing keys {}".format(msg.missing_keys))
                logger.info("=> Unexpected keys {}".format(msg.unexpected_keys))
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(full_ckpt_path, checkpoint['epoch']))
            else:
                print("=> Missing keys {}".format(msg.missing_keys))
                print("=> Unexpected keys {}".format(msg.unexpected_keys))
                print("=> loaded checkpoint '{}' (epoch {})".format(full_ckpt_path, checkpoint['epoch']))
                
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(full_ckpt_path))
    
    if encoder_only:
        if device is not None:
            encoder.to(device)
        return encoder, encoder_configs
    
    # full model
    model = MadrigalMultilabel(
        encoder, 
        feat_dim=feature_dim, 
        prediction_dim=prediction_dim, 
        prediction_dim_single_drug=prediction_dim_single_drug,
        normalize=decoder_normalize,
        use_single_drug=use_single_drug,
    )
    if device is not None:
        model = model.to(device)
    
    if frozen:
        for param in encoder.parameters():
            param.requires_grad = False
    
    # model configs (in addition to encoder configs)
    model_configs = {
        'feat_dim': feature_dim, 
        'prediction_dim': prediction_dim, 
        'normalize': normalize,
        "use_single_drug": use_single_drug,
    }
    
    return model, encoder_configs, model_configs
    

######
# Hook utils
######
def get_activation(name, activation):
    def hook(module, input, output):
        if name in activation.keys():
            activation[name].append(output)
        else:
            activation[name] = [output]
    return hook


######
# Pretraining modality sampler
######
# First derive a "modality subset bank", then in each epoch sample from the bank
def pretrain_modality_subset_sampler(all_subset_masks: np.ndarray, pretrain_mode: str = 'str_center_uni', unbalanced: bool = False):
    if pretrain_mode in {'str_center', 'str_center_uni', 'str_center_comb'}:
        if not unbalanced:  # In this case the all_subset_masks is a list of tuples of (subset_masks, subset_probs)
            aug1 = torch.ones_like(torch.from_numpy(all_subset_masks[0][0][0]))  # first mask list & prob tuple -> first mask list -> first mask
            aug1[0] = 0
            aug1 = aug1.repeat(len(all_subset_masks), 1).bool()
            aug2 = torch.stack([torch.from_numpy(subset_masks[np.random.choice(np.arange(len(subset_masks)), size=1, p=subset_probs)[0]]) for subset_masks, subset_probs in all_subset_masks], dim=0).bool()
            
        else:
            aug1 = torch.ones_like(all_subset_masks[0][0])  # first mask list -> first mask (subset)
            aug1[0] = 0
            aug1 = aug1.repeat(len(all_subset_masks), 1).bool()
            aug2 = torch.stack([subset_masks[torch.randint(len(subset_masks), (1,))[0].item()] for subset_masks in all_subset_masks], dim=0).bool()

    elif pretrain_mode == 'double_random':
        # Scale up for all possible modality subsets for arbitrary number of views
        sampled_masks = torch.stack([subset_masks[[torch.randperm(len(subset_masks))[:2]]] for subset_masks in all_subset_masks], dim=0)
        aug1 = sampled_masks[:, 0, :].bool()
        aug2 = sampled_masks[:, 1, :].bool()
    
    elif pretrain_mode == 'str_kg':
        aug1, aug2 = torch.ones_like(all_subset_masks[0][0]), torch.ones_like(all_subset_masks[0][0])
        aug1[0] = 0
        aug2[1] = 0
        aug1 = aug1.repeat(len(all_subset_masks), 1).bool()
        aug2 = aug2.repeat(len(all_subset_masks), 1).bool()
        
    else:
        raise NotImplementedError

    return aug1, aug2


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def from_indices_to_tensor(indices: Iterable, size: tuple, value: Union[int, float] = 0, base_value: Union[int, float] = 1, dim: int = -1):
    """ Transform a tensor of indices (where the dimensions except for the `dim` dimension are the same as expected output `size`) to a tensor of size `size` with value at the `indices` (over dim `dim`) and `base_value` elsewhere
    E.g., from_indices_to_tensor([[0, 1], [1, 2]], (2, 3), value=1, base_value=0, dim=1) -> [[1, 1, 0], [0, 1, 1]]
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)
    if len(indices.shape)>1:
        assert len(indices.shape) == len(size)
        assert indices.shape[0] == size[0]
    out = torch.ones(size) * base_value
    return out.scatter(dim=dim, index=indices, value=value)


def get_modality_subset_index_lists(avail_indices, group_tx: bool = False, require_str: bool = True):
    """Enumerate modality subsets (as lists of available modality indices) for random-subset
    finetuning (the `*_random_sample` modes). Generalizes `powerset(available)`.

    When `group_tx` is True, all TX cell-line modalities (indices NUM_NON_TX_MODALITIES ..
    NUM_MODALITIES-1) are treated as a SINGLE meta-modality: a sampled subset contains either
    all of a drug's available TX cell lines or none of them.

    `require_str` keeps only subsets containing the structure modality (index 0). The
    structure-only subset [0] (when present) is returned first, preserving the `+1` offset
    convention used elsewhere to draw a non-structure-only mask.
    """
    avail = {int(i) for i in avail_indices}
    if group_tx:
        groups = [[i] for i in range(NUM_NON_TX_MODALITIES)] + [list(range(NUM_NON_TX_MODALITIES, NUM_MODALITIES))]
    else:
        groups = [[i] for i in range(NUM_MODALITIES)]
    avail_groups = [grp for grp in ([i for i in g if i in avail] for g in groups) if len(grp) > 0]
    subsets = []
    for combo in powerset(range(len(avail_groups))):
        if len(combo) == 0:
            continue
        idxs = sorted(i for gi in combo for i in avail_groups[gi])
        if require_str and 0 not in idxs:
            continue
        subsets.append(idxs)
    return subsets


#######
# Saving utils
#######
def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')


######
# Optimization utils
######
def to_device(input, device):
    if isinstance(input, dict):
        for k, v in input.items():
            input[k] = to_device(v, device)
        return input
    elif isinstance(input, (list, tuple)):
        return [to_device(inp, device) for inp in input]
    elif isinstance(input, (torch.Tensor, PackedMolecule, HeteroData, Data)):  # NOTE: don't put some random stuff in the batch
        return input.to(device)
    else:  # np.ndarray, pd.DataFrame, int, float, str, bool, etc.
        return input
    
    
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
    
    
def get_parameter_names(model, forbidden_layer_types):
    """
    Adapted from transformers.trainer_pt_utils.get_parameter_names (change the logic of nn.Parameter)
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(name for name in model._parameters.keys() if 'cls' not in name and 'bottleneck_tokens' not in name)
    return result


def create_optimizer(model, hparams):
    """
    Enable independent learning rates for different parts of the model.
    """
    from .models.models import HAN, HGT, RGCN, MLPEncoder, TransformerFusion, PositionEncodingSinusoidal, PositionEncodingLearnable, MLPAdaptor, BilinearDDIScorer, MadrigalEncoder
    from .chemcpa.chemCPA.model import TxAdaptingComPert
    
    decay_params = get_parameter_names(model, [nn.LayerNorm])
    decay_params = [name for name in decay_params if "bias" not in name]
    
    str_encoder_params = get_parameter_names(model, [HAN, HGT, RGCN, MLPEncoder, TxAdaptingComPert, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, TransformerFusion, BilinearDDIScorer])  # Either GIN or GAT
    kg_encoder_params = get_parameter_names(model, [models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, MLPEncoder, TxAdaptingComPert, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, TransformerFusion, BilinearDDIScorer])  # Either HGT, HAN, or RGCN
    tabular_mod_encoders_params = get_parameter_names(model, [HAN, HGT, RGCN, models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, TxAdaptingComPert, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, TransformerFusion, BilinearDDIScorer])
    tx_encoders_params = get_parameter_names(model, [HAN, HGT, RGCN, models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, MLPEncoder, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, TransformerFusion, BilinearDDIScorer])
    fusion_params = get_parameter_names(model, [HAN, HGT, RGCN, models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, MLPEncoder, TxAdaptingComPert, BilinearDDIScorer])  # MLP, TransformerFusion, PositionEncoding
    fusion_params += list(model._parameters.keys())  # NOTE: Add [CLS] and [TX_BOTTLENECK] into fusion_params
    decoder_params = get_parameter_names(model, [MadrigalEncoder])  # just the decoder
    
    str_encoder_no_decay_params = list(
        set(str_encoder_params) - set(decay_params)
    )
    str_encoder_decay_params = list(
        set(str_encoder_params) & set(decay_params)
    )

    kg_encoder_no_decay_params = list(set(kg_encoder_params) - set(decay_params))
    kg_encoder_decay_params = list(set(kg_encoder_params) & set(decay_params))
    
    tabular_mod_encoders_no_decay_params = list(set(tabular_mod_encoders_params) - set(decay_params))
    tabular_mod_encoders_decay_params = list(set(tabular_mod_encoders_params) & set(decay_params))

    tx_encoders_no_decay_params = list(set(tx_encoders_params) - set(decay_params))
    tx_encoders_decay_params = list(set(tx_encoders_params) & set(decay_params))
    
    fusion_no_decay_params = list(set(fusion_params) - set(decay_params))
    fusion_decay_params = list(set(fusion_params) & set(decay_params))

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in str_encoder_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['structure_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in str_encoder_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['structure_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in kg_encoder_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['kg_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in kg_encoder_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['kg_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in tabular_mod_encoders_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in tabular_mod_encoders_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in tx_encoders_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in tx_encoders_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in fusion_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['fusion_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in fusion_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['fusion_lr'],
        },
        {
            "params": [
                p 
                for n, p in model.named_parameters() 
                if n in decoder_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['decoder_lr'],
        },
    ]

    OPTIMIZER_CLASSES = {"radam": torch.optim.RAdam, "adamw": torch.optim.AdamW}

    optimizer_class = OPTIMIZER_CLASSES[hparams['optimizer']]
    optimizer_kwargs = {
        "betas": (hparams['beta1'], hparams['beta2']),
        "eps": hparams['eps'],
    }

    return optimizer_class(
        optimizer_grouped_parameters, **optimizer_kwargs
    )


def get_loss_fn(loss_fn_name, task, loss_readout):
    if loss_fn_name == 'bce':
        # NOTE: note that multiclass task can also use BCE loss with negative sampling
        loss_fn = nn.BCELoss(reduction=loss_readout)
    elif loss_fn_name == 'ce' and task == 'multiclass':
        loss_fn = nn.CrossEntropyLoss(reduction=loss_readout)
    else:
        raise NotImplementedError("Loss function {} not implemented for task {}".format(loss_fn_name, task))

    return loss_fn


class LARS(torch.optim.Optimizer):
    """
    Copied from https://github.com/facebookresearch/moco-v3/blob/main/moco/optimizer.py
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self): # type: ignore
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


class LinearWarmupCosineDecaySchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, num_cycles=1., last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_cycles = num_cycles
        super(LinearWarmupCosineDecaySchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self): # type: ignore
        cur_epoch = self.last_epoch
        if cur_epoch < self.warmup_epochs:
            return [base_lr * cur_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            t = (cur_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [base_lr * (1 + math.cos(math.pi * self.num_cycles * t)) / 2 \
                for base_lr in self.base_lrs]


def adjust_learning_rate(optimizer, cur_epoch, lr, warmup_epochs, num_epochs):
    """
    During warmup, increase lr linearly. Then decays the learning rate with half-cycle cosine after warmup.
    """
    if cur_epoch < warmup_epochs:
        lr = lr * cur_epoch / warmup_epochs
    else:
        lr = lr * 0.5 * (1. + math.cos(math.pi * (cur_epoch - warmup_epochs) / (num_epochs - warmup_epochs)))  # cosine decay schedule
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


######
# Logging utils
######
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_root_logger(fname: str = "out", file: bool = True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("[%(asctime)-10s] %(message)s", '%m/%d/%Y %H:%M:%S')

    if file:
        handler = logging.FileHandler(fname, mode='w')
        handler.setFormatter(format)
        logger.addHandler(handler)

    logger.addHandler(logging.StreamHandler())

    return logger


######
# Checkpoint saving
######
def _atomic_torch_save(obj, path, retries=5, backoff=3.0):
    """torch.save written to a temp file in the same dir then atomically os.replace'd, with retries."""
    import tempfile, time as _time
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    last = None
    for attempt in range(retries):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            os.close(fd)
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except Exception as e:
            last = e
            if tmp is not None and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt < retries - 1:
                _time.sleep(backoff * (attempt + 1))
    raise last


def safe_save(obj, path, retries=5, backoff=3.0, staging_root=None):
    """Atomic, retried torch.save with an optional staging fallback.

    Tries to save `obj` to `path` atomically (temp + os.replace) with retries. If the target
    keeps failing and a staging root is configured (arg `staging_root` or env `SAFE_SAVE_STAGING`),
    the object is instead written to a mirror of the path under the staging root and recorded in
    `<staging_root>/SAFE_SAVE_MANIFEST.tsv`; a later `drain_staging()` moves staged files to their
    real destinations. Returns the path actually written.
    """
    staging_root = staging_root if staging_root is not None else os.environ.get("SAFE_SAVE_STAGING", "")
    try:
        _atomic_torch_save(obj, path, retries=retries, backoff=backoff)
        return path
    except Exception as primary_err:
        if not staging_root:
            raise
        stage_path = staging_root.rstrip("/") + os.path.abspath(path)  # mirror the full tree
        _atomic_torch_save(obj, stage_path, retries=retries, backoff=backoff)
        manifest = os.path.join(staging_root, "SAFE_SAVE_MANIFEST.tsv")
        os.makedirs(staging_root, exist_ok=True)
        with open(manifest, "a") as f:
            f.write(f"{stage_path}\t{os.path.abspath(path)}\n")
        print(f"[safe_save] primary write to {path} failed ({type(primary_err).__name__}: {primary_err}); "
              f"staged -> {stage_path} (recorded in {manifest})")
        return stage_path


def drain_staging(staging_root=None, retries=5, backoff=3.0, logger=None):
    """Move any previously-staged files (from safe_save fallback) to their intended destinations.

    Safe to call at the start of every job. Cross-filesystem aware (copy to a temp beside the dest,
    atomic-rename, then delete the source). Leaves un-movable entries in the manifest for next time.
    """
    import shutil as _shutil, time as _time
    staging_root = staging_root if staging_root is not None else os.environ.get("SAFE_SAVE_STAGING", "")
    if not staging_root:
        return
    manifest = os.path.join(staging_root, "SAFE_SAVE_MANIFEST.tsv")
    if not os.path.exists(manifest):
        return
    _log = (logger.info if logger is not None else print)
    with open(manifest) as f:
        entries = [ln.rstrip("\n") for ln in f if ln.strip()]
    remaining = []
    moved = 0
    for line in entries:
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        src, dst = parts
        if not os.path.exists(src):
            continue  # already moved or gone
        ok = False
        for attempt in range(retries):
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".movetmp"
                _shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                os.remove(src)
                ok = True
                break
            except Exception:
                if attempt < retries - 1:
                    _time.sleep(backoff * (attempt + 1))
        if ok:
            moved += 1
        else:
            remaining.append(line)
    with open(manifest, "w") as f:
        for line in remaining:
            f.write(line + "\n")
    if moved or remaining:
        _log(f"[drain_staging] moved {moved} staged file(s); {len(remaining)} still pending")


######
# Load model hparams
######
def get_str_encoder_hparams(args, hparams):
    if args.str_encoder == 'gat':
        structural_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['gat_hidden_dims', 'gat_edge_input_dim', 'gat_att_heads', 'gat_negative_slope', 'gat_batch_norm', 'gat_actn', 'gat_readout']}

    elif args.str_encoder == 'gin':
        structural_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['gin_hidden_dims', 'gin_edge_input_dim', 'gin_num_mlp_layer', 'gin_eps', 'gin_batch_norm', 'gin_actn', 'gin_readout']}

    else:
        raise NotImplementedError
    
    return structural_encoder_hparams


def get_kg_encoder_hparams(args, hparams):
    if 'han' in args.kg_encoder:
        kg_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['han_att_heads', 'han_hidden_dim', 'han_num_layers', 'han_negative_slope', 'han_dropout']}
    elif 'hgt' in args.kg_encoder:
        kg_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['hgt_hidden_dim', 'hgt_num_layers', 'hgt_att_heads', 'hgt_group']}
    else:
        raise NotImplementedError
    return kg_encoder_hparams


def get_cv_encoder_hparams(args, hparams, cv_input_dim):
    if args.cv_encoder == 'mlp':
        cv_encoder_hparams = {'cv_input_dim': cv_input_dim}
        cv_encoder_hparams.update({hparam_name:hparams[hparam_name] for hparam_name in ['cv_mlp_hidden_dims', 'cv_mlp_dropout', 'cv_mlp_norm', 'cv_mlp_actn', 'cv_mlp_order']})
    else:
        raise NotImplementedError
    return cv_encoder_hparams


def get_tx_encoder_hparams(args, hparams, tx_input_dim):
    if args.tx_encoder == 'mlp':
        tx_encoder_hparams = {'tx_input_dim': tx_input_dim}
        tx_encoder_hparams.update({hparam_name:hparams[hparam_name] for hparam_name in ['tx_mlp_hidden_dims', 'tx_mlp_dropout', 'tx_mlp_norm', 'tx_mlp_actn', 'tx_mlp_order']})
    elif args.tx_encoder == 'chemcpa':
        _, _, experiment_config = read_config(PROJECT_DIR + hparams['tx_chemcpa_config_path'])
        configs = generate_configs(experiment_config)
        assert len(configs) == 1
        tx_encoder_hparams = configs[0]
    else:
        raise NotImplementedError
    return tx_encoder_hparams


def get_proj_hparams(hparams):
    proj_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['proj_hidden_dims', 'proj_dropout', 'proj_norm', 'proj_actn', 'proj_order']}
    return proj_hparams


def get_transformer_fusion_hparams(args, hparams):
    transformer_hparams =  {hparam_name:hparams[hparam_name] for hparam_name in ['transformer_num_layers', 'transformer_att_heads', 'transformer_head_dim', 'transformer_ffn_dim', 'transformer_dropout', 'transformer_actn', 'transformer_norm_first']}
    transformer_hparams['transformer_batch_first'] = args.transformer_batch_first
    transformer_hparams['transformer_agg'] = args.transformer_agg
    return transformer_hparams
