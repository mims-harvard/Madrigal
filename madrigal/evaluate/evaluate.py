from itertools import combinations
from re import M
from typing import List, Union, Dict
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torchdrug.data import PackedMolecule

from .GeomCA import GeomCA
from .metrics import get_metrics
from ..utils import NUM_NON_TX_MODALITIES, from_indices_to_tensor, to_device, NUM_MODALITIES
from .eval_utils import (
    AVERAGE, 
    KEY_METRIC_DICT,
    MODALITY2NUMBER_LIST, 
    NUMBER2MODALITY, 
    FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_MAP, 
    FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_BETWEEN_MAP, 
    FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_WITHIN_MAP, 
    stacked_inst_dist_topk_accuracy, 
    uniform_loss, 
    alignment_loss, 
    knn_classifier, 
    foscttm, 
    get_evaluate_masks,
)

SEED = 42


####################
# DDI training evaluation
####################
@torch.no_grad()
def evaluate_ft(model, batch, loss_fn, k, task, split, finetune_mode, best_metrics, subgroup=False, verbose=True, device='cpu', logger=None, wandb=None, epoch=None, **kwargs):
    """ Wrapper for `evaluate_ddi`
    """
    save_scores = kwargs.pop('save_scores', False)
    output_dir = kwargs.pop('output_dir', None)
    label_map = kwargs.pop('label_map', None)
    data_source = kwargs.pop("data_source", "")
    assert (not save_scores) or (output_dir is not None and label_map is not None)
    assert (not save_scores) or (AVERAGE in {'micro', 'macro'})
    
    model.eval()
    model.to(device)
    
    batch_head = to_device(batch['head'], device)  # dict
    batch_tail = to_device(batch['tail'], device)
    batch_kg = to_device(batch['kg'], device)
    head_masks_base = batch['head']['masks']  # to device later
    tail_masks_base = batch['tail']['masks']  # to device later
    ddi_head_indices = batch['edge_indices']['head']
    ddi_tail_indices = batch['edge_indices']['tail']
    ddi_labels = batch['edge_indices']['label']
    
    if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
        ddi_pos_neg_samples = batch['edge_indices']['pos_neg'].float()
    elif isinstance(loss_fn, nn.CrossEntropyLoss):
        ddi_pos_neg_samples = batch['edge_indices']['pos_neg'].long()
    else:
        raise NotImplementedError
    
    if split in {'train', 'val', 'test', 'val_between', 'val_within', 'test_between', 'test_within'}:
        if 'between' in split:
            eval_type_for_model_selection = FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_BETWEEN_MAP[finetune_mode]
        elif 'within' in split:
            eval_type_for_model_selection = FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_WITHIN_MAP[finetune_mode]
        else:
            eval_type_for_model_selection = FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_MAP[finetune_mode]
        
        if "ONSIDES" in data_source: eval_type_for_model_selection = "full_full"
        
        split_eval_types = {
            'train': [
                'full_full', 'str_str', 'str_full', 
                'kg_kg', 'cv_cv', 'tx_tx', 
                'str+kg_full', 'str+cv_full', 'str+tx_full', 'str+cv+tx_full',
                'str+tx_str+tx', 'str+cv+tx_str+cv+tx',
            ],
            'val': [
                'full_full', 'str_str', 
                'str+tx_str+tx', 'str+cv+tx_str+cv+tx',
            ],
            'test': [
                'full_full', 'str_str', 
                'str+tx_str+tx', 'str+cv+tx_str+cv+tx',
            ],
            'between': [
                'full_full', 'str_str', 'str_full', 
                'kg_kg', 'cv_cv', 'tx_tx', 
                'str+cv_full', 'str+tx_full', 'str+cv+tx_full',
            ],
            'within': [
                'full_full', 'str_str', 
                'kg_kg', 'cv_cv', 'tx_tx', 
                'str+cv_str+cv', 'str+tx_str+tx', 'str+cv+tx_str+cv+tx'
            ],
        }
        
        for eval_type in split_eval_types[split.split('_')[-1]]:
            if eval_type != eval_type_for_model_selection:
                evaluate_ddi(model, batch_head, batch_tail, batch_kg, head_masks_base, tail_masks_base, ddi_head_indices, ddi_tail_indices, ddi_labels, ddi_pos_neg_samples, loss_fn, k, task, eval_type=eval_type, split=split, finetune_mode=finetune_mode, best_metrics=best_metrics, subgroup=subgroup, verbose=verbose, device=device, logger=logger, wandb=wandb, epoch=epoch, save_scores=save_scores, output_dir=output_dir, label_map=label_map, data_source=data_source, **kwargs)
            else:
                key_metric = evaluate_ddi(model, batch_head, batch_tail, batch_kg, head_masks_base, tail_masks_base, ddi_head_indices, ddi_tail_indices, ddi_labels, ddi_pos_neg_samples, loss_fn, k, task, eval_type=eval_type, split=split, finetune_mode=finetune_mode, best_metrics=best_metrics, subgroup=subgroup, verbose=verbose, device=device, logger=logger, wandb=wandb, epoch=epoch, save_scores=save_scores, output_dir=output_dir, label_map=label_map, data_source=data_source, **kwargs)
    


    else:
        raise ValueError('`split` must be either "train", "val", "test", "val_between", "val_within", "val_between_easy", "val_within_easy", "test_between", "test_within", "test_between_easy", or "test_within_easy".')

    return key_metric


@torch.no_grad()
def evaluate_ddi(model, batch_head, batch_tail, batch_kg, head_masks_base, tail_masks_base, ddi_head_indices, ddi_tail_indices, ddi_labels, ddi_pos_neg_samples, loss_fn, k, task, eval_type, split, finetune_mode, best_metrics, subgroup=False, verbose=True, device='cpu', logger=None, wandb=None, epoch=None, **kwargs):
    """
    Args:
        :param: eval_type: In this string, for the between case, the first modality composition refers to the novel set of drugs (src nodes, validation set), all of which have full modality availability, and the second corresponds to the known set of drugs (dst nodes, train set). For the other cases, they are used (and the metrics are calculated) bidirectionally.
    """
    assert len(eval_type.split('_')) == 2
    save_scores = kwargs.get('save_scores', False)
    output_dir = kwargs.get('output_dir', None)
    label_map = kwargs.get('label_map', None)
    data_source = kwargs.get("data_source", "")
    
    masks_head, masks_tail = get_evaluate_masks(head_masks_base, tail_masks_base, eval_type, finetune_mode, device)
    
    # NOTE: By default, the train ddis are undirected, and the val ddis are directed. However, sometimes we need to evalulate ddis in both directions.
    # (1) if split == 'train' AND eval_type (head and tail modalities) is symmetric, make ddis directed (because train DDIs are automatically made bidirectional during data loading), otherwise keep using undirected ddis.
    # (2) if split == 'val' or 'test' or 'val_within' or 'val_within_easy' or 'test_within' or 'test_within_easy', AND eval_type (head and tail modalities) is asymmetric, make ddis undirected, otherwise keep using directed ddis.
    # (3) if split == 'val_between' or 'val_between_easy' or 'test_between' or 'test_between_easy', keep using directed ddis (for all eval_types, should only consider directed ddis)
    
    # (1)
    if (split == 'train' and eval_type in {'str_str', 'full_full', 'kg_kg', 'cv_cv', 'tx_tx'}):
        directed_indices_bool = ddi_head_indices < ddi_tail_indices
        ddi_head_indices = ddi_head_indices[directed_indices_bool]
        ddi_tail_indices = ddi_tail_indices[directed_indices_bool]
        ddi_labels = ddi_labels[directed_indices_bool]
        ddi_pos_neg_samples = ddi_pos_neg_samples[directed_indices_bool]
        
    elif (split == 'train' and eval_type == 'str_full'):
        pass
    
    # (2)
    elif split in {'val', 'val_within', 'val_within_easy', 'test', 'test_within', 'test_within_easy'} and eval_type.split('_')[0] != eval_type.split('_')[1]:
        ddi_head_indices, ddi_tail_indices = torch.cat([ddi_head_indices, ddi_tail_indices]), torch.cat([ddi_tail_indices, ddi_head_indices])
        ddi_labels = ddi_labels.repeat(2)
        ddi_pos_neg_samples = ddi_pos_neg_samples.repeat(2)
        
        if split in {'val', 'val_within', 'test', 'test_within'} and eval_type == 'str_full':
            # NOTE: Same as above.
            pass
        
    # (3)
    elif split in {'val_between', 'val_between_easy', 'test_between', 'test_between_easy'}:
        pass
    
    pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_head, device), to_device(masks_tail, device), batch_kg, single_drug="ONSIDES" in data_source)).detach().cpu()
    if "ONSIDES" in data_source: 
        pred_ddis = pred_ddis[ddi_head_indices, ddi_labels]
    else:
        pred_ddis = pred_ddis[ddi_labels, ddi_head_indices, ddi_tail_indices]
    loss = loss_fn(pred_ddis, ddi_pos_neg_samples).item()
    
    logger.info(f'Evaluated for {task} classification task on {split} set with {eval_type} eval_type.')
    
    # Now get metrics
    average = AVERAGE
    average_opt = None if save_scores else average
    
    metrics_dict, pos_samples = get_metrics(
        pred_ddis.numpy(), 
        ddi_pos_neg_samples.numpy(), 
        ddi_labels.numpy(), 
        k=k, 
        task=task,
        logger=logger, 
        average=average_opt,
        verbose=verbose
    )
    
    # NOTE: If we are saving scores, the return from get_metrics will be the raw label-stratified metrics. We need to average the metrics here in order to report them.
    if average_opt is None:
        raw_metrics_dict = metrics_dict
        metric_names = list(metrics_dict.keys())
        if average == 'macro':
            averaged_metrics = np.array(list(metrics_dict.values())).T.mean(axis=0)
        elif average == 'weighted':
            averaged_metrics = np.array(list(metrics_dict.values())) @ pos_samples / pos_samples.sum()
        metrics_dict = {metric_name: metric for metric_name, metric in zip(metric_names, averaged_metrics)}
        metrics_str = ', '.join([f'{metric_name}: {metric:.4f}' for metric_name, metric in metrics_dict.items()])
        logger.info(f'{average}-averaged metrics: {metrics_str}')
    
    logger.info(f"{data_source}{split}_{eval_type}_loss: {loss:.4f}")
    wandb.log({f'{data_source}{split}_{eval_type}_loss': loss}, step=epoch)
    wandb.log({f'{data_source}{split}_{eval_type}_{metric_name}': metric if metric == metric else 0 for metric_name, metric in metrics_dict.items()}, step=epoch)
        
    key_metric_name = KEY_METRIC_DICT[task]
    assert key_metric_name in metrics_dict.keys()
    key_metric = metrics_dict[key_metric_name]
    if best_metrics is not None and (f'{data_source}best_{split}_{eval_type}_{key_metric_name}' not in best_metrics.keys() or key_metric > best_metrics[f'{data_source}best_{split}_{eval_type}_{key_metric_name}']):
        # NOTE: the "best" metrics are recorded with respect to each eval_type individually, e.g. the epoch that best metrics are recorded for val "str_full" will definitely be different from that of val "full_full"
        for metric_name, metric in metrics_dict.items():
            best_metrics[f'{data_source}best_{split}_{eval_type}_{metric_name}'] = metric
                
    if save_scores:
        # per-outcome metrics and positive counts
        raw_metrics_df = pd.DataFrame.from_dict(raw_metrics_dict, orient='columns')
        raw_metrics_df['pos_samples'] = pos_samples.astype(int)
        raw_metrics_df['label'] = label_map[ddi_labels.unique().tolist()]
        raw_metrics_df.to_csv(output_dir + f'{split}_{eval_type}_{finetune_mode}_label_stratified_metrics.csv', index=False)
        
    return key_metric


###
# Pretraining evaluation
###
@torch.no_grad()
def evaluate_pt(model, drugs, masks, too_hard_neg_mask, collator, split, wandb, logger, device, epoch):
    """ Wrapper for `evaluate_pretrain` across subset pair combinations
    """
    str_kg_ret_metrics = evaluate_pretrain_subsets(model, drugs, masks, too_hard_neg_mask, collator, MODALITY2NUMBER_LIST["str"], MODALITY2NUMBER_LIST["kg"], device)
    str_cv_ret_metrics = evaluate_pretrain_subsets(model, drugs, masks, too_hard_neg_mask, collator, MODALITY2NUMBER_LIST["str"], MODALITY2NUMBER_LIST["cv"], device)
    str_tx_mcf7_ret_metrics = evaluate_pretrain_subsets(model, drugs, masks, too_hard_neg_mask, collator, MODALITY2NUMBER_LIST["str"], MODALITY2NUMBER_LIST["tx_mcf7"], device)
    str_tx_pc3_ret_metrics = evaluate_pretrain_subsets(model, drugs, masks, too_hard_neg_mask, collator, MODALITY2NUMBER_LIST["str"], MODALITY2NUMBER_LIST["tx_pc3"], device)
    str_tx_vcap_ret_metrics = evaluate_pretrain_subsets(model, drugs, masks, too_hard_neg_mask, collator, MODALITY2NUMBER_LIST["str"], MODALITY2NUMBER_LIST["tx_vcap"], device)
    
    logger.info('Start logging topk, foscttm, and loss metrics...')
    report_dict = {}
    for comp_pair, metrics in zip(
        ['str v kg', 'str v cv', 
        'str v tx_mcf7', 'str v tx_pc3', 'str v tx_vcap'], 
        [str_kg_ret_metrics, str_cv_ret_metrics, str_tx_mcf7_ret_metrics, str_tx_pc3_ret_metrics, str_tx_vcap_ret_metrics]
    ):
        logger.info(len(metrics))
        metrics = iter(metrics)
        count = 0
        for side_type in ['one-side', 'both-side']:
            for embed_type in ['embed', 'CL-head']:
                for topk in ['top20', 'top5', 'top1']:
                    evaluation_type_text = f'{split} {topk} acc {comp_pair} {embed_type} {side_type} (cosine)'
                    report_dict[evaluation_type_text] = next(metrics)
                    count += 1
        logger.info(count)
        report_dict[f'{split} loss {comp_pair}'] = next(metrics)
        report_dict[f'{split} foscttm mu {comp_pair}'] = next(metrics)
        
    wandb.log(report_dict, step=epoch)
    
    # Uniformity metrics
    logger.info('Start logging uniformity metrics...')
    all_embeds = {}
    for mod in sum([
        MODALITY2NUMBER_LIST[mod] for mod in 
        ['str', 'kg', 'cv', 'tx_mcf7', 'tx_pc3', 'tx_vcap']
    ], start=[]):
    # [0,1,2,13,15,17]:  # str, kg, cv, mcf7, pc3, vcap
        valid_drugs = drugs[(1 - masks[drugs, :][:, [mod]]).sum(axis=1) == 1]  # Get valid drugs that have the required modalities
        valid_drugs, valid_data = to_device(collator([valid_drugs]), device)
        valid_mols, valid_kgs, valid_cvs, valid_tx_all_cell_lines = valid_data
        other_tabular_mod_data = {}
        masks_subset = to_device(from_indices_to_tensor([mod], masks.shape[1]).repeat(valid_drugs.shape[0], 1).bool(), device)
        embeds = model.base_encoder(valid_drugs, masks_subset, valid_mols, valid_kgs, valid_cvs, valid_tx_all_cell_lines, raw_encoder_output=model.raw_encoder_output, **other_tabular_mod_data).cpu()
        uniform_l = uniform_loss(embeds)

        indices_str = str(mod)
        wandb.log({f'{split} uniformity loss {NUMBER2MODALITY[indices_str]}': uniform_l}, step=epoch)
        logger.info(f'{split} uniformity loss {NUMBER2MODALITY[indices_str]}: {uniform_l}')

        all_embeds[indices_str] = {}
        all_embeds[indices_str]['embeds'] = embeds
        all_embeds[indices_str]['drugs'] = valid_drugs.detach().cpu().numpy()
    
    logger.info('Start logging alignment metrics...')
    for mod1, mod2 in [(0, i) for i in list(range(1, NUM_NON_TX_MODALITIES)) + MODALITY2NUMBER_LIST['tx_mcf7'] + MODALITY2NUMBER_LIST['tx_pc3'] + MODALITY2NUMBER_LIST['tx_vcap']]:
    # [(0, 1), (0, 2), (1, 2), (0, 13), (0, 15), (0, 17)]:
        mod_pair_str = NUMBER2MODALITY[str(mod1)] + ' v ' + NUMBER2MODALITY[str(mod2)]
        
        # NOTE: Need to ensure the extracted embeddings are aligned
        drugs_mod1 = all_embeds[str(mod1)]['drugs']
        drugs_mod2 = all_embeds[str(mod2)]['drugs']
        
        
        # get sorted shared drugs
        shared_drugs = np.intersect1d(drugs_mod1, drugs_mod2)
        
        
        
        # find the bool indices of the shared drugs in each input
        shared_bool_indices_mod1 = np.isin(drugs_mod1, shared_drugs)
        shared_bool_indices_mod2 = np.isin(drugs_mod2, shared_drugs)
        
        # argsort the extracted shared drugs
        sorted_drugs_pos_mod1 = np.argsort(drugs_mod1[shared_bool_indices_mod1])
        sorted_drugs_pos_mod2 = np.argsort(drugs_mod2[shared_bool_indices_mod2])
        
        # extract corresponding embeddings, sorted by drugs
        shared_embeddings_mod1 = all_embeds[str(mod1)]['embeds'][shared_bool_indices_mod1][sorted_drugs_pos_mod1]
        shared_embeddings_mod2 = all_embeds[str(mod2)]['embeds'][shared_bool_indices_mod2][sorted_drugs_pos_mod2]
        
        alignment_l = alignment_loss(shared_embeddings_mod1, shared_embeddings_mod2)
        wandb.log({f'{split} alignment loss {mod_pair_str}': alignment_l}, step=epoch)
        logger.info(f'{split} alignment loss {mod_pair_str}: {alignment_l}')
        
    return all_embeds


@torch.no_grad()
def evaluate_pretrain_subsets(model: nn.Module, drugs: np.ndarray, masks: np.ndarray, too_hard_neg_mask: torch.BoolTensor, collator: object, subset1: List[int], subset2: List[int], device: torch.device):
    """ Evaluate pretraining performance between representations of two specific subsets of modalities
    """
    valid_drugs = drugs[(1 - masks[drugs, :][:, np.unique(subset1 + subset2)]).sum(axis=1) == len(np.unique(subset1 + subset2))]  # Get drugs that have the required modalities (subset1 and subset2)
    valid_drugs = torch.from_numpy(np.random.choice(valid_drugs, size=min(1000, valid_drugs.shape[0]), replace=False))
    _, valid_data = to_device(collator([valid_drugs]), device)
    valid_mols, valid_kgs, valid_cvs, valid_tx_all_cell_lines = valid_data
    other_tabular_mod_data = {}
    
    masks1 = to_device(from_indices_to_tensor(subset1, masks.shape[1]).repeat(valid_drugs.shape[0], 1).bool(), device)
    masks2 = to_device(from_indices_to_tensor(subset2, masks.shape[1]).repeat(valid_drugs.shape[0], 1).bool(), device)

    embeds1 = model.base_encoder(valid_drugs.to(device), masks1, valid_mols, valid_kgs, valid_cvs, valid_tx_all_cell_lines, raw_encoder_output=model.raw_encoder_output, **other_tabular_mod_data).cpu()
    embeds2 = model.base_encoder(valid_drugs.to(device), masks2, valid_mols, valid_kgs, valid_cvs, valid_tx_all_cell_lines, raw_encoder_output=model.raw_encoder_output, **other_tabular_mod_data).cpu()

    # Get top5 and top1 accuracy
    top5_acc_cosine_real, top20_real_both, top5_real_both, top1_real_both, _, _ = get_inst_dist_topk_accuracy(embeds1, embeds2, 5, 'cosine')  # "real" embeddings for downstream
    top1_acc_cosine_real, _, _, _, _, _ = get_inst_dist_topk_accuracy(embeds1, embeds2, 1, 'cosine')
    top20_acc_cosine_real, _, _, _, _, _ = get_inst_dist_topk_accuracy(embeds1, embeds2, 20, 'cosine')

    # FOSCTTM
    foscttm_mu1, foscttm_std1 = foscttm(embeds1, embeds2)
    foscttm_mu2, foscttm_std2 = foscttm(embeds2, embeds1)
    foscttm_mu = (foscttm_mu1 + foscttm_mu2) / 2

    if too_hard_neg_mask is not None:
        too_hard_neg_mask = too_hard_neg_mask[valid_drugs, :][: , valid_drugs]
    
    valid_data = to_device(valid_data, "cpu")
    aug1, aug2, (logits, labels, loss) = model.cpu()(valid_drugs.to("cpu"), masks1.to("cpu"), masks2.to("cpu"), to_device(too_hard_neg_mask, "cpu"), valid_data, None, None)
    model.to(device)
    
    top5_acc_cosine_redundant, top20_redundant_both, top5_redundant_both, top1_redundant_both, _, _ = get_inst_dist_topk_accuracy(aug1.cpu(), aug2.cpu(), 5, 'cosine')
    top1_acc_cosine_redundant, _, _, _, _, _ = get_inst_dist_topk_accuracy(aug1.cpu(), aug2.cpu(), 1, 'cosine')
    top20_acc_cosine_redundant, _, _, _, _, _ = get_inst_dist_topk_accuracy(aug1.cpu(), aug2.cpu(), 20, 'cosine')

    return top20_acc_cosine_real, top5_acc_cosine_real, top1_acc_cosine_real, top20_acc_cosine_redundant, top5_acc_cosine_redundant, top1_acc_cosine_redundant, top20_real_both, top5_real_both, top1_real_both, top20_redundant_both, top5_redundant_both, top1_redundant_both, loss, foscttm_mu


def get_inst_dist_topk_accuracy(embeds1, embeds2, k: int, metric: str = 'cosine'):
    """ Calculate inst dist top k accuracy averaged for each drug and each modality among representations from THE OTHER (but not both) modality
    """
    if metric=='euclidean':
        sim_mat = distance_matrix(embeds1, embeds2, p=2)  # minkowski's p
        topk_rows = torch.topk(torch.from_numpy(sim_mat), dim=1, k=k, largest=False)[1]
        topk_cols = torch.topk(torch.from_numpy(sim_mat.T), dim=1, k=k, largest=False)[1]
    elif metric=='cosine':
        embeds1 = embeds1 / torch.norm(embeds1, dim=1, keepdim=True)
        embeds2 = embeds2 / torch.norm(embeds2, dim=1, keepdim=True)
        sim_mat = embeds1 @ embeds2.T  # cosine similarity between drug view subset features
        topk_rows = torch.topk(sim_mat, dim=1, k=k, largest=True)[1]
        topk_cols = torch.topk(sim_mat.T, dim=1, k=k, largest=True)[1]
    else:
        raise NotImplementedError
    
    topk_acc = 1 - (from_indices_to_tensor(topk_rows, (topk_rows.shape[0], topk_rows.shape[0])).diag().sum() + from_indices_to_tensor(topk_cols, (topk_cols.shape[0], topk_cols.shape[0])).diag().sum()) / (topk_rows.shape[0] + topk_cols.shape[0])  # Fill a square matrix with 1s in the topk positions and sum the diagonal
    
    features = torch.cat([embeds1, embeds2], dim=0)
    labels = torch.cat([torch.arange(embeds1.shape[0])] * 2, dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives

    # select only the negatives the negatives


    logits = similarity_matrix
    top1, top5, top20 = stacked_inst_dist_topk_accuracy(logits, labels)

    return topk_acc.item(), top20.item(), top5.item(), top1.item(), topk_rows, topk_cols


###
# Pretrained embeddings evaluation (kNN ATC classification; alignment of modalities)
###
def evaluate_final_embeds(train_outputs, val_outputs, save_dir, wandb, logger, epoch):
    for (indices_str_train1, train_outputs_subset1), (indices_str_train2, train_outputs_subset2) in combinations(train_outputs.items(), 2):
        get_alignment_metrics('train ' + NUMBER2MODALITY[indices_str_train1] + ' v ' + NUMBER2MODALITY[indices_str_train2], train_outputs_subset1, train_outputs_subset2, save_dir, wandb, logger, epoch)
    
    for (indices_str_val1, val_outputs_subset1), (indices_str_val2, val_outputs_subset2) in combinations(val_outputs.items(), 2):
        get_alignment_metrics('val ' + NUMBER2MODALITY[indices_str_val1] + ' v ' + NUMBER2MODALITY[indices_str_val2], val_outputs_subset1, val_outputs_subset2, save_dir, wandb, logger, epoch)


def get_alignment_metrics(pair_name, outputs_subset1, outputs_subset2, save_dir, wandb, logger, epoch):
    subset_drugs1 = outputs_subset1['drugs']
    subset_embeds1 = outputs_subset1['embeds']
    subset_drugs2 = outputs_subset2['drugs']
    subset_embeds2 = outputs_subset2['embeds']

    drug_ind2embed_ind1 = {drug_ind:ind for ind, drug_ind in enumerate(subset_drugs1)}
    drug_ind2embed_ind2 = {drug_ind:ind for ind, drug_ind in enumerate(subset_drugs2)}

    valid_drugs = np.intersect1d(subset_drugs1, subset_drugs2)
    
    # this is not the most efficient solution, but it ensures the embeddings extracted are aligned (though GeomCA doesn't need this), in order to ensure that this metric is computed with the same set of drugs as the other alignment metrics
    valid_embeds1 = torch.stack([subset_embeds1[drug_ind2embed_ind1[drug_ind]] for drug_ind in valid_drugs]).numpy()
    valid_embeds2 = torch.stack([subset_embeds2[drug_ind2embed_ind2[drug_ind]] for drug_ind in valid_drugs]).numpy()

    # GeomCA
    ## Setting up parameters and pass to GeomCA
    GeomCA_parameters = {
        'experiment_path': save_dir,      # Path to the experiment folder
        'experiment_filename_prefix': '',                     # Potential filename prefix to use
        'subfolder' : pair_name,
        'Rdist_ratio': 1.0,						# Percentage of R points to use for epsilon estimation
        'Rdist_percentile': 5,					# Percentile of R distances D determining epsilon estimate
        'gamma': 1,								# Portion of epsilon to use for sparsification: delta = gamma * epsilon(p)
        'reduceR': True,              			# Whether to reduce number of points in R
        'reduceE': True,              			# Whether to reduce number of points in E
        'sparsify': True,                		# Reduction type: sampling or sparsification
        'n_Rsamples': None, 					# Number of R samples if reducing by sampling
        'n_Esamples': None,						# Number of E samples if reducing by sampling
        'log_reduced': False,               	# Whether to save the reduced representations 
        'comp_consistency_threshold': 0.0,      # Component consistency threshold eta_c
        'comp_quality_threshold': 0.0,          # Component quality score eta_q
        'random_seed': SEED}

    # NOTE: We actually don't need to do this subsetting, but for efficiency (we have 15 pairs of embeddings, so we don't want each to take too long)
    GeomCA_graph = GeomCA(valid_embeds1, valid_embeds2, GeomCA_parameters, load_existing=False, subfolder=GeomCA_parameters["subfolder"])
    GeomCA_results = GeomCA_graph.get_connected_components()
    
    wandb.log({f'{pair_name} GeomCA precision': GeomCA_results[1]['precision'], f'{pair_name} GeomCA recall': GeomCA_results[1]['recall'], f'{pair_name} GeomCA network consistency': GeomCA_results[1]['network_consistency'], f'{pair_name} GeomCA network quality': GeomCA_results[1]['network_quality'], f'{pair_name} GeomCA sample size': len(valid_drugs)}, step=epoch)
    logger.info(f'GeomCA evaluation...\n{pair_name} precision: {GeomCA_results[1]["precision"]}, recall: {GeomCA_results[1]["recall"]}, network consistency: {GeomCA_results[1]["network_consistency"]}, network quality: {GeomCA_results[1]["network_quality"]}, sample size: {len(valid_drugs)}')
