import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, cdist, squareform

import plotly.express as px
from umap import UMAP

from ..utils import CELL_LINES, NUM_MODALITIES, NUM_NON_TX_MODALITIES, NON_TX_MODALITIES, powerset, to_device

KEY_METRIC_DICT = {'multilabel':'auprc', 'multiclass':'auprc'}
AVERAGE = 'macro'
NUMBER2MODALITY = {
    "".join(str(ind) for ind in tup): "+".join(NON_TX_MODALITIES[i] for i in tup)
    for tup in 
    list(powerset(range(NUM_NON_TX_MODALITIES)))[1:]
}
NUMBER2MODALITY.update({
    str(i + NUM_NON_TX_MODALITIES) : f'tx_{cell_line}'
    for i, cell_line in enumerate(CELL_LINES)
})
MODALITY2NUMBER_LIST = {
    mod: [i]
    for i, mod in enumerate(NON_TX_MODALITIES)
}
MODALITY2NUMBER_LIST.update({
    f'tx_{cell_line}' : [i + NUM_NON_TX_MODALITIES]
    for i, cell_line in enumerate(CELL_LINES)
})
MODALITY2NUMBER_LIST.update({
    'tx' : [i + NUM_NON_TX_MODALITIES for i in range(len(CELL_LINES))]
})
FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_BETWEEN_MAP = {
    'ablation_str_str':'str_str', 
    'ablation_kg_kg_subset':'kg_kg', 
    'ablation_kg_kg_padded':'kg_kg', 
    'ablation_cv_cv_padded': 'cv_cv',
    'ablation_tx_tx_padded': 'tx_tx',
    'ablation_str_random_str+kg_full_sample':'str_full', 
    'ablation_str_random_str+cv_full_sample':'str_full',
    'ablation_str_random_str+tx_full_sample':'str+tx_full',
    'ablation_str_random_str+kg+cv_full_sample':'str_full', 
    'ablation_str_random_str+kg+tx_full_sample':'str+tx_full',
    'ablation_str_random_str+cv+tx_full_sample':'str+tx_full',
    'str_full':'str_full', 
    'full_full':'str+tx_full', 
    'double_random':'str+tx_full', 
    'str_random_sample':'str+tx_full', 
    'str_str+random_sample':'str+tx_full', 
    'full_str+random_sample':'str+tx_full',
}
FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_WITHIN_MAP = {
    'ablation_str_str':'str_str', 
    'ablation_kg_kg_subset':'kg_kg', 
    'ablation_kg_kg_padded':'kg_kg', 
    'ablation_cv_cv_padded': 'cv_cv',
    'ablation_tx_tx_padded': 'tx_tx',
    'ablation_str_random_str+kg_full_sample':'full_full', 
    'ablation_str_random_str+cv_full_sample':'full_full',
    'ablation_str_random_str+tx_full_sample':'full_full',
    'ablation_str_random_str+kg+cv_full_sample':'full_full', 
    'ablation_str_random_str+kg+tx_full_sample':'full_full',
    'ablation_str_random_str+cv+tx_full_sample':'full_full',
    'str_full':'str_str', 
    'full_full':'str_str', 
    'double_random':'str_str', 
    'str_random_sample':'str_str', 
    'str_str+random_sample':'str_str', 
    'full_str+random_sample':'str_str',
}
FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_MAP = {
    'ablation_str_str':'str_str', 
    'ablation_kg_kg_subset':'kg_kg',
    'ablation_kg_kg_padded':'kg_kg', 
    'ablaiton_cv_cv_padded': 'cv_cv',
    'ablation_tx_tx_padded': 'tx_tx',
    'ablation_str_random_str+kg_full_sample':'full_full', 
    'ablation_str_random_str+cv_full_sample':'full_full',
    'ablation_str_random_str+tx_full_sample':'full_full',
    'ablation_str_random_str+kg+cv_full_sample':'full_full', 
    'ablation_str_random_str+kg+tx_full_sample':'full_full',
    'ablation_str_random_str+cv+tx_full_sample':'full_full',
    'str_full':'full_full', 
    'full_full':'full_full', 
    'double_random':'full_full', 
    'str_random_sample':'full_full', 
    'str_str+random_sample':'full_full', 
    'full_str+random_sample':'full_full',
}
FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP = {
    "ablation_str_str": list(range(1, NUM_MODALITIES)),
    "ablation_kg_kg_padded": [i for i in range(0, NUM_NON_TX_MODALITIES) if NON_TX_MODALITIES[i] != 'kg'] + [i + NUM_NON_TX_MODALITIES for i in range(len(CELL_LINES))],
    "ablation_cv_cv_padded": [i for i in range(0, NUM_NON_TX_MODALITIES) if NON_TX_MODALITIES[i] != 'cv'] + [i + NUM_NON_TX_MODALITIES for i in range(len(CELL_LINES))],
    "ablation_tx_tx_padded": list(range(0, NUM_NON_TX_MODALITIES)),
    "ablation_str_random_str+kg_full_sample": [i for i in range(0, NUM_NON_TX_MODALITIES) if not NON_TX_MODALITIES[i] in {'str', 'kg'}] + [i + NUM_NON_TX_MODALITIES for i in range(len(CELL_LINES))],
    "ablation_str_random_str+cv_full_sample": [i for i in range(0, NUM_NON_TX_MODALITIES) if not NON_TX_MODALITIES[i] in {'str', 'cv'}] + [i + NUM_NON_TX_MODALITIES for i in range(len(CELL_LINES))],
    "ablation_str_random_str+tx_full_sample": list(range(1, NUM_NON_TX_MODALITIES)),
    "ablation_str_random_str+kg+cv_full_sample": [i for i in range(0, NUM_NON_TX_MODALITIES) if not NON_TX_MODALITIES[i] in {'str', 'kg', 'cv'}] + [i + NUM_NON_TX_MODALITIES for i in range(len(CELL_LINES))],
    "ablation_str_random_str+kg+tx_full_sample": [i for i in range(0, NUM_NON_TX_MODALITIES) if not NON_TX_MODALITIES[i] in {'str', 'kg'}],
    "ablation_str_random_str+cv+tx_full_sample": [i for i in range(0, NUM_NON_TX_MODALITIES) if not NON_TX_MODALITIES[i] in {'str', 'cv'}],
    "ablation_str_random_str+kg+cv+tx_full_sample": [i for i in range(0, NUM_NON_TX_MODALITIES) if not NON_TX_MODALITIES[i] in {'str', 'kg', 'cv'}],
}
K = 50


def uniform_loss(x, t=2):
    x = x / torch.norm(x, dim=1, keepdim=True)  # NOTE: normalize before uniformity
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def alignment_loss(x1, x2, alpha=2):
    x1 = x1 / torch.norm(x1, dim=1, keepdim=True)
    x2 = x2 / torch.norm(x2, dim=1, keepdim=True)
    return (x1 - x2).norm(p=2, dim=1).pow(alpha).mean()


def stacked_inst_dist_topk_accuracy(output, target, topk=(1, 5, 20)):
    """ Computes the accuracy over the k top predictions for the specified values of k, among representations from both modalities
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(torch.where(target==1)[1].expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    
    return res


def knn_classifier(train_features, train_labels, test_features, test_labels, metric='cosine', k=5, T=1, num_classes=2):
    """ From DINO: https://github.com/facebookresearch/dino/blob/main/eval_knn.py
    To account for the multilabel nature of our data (drug ATC classification), we report separately for each of the labels. This also helps us be aware of the different imbalanceness of different labels.
    """
    top1, total = 0.0, 0
    num_test_samples, num_chunks = test_labels.shape[0], 1
    imgs_per_chunk = num_test_samples // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    
    for idx in range(0, num_test_samples, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_samples), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_samples)]
        batch_size = targets.shape[0]

        # calculate the similarity/distance and compute top-k neighbors
        if metric=='euclidean':
            similarity = torch.from_numpy(distance_matrix(features, train_features, p=2))  # p is minkowski's p
            distances, indices = similarity.topk(k, largest=False, sorted=True)
        elif metric=='cosine':
            features_norm = features / torch.norm(features, dim=1, keepdim=True)
            train_features_norm = train_features / torch.norm(train_features, dim=1, keepdim=True)  # L2-norm for cosine similarity
            similarity = torch.mm(features_norm, train_features_norm.T)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
        
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)  # get labels of all topk neighbors, shape=[test_batch_size, k]

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()  # construct empty one-hot labels for topk neighbors, flattened, shape=[test_batch_size*k, num_classes]
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1).type(torch.int64), 1)  # fill in the one-hot labels for topk neighbors, flattened, shape=[test_batch_size*k, num_classes]
        distances_transform = distances.clone().div_(T).exp_()  # shape=[test_batch_size, k]
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)
    
    top1 = top1 / total
    
    return top1


def foscttm(R, E):
    foscttm_vec = torch.zeros(E.shape[0])
    for i in range(E.shape[0]):
        dist = torch.norm(R - E[i], dim=-1, keepdim=True)
        foscttm = torch.sum(dist < dist[i]) / dist.shape[0]  # get the percentage of elements closer than the true pair
        foscttm_vec[i] = foscttm
    mu, std = torch.mean(foscttm_vec), torch.std(foscttm_vec)       

    
    print(f"FOSCTTM Metrics, Mean: {mu}, std: {std}")
    
    return mu, std


def get_full_evaluate_mask_for_finetune_mode(finetune_mode, head_or_tail_masks_base):
    """
    Get full modality masks based on finetune_mode.
    For non-ablation modes, all modalities are available.
    For ablation modes, mask out the modalities that have never been seen during training. If the evaluation is between kg-kg  or cv-cv, then need to also ensure that the corresponding modality is always available.
    """
    head_or_tail_masks = head_or_tail_masks_base.clone()
    if 'ablation' in finetune_mode:
        head_or_tail_masks[:, FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP[finetune_mode]] = True
        if 'kg_kg' in finetune_mode:
            head_or_tail_masks[:, 1] = False
        elif 'cv_cv' in finetune_mode:
            head_or_tail_masks[:, 2] = False
        elif 'tx_tx' in finetune_mode:
            head_or_tail_masks[:, 3:] = False

    return head_or_tail_masks


def get_modality_evaluate_mask(head_or_tail_masks_base, modality):
    if '+' not in modality:
        modality_col_list = MODALITY2NUMBER_LIST[modality]
        head_or_tail_masks = torch.ones_like(head_or_tail_masks_base)
        head_or_tail_masks[:, modality_col_list] = 0
        head_or_tail_masks = head_or_tail_masks.bool()
    else:  # NOTE: if '+' is in modality, e.g. 'str+cv+tx', we still want to keep the cv/tx modalities masked if they are not available for drugs --> set must-be masked modalities to 1 instead of those desired to 0
        modality_list = modality.split('+')
        head_or_tail_masks = head_or_tail_masks_base.clone()
        modality_col_list = []
        for modality in modality_list:
            modality_col_list.extend(MODALITY2NUMBER_LIST[modality])
        modality_must_mask_col_list = list(set(range(NUM_MODALITIES)) - set(modality_col_list))
        head_or_tail_masks[:, modality_must_mask_col_list] = 1  # Mask out the unavailable ones, leave the others as it is originally in the masks
        head_or_tail_masks = head_or_tail_masks.bool()
    return head_or_tail_masks


def get_evaluate_masks(head_masks_base, tail_masks_base, eval_type, finetune_mode, device):
    """
    Get modality masks for different evaluation settings. 
    Current modalities: [str, kg, cv, (bs,) tx_*]
    """
    assert len(eval_type.split('_')) == 2
    head_eval_mask_type, tail_eval_mask_type = eval_type.split('_')
    
    if head_eval_mask_type == 'full':
        head_masks = get_full_evaluate_mask_for_finetune_mode(finetune_mode, head_masks_base)
    else:
        head_masks = get_modality_evaluate_mask(head_masks_base, head_eval_mask_type)
        
    if tail_eval_mask_type == 'full':
        tail_masks = get_full_evaluate_mask_for_finetune_mode(finetune_mode, tail_masks_base)
    else:
        tail_masks = get_modality_evaluate_mask(tail_masks_base, tail_eval_mask_type)
        
    return head_masks.to(device), tail_masks.to(device)


@torch.no_grad()
def save_embeds(encoder, train_drugs, val_drugs, masks, collator, save_dir, device, raw_encoder_output=False):
    encoder.eval()


    train_outputs = {}
    val_outputs = {}
    
    for index, subset_mask in enumerate(torch.eye(masks.shape[1])):
        # only str, kg, cv, tx_mcf7, tx_pc3, tx_vcap
        if index not in sum([MODALITY2NUMBER_LIST[mod] for mod in 
                             ['str', 'kg', 'cv', 'tx_mcf7', 'tx_pc3', 'tx_vcap']], start=[]):
        # {0, 1, 2, 13, 15, 17}:
            continue
        val_valid_drugs = val_drugs[masks[val_drugs, :][:, index] == 0]  # Get val drugs that have the required modality
        val_valid_drugs, val_valid_data = to_device(collator([val_valid_drugs]), device)
        val_valid_mols, val_valid_kgs, val_valid_cvs, val_valid_tx_all_cell_lines = val_valid_data
        other_tabular_mod_data = {}
        val_valid_masks = to_device(~(subset_mask.repeat(val_valid_drugs.shape[0], 1).bool()), device)
        
        val_valid_embeds = encoder(val_valid_drugs, val_valid_masks, val_valid_mols, val_valid_kgs, val_valid_cvs, val_valid_tx_all_cell_lines, raw_encoder_output=raw_encoder_output, **other_tabular_mod_data).cpu()
        
        if save_dir is not None:
            torch.save({'drugs':val_valid_drugs.detach().cpu().numpy(), 'embeds':val_valid_embeds, 'masks':masks[val_valid_drugs.detach().cpu().numpy()]}, save_dir + f'/val_embeds_{index}.pt')
        
        val_outputs[str(index)] = {}
        val_outputs[str(index)]['embeds'] = val_valid_embeds
        val_outputs[str(index)]['drugs'] = val_valid_drugs.detach().cpu().numpy()

        train_valid_drugs = train_drugs[masks[train_drugs, :][:, index] == 0]  # Get train drugs that have the required modality
        train_valid_drugs, train_valid_data = to_device(collator([train_valid_drugs]), device)
        train_valid_mols, train_valid_kgs, train_valid_cvs, train_valid_tx_all_cell_lines = train_valid_data
        other_tabular_mod_data = {}
        train_valid_masks = to_device(~(subset_mask.repeat(train_valid_drugs.shape[0], 1).bool()), device)
        
        train_valid_embeds = encoder(train_valid_drugs, train_valid_masks, train_valid_mols, train_valid_kgs, train_valid_cvs, train_valid_tx_all_cell_lines, raw_encoder_output=raw_encoder_output, **other_tabular_mod_data).cpu()
        
        if save_dir is not None:
            torch.save({'drugs':train_valid_drugs.detach().cpu().numpy(), 'embeds':train_valid_embeds, 'masks':masks[train_valid_drugs.detach().cpu().numpy()]}, save_dir + f'/train_embeds_{index}.pt')

        train_outputs[str(index)] = {}
        train_outputs[str(index)]['embeds'] = train_valid_embeds
        train_outputs[str(index)]['drugs'] = train_valid_drugs.detach().cpu().numpy()
    

    return train_outputs, val_outputs


####
# Plotting utils
####
@torch.no_grad()
def draw_umap_plot(embeds, encoder, drug_ids, drug_loader, masks, collator, plotname, wandb, device, logger, epoch, output_dir=None, other_labels=None, raw_encoder_output=False):
    str_mod_ind = MODALITY2NUMBER_LIST["str"][0]
    kg_mod_ind = MODALITY2NUMBER_LIST["kg"][0]
    cv_mod_ind = MODALITY2NUMBER_LIST["cv"][0]
    tx_mcf7_mod_ind = MODALITY2NUMBER_LIST["tx_mcf7"][0]
    tx_pc3_mod_ind = MODALITY2NUMBER_LIST["tx_pc3"][0]
    tx_vcap_mod_ind = MODALITY2NUMBER_LIST["tx_vcap"][0]
    
    if embeds is not None:
        str_embedding = embeds[str(str_mod_ind)]
        kg_embedding = embeds[str(kg_mod_ind)]
        cv_embedding = embeds[str(cv_mod_ind)]
        tx_mcf7_embedding = embeds[str(tx_mcf7_mod_ind)]
        tx_pc3_embedding = embeds[str(tx_pc3_mod_ind)]
        tx_vcap_embedding = embeds[str(tx_vcap_mod_ind)]
        if drug_ids is not None:
            drug_ids = drug_ids[str(str_mod_ind)] + drug_ids[str(kg_mod_ind)] + drug_ids[str(cv_mod_ind)] + drug_ids[str(tx_mcf7_mod_ind)] + drug_ids[str(tx_pc3_mod_ind)] + drug_ids[str(tx_vcap_mod_ind)]
    
    elif drug_ids is not None:
        if isinstance(drug_ids, list):
            drug_ids = np.array(drug_ids)
        
        masks = masks[drug_ids]
        _, modality_data = to_device(collator([drug_ids]), device)
        mols, kgs, cvs, tx_all_cell_lines = modality_data
        other_tabular_mod_data = {}
        assert drug_ids.shape[0] == masks.shape[0]
        mod_avail = 1 - masks
        
        keep_str_mask = torch.ones(masks.shape[1])
        keep_str_mask[str_mod_ind] = 0
        keep_str_mask = to_device(keep_str_mask.repeat(mod_avail.sum(axis=0)[str_mod_ind], 1).bool(), device)
        keep_kg_mask = torch.ones(masks.shape[1])
        keep_kg_mask[kg_mod_ind] = 0
        keep_kg_mask = to_device(keep_kg_mask.repeat(mod_avail.sum(axis=0)[kg_mod_ind], 1).bool(), device)
        keep_cv_mask = torch.ones(masks.shape[1])
        keep_cv_mask[cv_mod_ind] = 0
        keep_cv_mask = to_device(keep_cv_mask.repeat(mod_avail.sum(axis=0)[cv_mod_ind], 1).bool(), device)
        keep_tx_mcf7_mask = torch.ones(masks.shape[1])
        keep_tx_mcf7_mask[tx_mcf7_mod_ind] = 0
        keep_tx_mcf7_mask = to_device(keep_tx_mcf7_mask.repeat(mod_avail.sum(axis=0)[tx_mcf7_mod_ind], 1).bool(), device)
        keep_tx_pc3_mask = torch.ones(masks.shape[1])
        keep_tx_pc3_mask[tx_pc3_mod_ind] = 0
        keep_tx_pc3_mask = to_device(keep_tx_pc3_mask.repeat(mod_avail.sum(axis=0)[tx_pc3_mod_ind], 1).bool(), device)
        keep_tx_vcap_mask = torch.ones(masks.shape[1])
        keep_tx_vcap_mask[tx_vcap_mod_ind] = 0
        keep_tx_vcap_mask = to_device(keep_tx_vcap_mask.repeat(mod_avail.sum(axis=0)[tx_vcap_mod_ind], 1).bool(), device)

        encoder.eval()
        drug_ids = torch.from_numpy(drug_ids)
        str_indices_bool = torch.from_numpy(masks[:, str_mod_ind] == 0)
        str_embedding = encoder(
            drug_ids[str_indices_bool].to(device), 
            keep_str_mask, 
            mols[str_indices_bool], 
            kgs, 
            cvs[str_indices_bool], 
            {cell_line: {k: v[str_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, 
            raw_encoder_output=raw_encoder_output, 
            **other_tabular_mod_data
        ).cpu().numpy()
        kg_indices_bool = torch.from_numpy(masks[:, kg_mod_ind] == 0)
        kg_embedding = encoder(
            drug_ids[kg_indices_bool].to(device), 
            keep_kg_mask, 
            mols[kg_indices_bool], 
            kgs,
            cvs[kg_indices_bool],
            {cell_line: {k: v[kg_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, 
            raw_encoder_output=raw_encoder_output,
            **other_tabular_mod_data
        ).cpu().numpy()
        cv_indices_bool = torch.from_numpy(masks[:, cv_mod_ind] == 0)
        cv_embedding = encoder(
            drug_ids[cv_indices_bool].to(device), 
            keep_cv_mask,
            mols[cv_indices_bool],
            kgs,
            cvs[cv_indices_bool],
            {cell_line: {k: v[cv_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, 
            raw_encoder_output=raw_encoder_output,
            **other_tabular_mod_data
        ).cpu().numpy()
        tx_mcf7_indices_bool = torch.from_numpy(masks[:, tx_mcf7_mod_ind] == 0)
        tx_mcf7_embedding = encoder(
            drug_ids[tx_mcf7_indices_bool].to(device), 
            keep_tx_mcf7_mask,
            mols[tx_mcf7_indices_bool],
            kgs,
            cvs[tx_mcf7_indices_bool],
            {cell_line: {k: v[tx_mcf7_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, 
            raw_encoder_output=raw_encoder_output,
            **other_tabular_mod_data
        ).cpu().numpy()
        tx_pc3_indices_bool = torch.from_numpy(masks[:, tx_pc3_mod_ind] == 0)
        tx_pc3_embedding = encoder(
            drug_ids[tx_pc3_indices_bool].to(device),
            keep_tx_pc3_mask,
            mols[tx_pc3_indices_bool],
            kgs,
            cvs[tx_pc3_indices_bool],
            {cell_line: {k: v[tx_pc3_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, 
            raw_encoder_output=raw_encoder_output,
            **other_tabular_mod_data
        ).cpu().numpy()
        tx_vcap_indices_bool = torch.from_numpy(masks[:, tx_vcap_mod_ind] == 0)
        tx_vcap_embedding = encoder(
            drug_ids[tx_vcap_indices_bool].to(device),
            keep_tx_vcap_mask,
            mols[tx_vcap_indices_bool],
            kgs,
            cvs[tx_vcap_indices_bool],
            {cell_line: {k: v[tx_vcap_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, 
            raw_encoder_output=raw_encoder_output,
            **other_tabular_mod_data
        ).cpu().numpy()

        drug_ids = drug_ids[str_indices_bool].tolist() + drug_ids[kg_indices_bool].tolist() + drug_ids[cv_indices_bool].tolist() + drug_ids[tx_mcf7_indices_bool].tolist() + drug_ids[tx_pc3_indices_bool].tolist() + drug_ids[tx_vcap_indices_bool].tolist()
    
    elif drug_loader is not None:
        raise NotImplementedError

            
                
            
            

    full_embeddings = np.concatenate([str_embedding, kg_embedding, cv_embedding] + [tx_mcf7_embedding, tx_pc3_embedding, tx_vcap_embedding])
    modality_labels = ['str'] * str_embedding.shape[0] + ['kg'] * kg_embedding.shape[0] + ['cv'] * cv_embedding.shape[0] + ['tx_mcf7'] * tx_mcf7_embedding.shape[0] + ['tx_pc3'] * tx_pc3_embedding.shape[0] + ['tx_vcap'] * tx_vcap_embedding.shape[0]
    if other_labels is not None:
        other_labels = other_labels[masks[:, str_mod_ind] == 0].tolist() + other_labels[masks[:, kg_mod_ind] == 0].tolist() + other_labels[masks[:, cv_mod_ind] == 0].tolist() + other_labels[masks[:, tx_mcf7_mod_ind] == 0].tolist() + other_labels[masks[:, tx_pc3_mod_ind] == 0].tolist() + other_labels[masks[:, tx_vcap_mod_ind] == 0].tolist()
    
    logger.info("=> Fitting UMAP")
    
    # NOTE: 'cosine' can lead to Segmentation Fault
    dat = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(full_embeddings)  # Use 'euclidean' if 'cosine' leads to Segmentation Fault
    
    if other_labels is None:
        if drug_ids is not None:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=modality_labels, hover_data={'drug_id':drug_ids})
        else:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=modality_labels)
    else:
        if drug_ids is not None:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=other_labels, hover_data={'drug_id':drug_ids})
        else:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=other_labels)

    if wandb is not None:
        wandb.log({plotname:fig}, step=epoch)
    if output_dir is not None:
        fig.write_image(output_dir+plotname)

