import math
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Dict, Any

from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score, 
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)


def fmax_score(ys: np.ndarray, preds: np.ndarray, beta = 1.0, pos_label = 1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    """
    precision, recall, thresholds = precision_recall_curve(y_true = ys, probas_pred = preds, pos_label = pos_label)
    numerator = (1 + beta**2) * (precision * recall)
    denominator = ((beta**2 * precision) + recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
    return np.nanmax(fbeta), thresholds[np.argmax(fbeta)]


def precision_recall_at_k(y: np.ndarray, preds: np.ndarray, k: int, names: np.ndarray = None):
    """ Calculate recall@k, precision@k, and AP@k for binary classification.
    """
    assert preds.shape == y.shape
    assert k > 0
    
    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    if names is not None:
        sorted_names = names[sorted_indices]
    else: sorted_names = None

    # Get the scores of the k highest predictions
    topk_preds = sorted_preds[:k]
    topk_y = sorted_y[:k]
    
    # Calculate the recall@k and precision@k
    recall_k = np.sum(topk_y, axis=-1) / np.sum(y, axis=-1)
    precision_k = np.sum(topk_y, axis=-1) / k
    
    # Calculate the AP@k
    ap_k = average_precision_score(topk_y, topk_preds)

    if k > preds.shape[-1]:
        recall_k = np.nan
        precision_k = np.nan
        ap_k = np.nan

    return recall_k, precision_k, ap_k, (sorted_y, sorted_preds, sorted_names)


def get_metrics_binary(preds, ys, k, verbose=False, logger=None, context=None):
    """ Wrapper for getting binary classification metrics. If k is a float, then get top k*100% of predictions.
    """
    if type(k) is float and k < 1:
        k = int(k * ys.shape[0])
    
    # Efficiently compute all these metrics together
    tn, fp, fn, tp = confusion_matrix(ys, np.round(preds)).ravel()
    specificity = np.divide(tn, (tn + fp))
    recall = np.divide(tp, (tp + fn))
    with np.errstate(divide='ignore', invalid='ignore'):
        npv = np.divide(tn, (tn + fn))
        precision = np.divide(tp, (tp + fp))
        f1 = np.divide(2 * precision * recall, (precision + recall))
    accuracy = (tp + tn) / (tn + fn + tp + fp)
    
    fmax, _ = fmax_score(ys, preds)
    recall_k, precision_k, ap_k, _ = precision_recall_at_k(ys, preds, k)
    auroc_score = roc_auc_score(ys, preds)
    auprc_score = average_precision_score(ys, preds)
    mcc = matthews_corrcoef(ys, np.round(preds))
    
    metrics_dict = {
        "fmax": fmax,
        "mcc": mcc,
        "auroc": auroc_score,
        "auprc": auprc_score,
        "npv": npv,
        "specificity": specificity,
        "f1": f1,
        f"recall@{k}": recall_k,
        f"precision@{k}": precision_k,
        f"ap@{k}": ap_k,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    
    if context is not None and context == "multiclass":
        # Cross-entropy loss
        cohen_kappa = cohen_kappa_score(ys, np.round(preds))
        metrics_dict["cohen_kappa"] = cohen_kappa

    if verbose:
        metrics_str = ', '.join([f'{key} = {value:.4f}' for key, value in metrics_dict.items()])
        if logger is None:
            print(metrics_str)
        else:
            logger.info(metrics_str)
    
    return tuple(metrics_dict.values()), tuple(metrics_dict.keys())


def get_metrics_for_indices(preds, ys, indices, k):
    valid_ys = ys[indices]
    valid_preds = preds[indices]
    metrics, metric_names = get_metrics_binary(valid_preds, valid_ys, k, verbose=False)
    return metrics, metric_names


def get_metrics(preds: np.ndarray, ys: np.ndarray, labels: np.ndarray, k: Union[int, float] = 50, task: str = 'multilabel', logger: Any = None, average: str = "macro", verbose: bool = True) -> Tuple[Dict[str, Union[np.ndarray, float]], Union[np.ndarray, int]]:
    """ Wrapper for getting classification metrics. 
    Binary & Multilabel: Accuracy, AUROC, AUPRC, precision, recall, recall@50, precision@50, ap@50, fmax, f1
    Multiclass: Accuracy, AUROC, AUPRC, precision, recall, recall@50, precision@50, ap@50, fmax, f1, Cohen's kappa
    
    Args:
        preds: Predictions from model for each sample
        ys: Indicators for each sample being positive or negative
        labels: Labels for each sample
        k: Number of top predictions to consider for recall@k, precision@k, and ap@k
        task: Type of classification task, one of 'binary', 'multilabel', or 'multiclass'
        logger: Logger to use for printing metrics
        average: Type of averaging to use for multilabel or multiclass metrics, one of 'macro', 'weighted', 'micro', or None
        verbose: Whether to print metrics or not
    
    Returns:
        metrics_dict: Dictionary of metrics
        pos_samples: Number of positive samples for each class
    """
    assert average is None or average in ["macro", "weighted", "micro"]
    
    if task == 'binary':
        metrics, metric_names = get_metrics_binary(preds, ys, k, verbose=verbose, logger=logger)
        pos_samples = sum(ys)
        
    else:
        # compute macro metrics. NOTE: We use sort + return_index & return_counts to efficiently get all indices of unique values, ref: https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(labels)
        # sorts records array so all unique elements are together 
        sorted_labels = labels[idx_sort]
        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_labels, return_counts=True, return_index=True)
        # splits the indices into separate arrays
        indices_grouped_list = np.split(idx_sort, idx_start[1:])
        
        pos_samples = np.array([sum(ys[indices]) for indices in indices_grouped_list])
        
        if average == 'micro':
            metrics, metric_names = get_metrics_binary(preds, ys, k, verbose=False)
            
        else:
            metrics_list = []
            for indices in indices_grouped_list:
                metrics, metric_names = get_metrics_for_indices(preds, ys, indices, k)
                metrics_list.append(metrics)
                
            if average == 'macro':
                metrics = np.array(metrics_list).mean(axis=0)
            elif average == 'weighted':
                metrics = np.array(metrics_list).T @ pos_samples / pos_samples.sum()
            elif average is None:  # output label-stratified scores without averaging
                metrics = np.array(metrics_list).T  # from (n_classes, n_metrics) to (n_metrics, n_classes)
                
        if verbose and average is not None:
            metrics_str = ', '.join([f'{key} = {value:.4f}' for key, value in zip(metric_names, metrics)])
            if logger is not None:
                logger.info(metrics_str)
            else:
                print(metrics_str)
        
    metrics_dict = dict(zip(metric_names, metrics))
    return metrics_dict, pos_samples
