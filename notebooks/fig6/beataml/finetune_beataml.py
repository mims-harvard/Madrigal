import os, argparse
from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from madrigal.utils import DATA_DIR, BASE_DIR
from beataml_data import prepare_data, BeatAMLCollator, get_rdkit_descriptors, OUTPUT_DIR
from beataml_args import parse_args

import torch
import torch.nn as nn
from torch.nn import functional as F

from madrigal.models.models import BilinearDDIScorer, Symmetric, MLPAdaptor
from madrigal.utils import to_device, set_seed

# Madrigal runs trained on all DrugBank data (one per seed); their saved drug embeddings are used as frozen inputs.
CHECKPOINTS = [
    "all_train_seed42",
    "all_train_seed99",
    "all_train_seed1",
    "all_train_seed2",
    "all_train_seed0",
]
EMBEDDINGS_FILE = "drugbank_all_metadata_drug_embeddings_full.pt"


class BeatAMLPredictor(nn.Module):
    def __init__(self, encoder, feat_dim=128, drug_combo_dim=64, normalize=False, norm_expr_dim=150, clin_attr_dim=21, mut_profile_dim=30, scorer_hidden_dims=(128,), scorer_dropout=0.2, scorer_norm="ln", use_clin_feature=True, use_mut_profile=True, use_rnaseq_profile=True,):
        super(BeatAMLPredictor, self).__init__()
        self.drug_encoder = encoder  # None when using precomputed frozen embeddings
        self.drug_embeddings = None  # [n_drugs, embed_dim] frozen MADRIGAL embeddings, indexed by global drug id
        self.embed_dim = feat_dim
        self.normalize = normalize
        self.drug_pair_decoder = BilinearDDIScorer(input_dim1 = self.embed_dim, input_dim2 = self.embed_dim, output_dim = drug_combo_dim)
        nn.utils.parametrize.register_parametrization(self.drug_pair_decoder, 'weight', Symmetric())

        self.use_clin_feature = use_clin_feature
        self.use_mut_profile = use_mut_profile
        self.use_rnaseq_profile = use_rnaseq_profile
        
        in_dim = drug_combo_dim + (norm_expr_dim if use_rnaseq_profile else 0) + (clin_attr_dim if use_clin_feature else 0) + (mut_profile_dim if use_mut_profile else 0)
        self.scorer = MLPAdaptor(in_dim=in_dim, hidden_dims=scorer_hidden_dims, output_dim=1, 
                                 p=scorer_dropout, norm=scorer_norm, actn="relu")
        
    def forward(self, batch_head, batch_tail, batch_head_mod_masks, batch_tail_mod_masks, batch_kg, 
                head_indices, tail_indices, X_norm_expr, X_clinical_attrs, X_mut_profiles):
        head_unique_drugs = batch_head['drugs']
        head_mol_strs = batch_head['strs']
        head_cv = batch_head['cv']
        head_tx_all_cell_lines = batch_head['tx']
        head_masks = batch_head_mod_masks
        
        tail_unique_drugs = batch_tail['drugs']
        tail_mol_strs = batch_tail['strs']
        tail_cv = batch_tail['cv']
        tail_tx_all_cell_lines = batch_tail['tx']
        tail_masks = batch_tail_mod_masks
        
        if self.drug_embeddings is not None:  # frozen MADRIGAL embeddings -> lookup (exact; encoder is frozen)
            z_head = self.drug_embeddings[head_unique_drugs]
            z_tail = self.drug_embeddings[tail_unique_drugs]
        else:
            z_head = self.drug_encoder(head_unique_drugs, head_masks, head_mol_strs, batch_kg, head_cv, head_tx_all_cell_lines)
            z_tail = self.drug_encoder(tail_unique_drugs, tail_masks, tail_mol_strs, batch_kg, tail_cv, tail_tx_all_cell_lines)
        if self.normalize:
            z_head = F.normalize(z_head)  # same as z_head / torch.norm(z_head, dim=1, keepdim=True)
            z_tail = F.normalize(z_tail)
            
        drug_pair_features = self.drug_pair_decoder(z_head, z_tail)  # [drug_combo_feature_dim, num_heads, num_tails]
        X_drug_pair_features = drug_pair_features[:, head_indices, tail_indices].T
        
        X = X_drug_pair_features
        if self.use_rnaseq_profile:
            X = torch.cat([X, X_norm_expr], dim=1)
        if self.use_clin_feature:
            X = torch.cat([X, X_clinical_attrs], dim=1)
        if self.use_mut_profile:
            X = torch.cat([X, X_mut_profiles], dim=1)
        
        out = self.scorer(X)
        
        return out
        

class BeatAMLBaseline(nn.Module):
    def __init__(self, feat_dim=195, drug_combo_dim=64, normalize=False, norm_expr_dim=150, clin_attr_dim=21, mut_profile_dim=30, 
                 scorer_hidden_dims=(128,), scorer_dropout=0.2, scorer_norm="ln", use_clin_feature=True, use_mut_profile=True, use_rnaseq_profile=True):
        super(BeatAMLBaseline, self).__init__()
        self.drug_combo_dim = drug_combo_dim
        if drug_combo_dim == 0:
            assert use_clin_feature or use_mut_profile or use_rnaseq_profile, "At least one of clinical feature, mutation profile, or RNA-seq profile should be used when drug features are not used"
        self.embed_dim = feat_dim
        self.normalize = normalize
        if self.drug_combo_dim > 0:
            self.drug_pair_decoder = BilinearDDIScorer(input_dim1 = self.embed_dim, input_dim2 = self.embed_dim, output_dim = drug_combo_dim)
            nn.utils.parametrize.register_parametrization(self.drug_pair_decoder, 'weight', Symmetric())

        self.use_clin_feature = use_clin_feature
        self.use_mut_profile = use_mut_profile
        self.use_rnaseq_profile = use_rnaseq_profile
        
        in_dim = drug_combo_dim + (norm_expr_dim if use_rnaseq_profile else 0) + (clin_attr_dim if use_clin_feature else 0) + (mut_profile_dim if use_mut_profile else 0)
        self.scorer = MLPAdaptor(in_dim=in_dim, hidden_dims=scorer_hidden_dims, output_dim=1, 
                                 p=scorer_dropout, norm=scorer_norm, actn="relu")
        
    def forward(self, X_drug_descriptors_1, X_drug_descriptors_2, head_indices, tail_indices, 
                X_norm_expr, X_clinical_attrs, X_mut_profiles):
        if self.drug_combo_dim > 0:
            z_head = X_drug_descriptors_1
            z_tail = X_drug_descriptors_2
            if self.normalize:
                z_head = F.normalize(z_head)  # same as z_head / torch.norm(z_head, dim=1, keepdim=True)
                z_tail = F.normalize(z_tail)
                
            drug_pair_features = self.drug_pair_decoder(z_head, z_tail)  # [drug_combo_feature_dim, num_heads, num_tails]
            X_drug_pair_features = drug_pair_features[:, head_indices, tail_indices].T
        else:
            X_drug_pair_features = torch.Tensor([]).to(device=X_norm_expr.device)
        
        X = X_drug_pair_features
        if self.use_rnaseq_profile:
            X = torch.cat([X, X_norm_expr], dim=1)
        if self.use_clin_feature:
            X = torch.cat([X, X_clinical_attrs], dim=1)
        if self.use_mut_profile:
            X = torch.cat([X, X_mut_profiles], dim=1)
        
        out = self.scorer(X)
        
        return out
        

def beataml_evaluate(all_preds, all_labels, all_patients, all_drug_pairs):
    micro_auroc = roc_auc_score(all_labels, all_preds)

    patient_aurocs = []
    drug_pair_aurocs = []

    for patient_id in np.unique(all_patients):
        indicator = all_patients == patient_id
        preds_patient = all_preds[indicator]
        labels_patient = all_labels[indicator]

        if indicator.sum() <= 1:
            continue
        elif labels_patient.sum() < 1:
            continue
        elif (1-labels_patient).sum() < 1:
            continue
        patient_aurocs.append(roc_auc_score(labels_patient, preds_patient))

    for drug_pair in np.unique(all_drug_pairs):
        indicator = all_drug_pairs == drug_pair
        preds_drug_pair = all_preds[indicator]
        labels_drug_pair = all_labels[indicator]

        if indicator.sum() <= 1:
            continue
        elif labels_drug_pair.sum() < 1:
            continue
        elif (1-labels_drug_pair).sum() < 1:
            continue
        drug_pair_aurocs.append(roc_auc_score(labels_drug_pair, preds_drug_pair))

    patient_macro_auroc = np.mean(patient_aurocs)
    drug_pair_macro_auroc = np.mean(drug_pair_aurocs)
    
    return micro_auroc, patient_macro_auroc, drug_pair_macro_auroc


def run_batch_madrigal(batch, model, device):
    batch_head = to_device(batch['head'], device)  # dict
    batch_tail = to_device(batch['tail'], device)
    batch_kg = to_device(batch['kg'], device)
    head_masks_base = to_device(batch['head']['masks'], device)
    tail_masks_base = to_device(batch['tail']['masks'], device)

    head_indices = to_device(batch["edge_indices"]["head"], device)
    tail_indices = to_device(batch["edge_indices"]["tail"], device)
    norm_exprs = to_device(batch["edge_indices"]["norm_expr"], device)
    clinical_attrs = to_device(batch["edge_indices"]["clinical_attrs"], device)
    mut_profiles = to_device(batch["edge_indices"]["mut_profiles"], device)

    preds = model(batch_head, batch_tail, head_masks_base, tail_masks_base, batch_kg, head_indices, tail_indices, norm_exprs, clinical_attrs, mut_profiles)

    return preds.flatten(), len(head_indices)


def run_batch_rdkit(rdkit_descriptors_torch, batch, model, device):
    batch_drug_descriptors_head = to_device(rdkit_descriptors_torch[batch["head"]["drugs"]], device)
    batch_drug_descriptors_tail = to_device(rdkit_descriptors_torch[batch["tail"]["drugs"]], device)

    head_indices = to_device(batch["edge_indices"]["head"], device)
    tail_indices = to_device(batch["edge_indices"]["tail"], device)
    norm_exprs = to_device(batch["edge_indices"]["norm_expr"], device)
    clinical_attrs = to_device(batch["edge_indices"]["clinical_attrs"], device)
    mut_profiles = to_device(batch["edge_indices"]["mut_profiles"], device)

    preds = model(batch_drug_descriptors_head, batch_drug_descriptors_tail, head_indices, tail_indices, norm_exprs, clinical_attrs, mut_profiles)

    return preds.flatten(), len(head_indices)


def run_batch_no_drug(batch, model, device):
    head_indices = to_device(batch["edge_indices"]["head"], device)
    tail_indices = to_device(batch["edge_indices"]["tail"], device)
    norm_exprs = to_device(batch["edge_indices"]["norm_expr"], device)
    clinical_attrs = to_device(batch["edge_indices"]["clinical_attrs"], device)
    mut_profiles = to_device(batch["edge_indices"]["mut_profiles"], device)

    preds = model(None, None, head_indices, tail_indices, norm_exprs, clinical_attrs, mut_profiles)

    return preds.flatten(), len(head_indices)


def run_finetune(args, prefix, checkpoint_name, collator: BeatAMLCollator, train_loader, val_loader, test_loader, model_type):
    if model_type == "rdkit":
        rdkit_descriptors = get_rdkit_descriptors()
        rdkit_descriptors_torch = torch.from_numpy(rdkit_descriptors).float()
    
    data_source = 'DrugBank'
    split_method = 'split_by_pairs'
    
    checkpoint_dir = BASE_DIR + f'model_output/{data_source}/{split_method}/{checkpoint_name}/'

    # GPU when available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Frozen MADRIGAL drug embeddings for this pretraining run (all-metadata, 21842 drugs).
    drug_embeddings = torch.load(checkpoint_dir + EMBEDDINGS_FILE, map_location="cpu").float()
    n_drugs = collator.drug_metadata_ft.shape[0]
    assert drug_embeddings.shape[0] >= n_drugs, (
        f"{EMBEDDINGS_FILE} has {drug_embeddings.shape[0]} drugs but BeatAML needs {n_drugs}")
    drug_embeddings = drug_embeddings[:n_drugs].to(device)
    encoder = None  # not needed: the encoder is frozen and its outputs are precomputed above
    feat_dim = drug_embeddings.shape[1]
    print(f"loaded frozen drug embeddings {tuple(drug_embeddings.shape)} from {checkpoint_name}")

    if model_type == "madrigal":
        model = BeatAMLPredictor(
            encoder, feat_dim=feat_dim, drug_combo_dim=args.drug_combo_dim, normalize=False, 
            norm_expr_dim=collator.X_norm_expr.shape[1], 
            clin_attr_dim=collator.X_clinical_attrs.shape[1], 
            mut_profile_dim=collator.X_mut.shape[1], 
            scorer_hidden_dims=args.scorer_hidden_dims, scorer_dropout=args.scorer_dropout, scorer_norm="ln",
            use_clin_feature=args.use_clin_feature, use_mut_profile=args.use_mut_profile, use_rnaseq_profile=args.use_rnaseq_profile,
        )
    elif model_type == "no_patient":
        model = BeatAMLPredictor(
            encoder, feat_dim=feat_dim, drug_combo_dim=args.drug_combo_dim, normalize=False, 
            norm_expr_dim=collator.X_norm_expr.shape[1], 
            clin_attr_dim=collator.X_clinical_attrs.shape[1], 
            mut_profile_dim=collator.X_mut.shape[1], 
            scorer_hidden_dims=args.scorer_hidden_dims, scorer_dropout=args.scorer_dropout, scorer_norm="ln",
            use_clin_feature=False, use_mut_profile=False, use_rnaseq_profile=False,
        )
    elif model_type == "no_patient_clin":
        model = BeatAMLPredictor(
            encoder, feat_dim=feat_dim, drug_combo_dim=args.drug_combo_dim, normalize=False, 
            norm_expr_dim=collator.X_norm_expr.shape[1], 
            clin_attr_dim=collator.X_clinical_attrs.shape[1], 
            mut_profile_dim=collator.X_mut.shape[1], 
            scorer_hidden_dims=args.scorer_hidden_dims, scorer_dropout=args.scorer_dropout, scorer_norm="ln",
            use_clin_feature=False, use_mut_profile=args.use_mut_profile, use_rnaseq_profile=args.use_rnaseq_profile,
        )
    elif model_type == "no_patient_genomics":
        model = BeatAMLPredictor(
            encoder, feat_dim=feat_dim, drug_combo_dim=args.drug_combo_dim, normalize=False, 
            norm_expr_dim=collator.X_norm_expr.shape[1], 
            clin_attr_dim=collator.X_clinical_attrs.shape[1], 
            mut_profile_dim=collator.X_mut.shape[1], 
            scorer_hidden_dims=args.scorer_hidden_dims, scorer_dropout=args.scorer_dropout, scorer_norm="ln",
            use_clin_feature=args.use_clin_feature, use_mut_profile=False, use_rnaseq_profile=False,
        )
    elif model_type == "rdkit":
        model = BeatAMLBaseline(
            feat_dim=rdkit_descriptors.shape[1], drug_combo_dim=args.drug_combo_dim, normalize=False, 
            norm_expr_dim=collator.X_norm_expr.shape[1], 
            clin_attr_dim=collator.X_clinical_attrs.shape[1], 
            mut_profile_dim=collator.X_mut.shape[1], 
            scorer_hidden_dims=args.scorer_hidden_dims, scorer_dropout=args.scorer_dropout, scorer_norm="ln",
            use_clin_feature=args.use_clin_feature, use_mut_profile=args.use_mut_profile, use_rnaseq_profile=args.use_rnaseq_profile,
        )
    elif model_type == "no_drug":
        model = BeatAMLBaseline(
            feat_dim=0, drug_combo_dim=0, normalize=False, 
            norm_expr_dim=collator.X_norm_expr.shape[1], 
            clin_attr_dim=collator.X_clinical_attrs.shape[1], 
            mut_profile_dim=collator.X_mut.shape[1], 
            scorer_hidden_dims=args.scorer_hidden_dims, scorer_dropout=args.scorer_dropout, scorer_norm="ln",
            use_clin_feature=args.use_clin_feature, use_mut_profile=args.use_mut_profile, use_rnaseq_profile=args.use_rnaseq_profile,
        )
    model.to(device)
    if isinstance(model, BeatAMLPredictor):
        model.drug_embeddings = drug_embeddings  # frozen; not a Parameter, so never trained/saved

    for name, param in model.named_parameters():
        if args.freeze_level == "encoder_only":  # freeze encoders
            if name.split(".")[1].endswith("_encoder"):
                param.requires_grad = False
        elif args.freeze_level == "madrigal":  # freeze all but decoder & scorer
            if (name.split(".")[0] != "drug_pair_decoder") and (name.split(".")[0] != "scorer"):
                param.requires_grad = False
        else:  # fine-tune all
            pass

    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, )

    # The loaders are full-batch and the collator is deterministic, so each split's batch is built once.
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    all_train_metrics = defaultdict(list)
    all_val_metrics = defaultdict(list)
    num_epochs = 200
    best_metric = 0
    for epoch in range(num_epochs):
        print(epoch)
        model.train()

        all_preds = []
        all_labels = []
        all_drug_pairs = []
        all_patients = []
        total_loss = 0
        total_sample = 0

        # batch train
        for batch in [train_batch]:
            optim.zero_grad()
            if model_type in {"madrigal", "no_patient", "no_patient_clin", "no_patient_genomics"}:
                preds, num_samples = run_batch_madrigal(batch, model, device)
            elif model_type == "rdkit":
                preds, num_samples = run_batch_rdkit(rdkit_descriptors_torch, batch, model, device)
            elif model_type == "no_drug":
                preds, num_samples = run_batch_no_drug(batch, model, device)
            else:
                raise ValueError("Invalid model type")
            
            labels = to_device(batch["edge_indices"]["label"], device)
            loss = loss_fn(preds, labels)
            loss.backward()

            optim.step()
            
            total_loss += loss.item() * num_samples
            total_sample += num_samples
            
            patients = batch["edge_indices"]["patients"]
            drug_pairs = batch["edge_indices"]["drug_pairs"]
            all_preds.extend(torch.sigmoid(preds.detach()).cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_drug_pairs.extend(drug_pairs)
            all_patients.extend(patients)
        
        # report train metric over the whole epoch
        print(f"\tTrain loss: {total_loss/total_sample:.4f}")
        
        all_train_metrics["Loss"].append(total_loss/total_sample)
        
        # evaluate on val set every batch
        # NOTE: assume val set is full batch
        model.eval()
        with torch.no_grad():
            if model_type in {"madrigal", "no_patient", "no_patient_clin", "no_patient_genomics"}:
                val_preds, _ = run_batch_madrigal(val_batch, model, device)
            elif model_type == "rdkit":
                val_preds, _ = run_batch_rdkit(rdkit_descriptors_torch, val_batch, model, device)
            elif model_type == "no_drug":
                val_preds, _ = run_batch_no_drug(val_batch, model, device)
            val_labels = to_device(val_batch["edge_indices"]["label"], device)
            val_loss = loss_fn(val_preds, val_labels).item()
            
        val_patients = val_batch["edge_indices"]["patients"]
        val_drug_pairs = val_batch["edge_indices"]["drug_pairs"]
        micro_auroc, patient_macro_auroc, drug_pair_macro_auroc = beataml_evaluate(
            torch.sigmoid(val_preds.detach().cpu()).numpy(), val_labels.detach().cpu().numpy(), np.array(val_patients), np.array(val_drug_pairs)
        )
        print(f"\tVal loss: {val_loss:.4f}")
        print(f"\tVal micro AUROC: {micro_auroc:.4f}")
        print(f"\tVal patient-macro AUROC: {patient_macro_auroc:.4f}")
        print(f"\tVal drug pair-macro AUROC: {drug_pair_macro_auroc:.4f}")
        all_val_metrics["Loss"].append(val_loss)
        all_val_metrics["Micro AUROC"].append(micro_auroc)
        all_val_metrics["Patient-macro AUROC"].append(patient_macro_auroc)
        all_val_metrics["Drug pair-macro AUROC"].append(drug_pair_macro_auroc)
        
        if patient_macro_auroc > best_metric:
            best_metric = patient_macro_auroc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_micro_auroc": micro_auroc,
                "val_patient_macro_auroc": patient_macro_auroc,
                "val_drug_pair_macro_auroc": drug_pair_macro_auroc,
            }, OUTPUT_DIR+f"{prefix}/best_model_{model_type}_{checkpoint_name}_{args.seed}.pt")
    
    model.load_state_dict(torch.load(OUTPUT_DIR+f"{prefix}/best_model_{model_type}_{checkpoint_name}_{args.seed}.pt", map_location="cpu")["model_state_dict"])

    model.eval()
    with torch.no_grad():
        if model_type in {"madrigal", "no_patient", "no_patient_clin", "no_patient_genomics"}:
            test_preds, _ = run_batch_madrigal(test_batch, model, device)
        elif model_type == "rdkit":
            test_preds, _ = run_batch_rdkit(rdkit_descriptors_torch, test_batch, model, device)
        elif model_type == "no_drug":
            test_preds, _ = run_batch_no_drug(test_batch, model, device)
        else:
            raise ValueError("Invalid model type")
        test_labels = to_device(test_batch["edge_indices"]["label"], device)
        test_loss = loss_fn(test_preds, test_labels).item()

    test_patients = test_batch["edge_indices"]["patients"]
    test_drug_pairs = test_batch["edge_indices"]["drug_pairs"]
    micro_auroc, patient_macro_auroc, drug_pair_macro_auroc = beataml_evaluate(
        torch.sigmoid(test_preds.detach().cpu()).numpy(), test_labels.detach().cpu().numpy(), np.array(test_patients), np.array(test_drug_pairs)
    )
    print(f"\tTest micro AUROC: {micro_auroc:.4f}")
    print(f"\tTest patient-macro AUROC: {patient_macro_auroc:.4f}")
    print(f"\tTest drug pair-macro AUROC: {drug_pair_macro_auroc:.4f}")
    
    torch.save({
        "all_train_metrics": all_train_metrics,
        "all_val_metrics": all_val_metrics,
        "test_loss": test_loss,
        "test_micro_auroc": micro_auroc,
        "test_patient_macro_auroc": patient_macro_auroc,
        "test_drug_pair_macro_auroc": drug_pair_macro_auroc,
    }, OUTPUT_DIR+f"{prefix}/metrics_{model_type}_{checkpoint_name}_{args.seed}.pkl")
    
    return micro_auroc, patient_macro_auroc, drug_pair_macro_auroc

    
def main():
    args = parse_args()
    set_seed(args.seed)
    
    collator, train_loader, val_loader, test_loader = prepare_data(args.split_method)
    
    prefix = f"{args.split_method}/BeatAML_ft_freeze_{args.freeze_level}_combo_{str(args.drug_combo_dim)}_dims_{'_'.join([str(dim) for dim in args.scorer_hidden_dims])}_dropout_{str(args.scorer_dropout)}_mut_{str(args.use_mut_profile)}_lr_{str(args.lr)}"
    os.makedirs(OUTPUT_DIR+f"{prefix}", exist_ok=True)
    
    all_metrics = defaultdict(list)
    for checkpoint in CHECKPOINTS:  # DrugBank (trained on all data), one per pretraining seed
        print(f"Finetuning on {checkpoint}")
        for model_type in ["madrigal", "rdkit", "no_drug", "no_patient", "no_patient_clin", "no_patient_genomics"]:
            micro_auroc, patient_macro_auroc, drug_pair_macro_auroc = run_finetune(args, prefix, checkpoint, collator, train_loader, val_loader, test_loader, model_type)
            all_metrics[model_type].append((micro_auroc, patient_macro_auroc, drug_pair_macro_auroc))
    
    all_metrics_madrigal = np.array(all_metrics["madrigal"])
    all_metrics_rdkit = np.array(all_metrics["rdkit"])
    print("\nMadrigal:")
    print(f"Test micro AUROC (averaged): {all_metrics_madrigal[:, 0].mean():.4f}")
    print(f"Test patient-macro AUROC (averaged): {all_metrics_madrigal[:, 1].mean():.4f}")
    print(f"Test drug pair-macro AUROC (averaged): {all_metrics_madrigal[:, 2].mean():.4f}")
    print("\nRDKit:")
    print(f"Test micro AUROC (averaged): {all_metrics_rdkit[:, 0].mean():.4f}")
    print(f"Test patient-macro AUROC (averaged): {all_metrics_rdkit[:, 1].mean():.4f}")
    print(f"Test drug pair-macro AUROC (averaged): {all_metrics_rdkit[:, 2].mean():.4f}")
    

if __name__ == "__main__":
    main()

