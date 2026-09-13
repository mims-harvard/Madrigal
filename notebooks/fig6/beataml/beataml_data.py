import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import prince

from madrigal.utils import DATA_DIR, BASE_DIR
from madrigal.utils import CELL_LINES
from madrigal.data.data_utils import sample_kg_data

import torch
from torch.utils.data import DataLoader

PATH_BASE = BASE_DIR + "processed_data/"
RAW_DATA_DIR = BASE_DIR + "raw_data/beataml2.0_data/"
OUTPUT_DIR = BASE_DIR + "model_output/BeatAML/"


class BeatAMLDataset(torch.utils.data.Dataset):
    def __init__(self, indicator):
        self.indices = torch.arange(indicator.shape[0])[indicator]
    def __getitem__(self, index):
        return self.indices[index]
    def __len__(self):
        return len(self.indices)


class BeatAMLCollator:
    def __init__(self, X_norm_expr, X_clinical_attrs, X_mut, drug_inds_1, drug_inds_2, response_averaged_combinations_w_rnaseq, y_bin, all_molecules, drug_metadata_ft, cv_df, tx_df, all_kg_data):
        self.X_norm_expr = torch.from_numpy(X_norm_expr).float()
        self.X_clinical_attrs = torch.from_numpy(X_clinical_attrs).float()
        self.X_mut = torch.from_numpy(X_mut).float()
        self.drug_inds_1 = torch.from_numpy(drug_inds_1).long()
        self.drug_inds_2 = torch.from_numpy(drug_inds_2).long()
        self.all_drug_pairs = response_averaged_combinations_w_rnaseq["drug"].values
        self.all_patients = response_averaged_combinations_w_rnaseq["patient"].values
        self.y_bin = torch.from_numpy(y_bin).float()
        self.all_molecules = all_molecules
        self.drug_metadata_ft = drug_metadata_ft
        self.cv_df = cv_df
        self.tx_df = tx_df
        self.all_kg_data = all_kg_data
        self.kg_sampling_num_neighbors = None
        self.kg_sampling_num_layers = None
        
    
    def __call__(self, batch):
        heads = self.drug_inds_1[batch]
        tails = self.drug_inds_2[batch]
        unique_head_indices, batch_heads_new = torch.unique(heads, return_inverse=True)
        unique_tail_indices, batch_tails_new = torch.unique(tails, return_inverse=True)
        batch_X_norm_expr = self.X_norm_expr[batch]
        batch_X_clinical_attrs = self.X_clinical_attrs[batch]
        batch_X_mut_profiles = self.X_mut[batch]
        batch_y_bin = self.y_bin[batch]
        batch_drug_pairs = self.all_drug_pairs[batch]
        batch_patients = self.all_patients[batch]
        
        unique_head_mod_avail = self.drug_metadata_ft.loc[unique_head_indices][
            ['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        unique_tail_mod_avail = self.drug_metadata_ft.loc[unique_tail_indices][
            ['view_str', 'view_kg', 'view_cv'] + [f'view_tx_{cell_line}' for cell_line in CELL_LINES]]
        
        unique_head_mol_strs = self.all_molecules[unique_head_indices.tolist()]
        unique_tail_mol_strs = self.all_molecules[unique_tail_indices.tolist()]
        
        unique_drug_indices = torch.unique(torch.cat([unique_head_indices, unique_tail_indices], dim=0))
        kg_dict = sample_kg_data(self.all_kg_data, unique_drug_indices, self.kg_sampling_num_neighbors, 
                                 'neighborloader', self.kg_sampling_num_layers, drug_only=True)
        
        # extract Cv, Tx signatures, fill dummies for unavailable ones
        def get_signatures_and_fill_dummy(unique_indices, sig_df, unique_mod_avail, sig, unique_sig_ids):
            unique_sig_output = torch.randn((unique_indices.shape[0], sig_df.shape[0]))
            unique_sig_avail_indices = unique_mod_avail[sig].values == 1
            unique_sig_output[unique_sig_avail_indices, :] = torch.from_numpy(
                sig_df[unique_sig_ids[unique_sig_avail_indices]].values.T).float()
            unique_sig_output[~unique_sig_avail_indices, :] = 0
            return unique_sig_output

        unique_head_cv_sig_ids = self.drug_metadata_ft.loc[unique_head_indices.tolist(), 'cv_sig_id'].values
        unique_tail_cv_sig_ids = self.drug_metadata_ft.loc[unique_tail_indices.tolist(), 'cv_sig_id'].values
        unique_head_cv = get_signatures_and_fill_dummy(
            unique_head_indices, self.cv_df, unique_head_mod_avail, 'view_cv', unique_head_cv_sig_ids)
        unique_tail_cv = get_signatures_and_fill_dummy(
            unique_tail_indices, self.cv_df, unique_tail_mod_avail, 'view_cv', unique_tail_cv_sig_ids)
        
        unique_head_tx_dict = {}
        unique_tail_tx_dict = {}
        for cell_line in CELL_LINES:
            unique_head_tx_dict[cell_line] = {}
            unique_tail_tx_dict[cell_line] = {}
            
            # sigs
            unique_head_tx_cell_line_sig_ids = self.drug_metadata_ft.loc[
                unique_head_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_tail_tx_cell_line_sig_ids = self.drug_metadata_ft.loc[
                unique_tail_indices.tolist(), f'{cell_line}_max_dose_averaged_sig_id'].values
            unique_head_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(
                unique_head_indices, self.tx_df, unique_head_mod_avail, f'view_tx_{cell_line}', unique_head_tx_cell_line_sig_ids)
            unique_tail_tx_dict[cell_line]['sigs'] = get_signatures_and_fill_dummy(
                unique_tail_indices, self.tx_df, unique_tail_mod_avail, f'view_tx_{cell_line}', unique_tail_tx_cell_line_sig_ids)
            
            # drugs
            unique_head_tx_dict[cell_line]['drugs'] = unique_head_indices.long()
            unique_tail_tx_dict[cell_line]['drugs'] = unique_tail_indices.long()
            
            # dosages
            unique_head_tx_dosages = self.drug_metadata_ft.loc[unique_head_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_tail_tx_dosages = self.drug_metadata_ft.loc[unique_tail_indices.tolist(), f'{cell_line}_pert_dose'].fillna(0).values
            unique_head_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_head_tx_dosages).float()
            unique_tail_tx_dict[cell_line]['dosages'] = torch.from_numpy(unique_tail_tx_dosages).float()
            
            # cell_lines (in str, converted to one-hot )
            unique_head_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_head_indices))
            unique_tail_tx_dict[cell_line]['cell_lines'] = np.array([cell_line] * len(unique_tail_indices))
        
        unique_head_masks = torch.from_numpy(1 - unique_head_mod_avail.values).bool()
        unique_tail_masks = torch.from_numpy(1 - unique_tail_mod_avail.values).bool()
            
        return {
            # unique heads
            'head':{
                'drugs': unique_head_indices,
                'strs': unique_head_mol_strs,
                'cv': unique_head_cv,
                'tx': unique_head_tx_dict,
                'masks': unique_head_masks,
            },
            # unique tails
            'tail':{
                'drugs': unique_tail_indices,
                'strs': unique_tail_mol_strs,
                'cv': unique_tail_cv,
                'tx': unique_tail_tx_dict,
                'masks': unique_tail_masks,
            },
            # kg
            'kg': kg_dict,
            # used for indexing the label matrix
            'edge_indices':{
                'head': batch_heads_new,
                'tail': batch_tails_new,
                "norm_expr": batch_X_norm_expr,
                "clinical_attrs": batch_X_clinical_attrs,
                "mut_profiles": batch_X_mut_profiles,
                "drug_pairs": batch_drug_pairs,
                "patients": batch_patients,
                'label': batch_y_bin,
            },
        }
    

def prepare_data(split_method="patient"):
    # load data
    drug_metadata = pd.read_pickle(BASE_DIR + "processed_data/drug_features/drug_metadata_ddi.pkl")

    _, (
        sample_id_map, 
        response_averaged_combinations_w_rnaseq, 
        norm_expr_samples_w_rnaseq_mat, 
        mutation_calls_filtered_wide, 
        clinical_attrs_samples_w_rnaseq,
        beataml_drug_to_our_index_flatten,
    ) = zip(*torch.load(RAW_DATA_DIR+"processed_beataml_data_ready_for_ml.pt").items())

    lab_id_to_rnaseq_id = sample_id_map[["labId", "dbgap_rnaseq_sample"]].set_index("labId").to_dict()["dbgap_rnaseq_sample"]
    y_cont = response_averaged_combinations_w_rnaseq["comb_ratio_log_transformed"].values
    y_bin = (response_averaged_combinations_w_rnaseq["comb_ratio"] < 1).astype(int).values

    if split_method == "patient":
        # Split-by-patients
        patients_train, patients_test = train_test_split(
            response_averaged_combinations_w_rnaseq["patient"].unique(), 
            train_size=0.9, random_state=42,
        )
        test_indicator = response_averaged_combinations_w_rnaseq["patient"].isin(patients_test)
        train_indicator = ~test_indicator

        # PCA for gene expression
        pca = PCA(n_components=256)
        pca.fit(
            norm_expr_samples_w_rnaseq_mat.to_df().loc[list(
                set(sample_id_map.set_index("patientId").loc[patients_train]["dbgap_rnaseq_sample"].dropna().values) & \
                set(norm_expr_samples_w_rnaseq_mat.obs.index.values)
            )].values)
        norm_expr_samples_w_rnaseq_mat_pca = pca.transform(norm_expr_samples_w_rnaseq_mat.to_df().values)
        norm_expr_samples_w_rnaseq_mat_pca = norm_expr_samples_w_rnaseq_mat_pca[:, :150]

        # MCA for mutation
        mutation_calls_filtered_wide = mutation_calls_filtered_wide.loc[:, (mutation_calls_filtered_wide > 0).sum(0) > 2]
        mca = prince.MCA(n_components=60, n_iter=10, check_input=True, copy=False, engine="sklearn", 
                        random_state=42, one_hot=True)
        mca.fit(mutation_calls_filtered_wide.loc[list(
            set(sample_id_map.set_index("patientId").loc[patients_train]["dbgap_rnaseq_sample"].dropna().values) & \
            set(mutation_calls_filtered_wide.index.values)
        )])
        mutation_calls_filtered_wide_mca = mca.transform(mutation_calls_filtered_wide)
        mutation_calls_filtered_wide_mca = mutation_calls_filtered_wide_mca.iloc[:, :30]

        # Impute clinical attributes
        clinical_attrs_samples_w_rnaseq_mat = clinical_attrs_samples_w_rnaseq.values

        imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imp_most_freq.fit(clinical_attrs_samples_w_rnaseq.loc[list(
            set(sample_id_map.set_index("patientId").loc[patients_train]["labId"].dropna().values) & \
            set(clinical_attrs_samples_w_rnaseq.index.values)
        ), :].iloc[:, [0]].values)
        clinical_attrs_samples_w_rnaseq_mat[:, [0]] = imp_most_freq.transform(clinical_attrs_samples_w_rnaseq_mat[:, [0]])

        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp_mean.fit(clinical_attrs_samples_w_rnaseq.loc[list(
            set(sample_id_map.set_index("patientId").loc[patients_train]["labId"].dropna().values) & \
            set(clinical_attrs_samples_w_rnaseq.index.values)
        ), :].iloc[:, [1]].values)
        clinical_attrs_samples_w_rnaseq_mat[:, [1]] = imp_mean.transform(clinical_attrs_samples_w_rnaseq_mat[:, [1]])

        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp_mean.fit(clinical_attrs_samples_w_rnaseq.loc[list(
            set(sample_id_map.set_index("patientId").loc[patients_train]["labId"].dropna().values) & \
            set(clinical_attrs_samples_w_rnaseq.index.values)
        ), :].iloc[:, [13]].values)
        clinical_attrs_samples_w_rnaseq_mat[:, [13]] = imp_mean.transform(clinical_attrs_samples_w_rnaseq_mat[:, [13]])

        imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imp_most_freq.fit(clinical_attrs_samples_w_rnaseq.loc[list(
            set(sample_id_map.set_index("patientId").loc[patients_train]["labId"].dropna().values) & \
            set(clinical_attrs_samples_w_rnaseq.index.values)
        ), :].iloc[:, [14]].values)
        clinical_attrs_samples_w_rnaseq_mat[:, [14]] = imp_most_freq.transform(clinical_attrs_samples_w_rnaseq_mat[:, [14]])
        
    elif split_method == "drug":
        # Split-by-drugs
        drugs_train, drugs_test = train_test_split(
            np.unique(response_averaged_combinations_w_rnaseq[["drug_1", "drug_2"]].values.flatten()), 
            train_size=0.9, random_state=42,
        )
        test_indicator = (
            response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_test) & \
            response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_train)
        ) | \
        (
            response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_train) & \
            response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_test)
        )
        train_indicator = response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_train) & response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_train)

        # verify it's still containing all patients (split-by-drugs is for sure not going contain all drugs in train)
        assert response_averaged_combinations_w_rnaseq["lab_id"][
            response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_train) & \
            response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_train)
        ].nunique() == response_averaged_combinations_w_rnaseq["lab_id"].nunique()
        
        # PCA for gene expression
        pca = PCA(n_components=256)
        norm_expr_samples_w_rnaseq_mat_pca = pca.fit_transform(norm_expr_samples_w_rnaseq_mat.to_df().values)
        norm_expr_samples_w_rnaseq_mat_pca = norm_expr_samples_w_rnaseq_mat_pca[:, :150]

        # MCA for mutation
        mutation_calls_filtered_wide = mutation_calls_filtered_wide.loc[:, (mutation_calls_filtered_wide > 0).sum(0) > 2]
        mca = prince.MCA(n_components=60, n_iter=10, check_input=True, copy=False, engine="sklearn", random_state=42, one_hot=True)
        mutation_calls_filtered_wide_mca = mca.fit_transform(mutation_calls_filtered_wide)
        mutation_calls_filtered_wide_mca = mutation_calls_filtered_wide_mca.iloc[:, :30]

        # Impute clinical attributes
        clinical_attrs_samples_w_rnaseq_mat = clinical_attrs_samples_w_rnaseq.values
        
        imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        clinical_attrs_samples_w_rnaseq_mat[:, [0]] = imp_most_freq.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [0]])

        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        clinical_attrs_samples_w_rnaseq_mat[:, [1]] = imp_mean.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [1]])

        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        clinical_attrs_samples_w_rnaseq_mat[:, [13]] = imp_mean.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [13]])

        imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        clinical_attrs_samples_w_rnaseq_mat[:, [14]] = imp_most_freq.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [14]])

    y_bin_train = y_bin[train_indicator]
    y_bin_test = y_bin[test_indicator]

    print("Train+val:",y_bin_train.shape[0])
    print("Test:", y_bin_test.shape[0])
    
    clinical_attrs_sample_ids = clinical_attrs_samples_w_rnaseq.index.tolist()
    norm_expr_rnaseq_ids = norm_expr_samples_w_rnaseq_mat.obs.index.tolist()
    X_norm_expr = norm_expr_samples_w_rnaseq_mat_pca[
        [norm_expr_rnaseq_ids.index(lab_id_to_rnaseq_id[lab_id]) 
        for lab_id in response_averaged_combinations_w_rnaseq["lab_id"]]]
    X_clinical_attrs = clinical_attrs_samples_w_rnaseq_mat[
        [clinical_attrs_sample_ids.index(lab_id) 
        for lab_id in response_averaged_combinations_w_rnaseq["lab_id"]]]
    X_mut = mutation_calls_filtered_wide_mca.loc[
        [lab_id_to_rnaseq_id[lab_id] for lab_id in response_averaged_combinations_w_rnaseq["lab_id"]]].values

    first_num_drugs = max(beataml_drug_to_our_index_flatten.values()) + 1
    drug_inds_1 = np.array([beataml_drug_to_our_index_flatten[drug_name] for drug_name in response_averaged_combinations_w_rnaseq["drug_1"]])
    drug_inds_2 = np.array([beataml_drug_to_our_index_flatten[drug_name] for drug_name in response_averaged_combinations_w_rnaseq["drug_2"]])
    
    # load drug metadata and structure modality data
    drug_metadata_ft = drug_metadata.iloc[:first_num_drugs, :]
    all_molecules = torch.load(os.path.join(
        PATH_BASE, "drug_features/str/all_molecules_torchdrug.pt"), map_location="cpu")[:first_num_drugs]

    drug_metadata_ft["view_str"] = 1  # all drugs must have structure (filtered already during preprocessing)

    # load all KG modality data
    kg_encoder = 'hgt'
    all_kg_data = torch.load(os.path.join(PATH_BASE, f"drug_features/kg/KG_data_{kg_encoder}.pt"), map_location="cpu")

    # load perturbation data
    cv_df = pd.read_csv(PATH_BASE + 'drug_features/cv/cv_cp_data.csv', index_col=0)
    tx_df = pd.read_csv(PATH_BASE + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)
    
    collator = BeatAMLCollator(X_norm_expr, X_clinical_attrs, X_mut, drug_inds_1, drug_inds_2, response_averaged_combinations_w_rnaseq, y_bin, all_molecules, drug_metadata_ft, cv_df, tx_df, all_kg_data)
    
    if split_method == "patient":
        # Split-by-patients
        patients_train_train, patients_train_val = train_test_split(
            patients_train, 
            train_size=8/9, random_state=42,
        )
        train_train_indicator = response_averaged_combinations_w_rnaseq["patient"].isin(patients_train_train)
        train_val_indicator = response_averaged_combinations_w_rnaseq["patient"].isin(patients_train_val)
    elif split_method == "drug":
        # Split-by-drugs
        drugs_train_train, drugs_train_val = train_test_split(
            drugs_train, 
            train_size=8/9, random_state=42,
        )
        train_train_indicator = (
            response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_train_train) & \
            response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_train_train)
        )
        train_val_indicator = (
            response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_train_val) & \
            response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_train_train)
        ) | \
        (
            response_averaged_combinations_w_rnaseq["drug_1"].isin(drugs_train_train) & \
            response_averaged_combinations_w_rnaseq["drug_2"].isin(drugs_train_val)
        )

    y_bin_train_train = y_bin[train_train_indicator]
    y_bin_train_val = y_bin[train_val_indicator]

    print("Train:", y_bin_train_train.shape[0])
    print("Val:", y_bin_train_val.shape[0])

    # load datasets
    train_dataset = BeatAMLDataset(train_train_indicator.values)
    val_dataset = BeatAMLDataset(train_val_indicator.values)
    test_dataset = BeatAMLDataset(test_indicator.values)

    # load collators
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), drop_last=False, pin_memory=True, shuffle=True, num_workers=4, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), drop_last=False, pin_memory=True, shuffle=False, num_workers=4, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), drop_last=False, pin_memory=True, shuffle=False, num_workers=4, collate_fn=collator)
    
    return collator, train_loader, val_loader, test_loader
    
    
def get_rdkit_descriptors():
    rdkit_descriptors = np.load(BASE_DIR+"model_output/DrugBank/rdkit_descriptors.npy")
    nan_features = set(np.where(np.isnan(rdkit_descriptors))[1])
    inf_features = set(np.where(np.isinf(rdkit_descriptors))[1]) | set(np.where(rdkit_descriptors > 1e6)[1])

    rdkit_descriptors = rdkit_descriptors[:, [i for i in range(rdkit_descriptors.shape[1]) if i not in (inf_features | nan_features)]]
    
    return rdkit_descriptors


def prepare_data_all():
    # load data
    drug_metadata = pd.read_pickle(BASE_DIR + "processed_data/drug_features/drug_metadata_ddi.pkl")

    _, (
        sample_id_map, 
        response_averaged_combinations_w_rnaseq, 
        norm_expr_samples_w_rnaseq_mat, 
        mutation_calls_filtered_wide, 
        clinical_attrs_samples_w_rnaseq,
        beataml_drug_to_our_index_flatten,
    ) = zip(*torch.load(RAW_DATA_DIR+"processed_beataml_data_ready_for_ml.pt").items())

    lab_id_to_rnaseq_id = sample_id_map[["labId", "dbgap_rnaseq_sample"]].set_index("labId").to_dict()["dbgap_rnaseq_sample"]

    response_averaged_combinations_w_rnaseq[["drug_1", "drug_2", "lab_id"]]

    # PCA for gene expression
    pca = PCA(n_components=256)
    norm_expr_samples_w_rnaseq_mat_pca = pca.fit_transform(norm_expr_samples_w_rnaseq_mat.to_df().values)
    norm_expr_samples_w_rnaseq_mat_pca = norm_expr_samples_w_rnaseq_mat_pca[:, :150]

    # MCA for mutation
    mutation_calls_filtered_wide = mutation_calls_filtered_wide.loc[:, (mutation_calls_filtered_wide > 0).sum(0) > 2]
    mca = prince.MCA(n_components=60, n_iter=10, check_input=True, copy=False, engine="sklearn", random_state=42, one_hot=True)
    mutation_calls_filtered_wide_mca = mca.fit_transform(mutation_calls_filtered_wide)
    mutation_calls_filtered_wide_mca = mutation_calls_filtered_wide_mca.iloc[:, :30]

    # Impute clinical attributes
    clinical_attrs_samples_w_rnaseq_mat = clinical_attrs_samples_w_rnaseq.values
    
    imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    clinical_attrs_samples_w_rnaseq_mat[:, [0]] = imp_most_freq.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [0]])

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    clinical_attrs_samples_w_rnaseq_mat[:, [1]] = imp_mean.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [1]])

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    clinical_attrs_samples_w_rnaseq_mat[:, [13]] = imp_mean.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [13]])

    imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    clinical_attrs_samples_w_rnaseq_mat[:, [14]] = imp_most_freq.fit_transform(clinical_attrs_samples_w_rnaseq_mat[:, [14]])

    clinical_attrs_sample_ids = clinical_attrs_samples_w_rnaseq.index.tolist()
    norm_expr_rnaseq_ids = norm_expr_samples_w_rnaseq_mat.obs.index.tolist()
    X_norm_expr = norm_expr_samples_w_rnaseq_mat_pca[
        [norm_expr_rnaseq_ids.index(lab_id_to_rnaseq_id[lab_id]) 
        for lab_id in response_averaged_combinations_w_rnaseq["lab_id"]]]
    X_clinical_attrs = clinical_attrs_samples_w_rnaseq_mat[
        [clinical_attrs_sample_ids.index(lab_id) 
        for lab_id in response_averaged_combinations_w_rnaseq["lab_id"]]]
    X_mut = mutation_calls_filtered_wide_mca.loc[
        [lab_id_to_rnaseq_id[lab_id] for lab_id in response_averaged_combinations_w_rnaseq["lab_id"]]].values

    first_num_drugs = max(beataml_drug_to_our_index_flatten.values()) + 1
    drug_inds_1 = np.array([beataml_drug_to_our_index_flatten[drug_name] for drug_name in response_averaged_combinations_w_rnaseq["drug_1"]])
    drug_inds_2 = np.array([beataml_drug_to_our_index_flatten[drug_name] for drug_name in response_averaged_combinations_w_rnaseq["drug_2"]])
    
    # load drug metadata and structure modality data
    drug_metadata_ft = drug_metadata.iloc[:first_num_drugs, :]
    all_molecules = torch.load(os.path.join(
        PATH_BASE, "drug_features/str/all_molecules_torchdrug.pt"), map_location="cpu")[:first_num_drugs]

    drug_metadata_ft["view_str"] = 1  # all drugs must have structure (filtered already during preprocessing)

    # load all KG modality data
    kg_encoder = 'hgt'
    all_kg_data = torch.load(os.path.join(PATH_BASE, f"drug_features/kg/KG_data_{kg_encoder}.pt"), map_location="cpu")

    # load perturbation data
    cv_df = pd.read_csv(PATH_BASE + 'drug_features/cv/cv_cp_data.csv', index_col=0)
    tx_df = pd.read_csv(PATH_BASE + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)
    
    collator = BeatAMLCollator(X_norm_expr, X_clinical_attrs, X_mut, drug_inds_1, drug_inds_2, response_averaged_combinations_w_rnaseq, y_bin, all_molecules, drug_metadata_ft, cv_df, tx_df, all_kg_data)
    
    # load datasets
    dataset = BeatAMLDataset(np.ones(num_total).astype(bool))

    # load collators
    loader = DataLoader(dataset, batch_size=len(dataset), drop_last=False, pin_memory=True, shuffle=True, num_workers=4, collate_fn=collator)
    
    return collator, loader



