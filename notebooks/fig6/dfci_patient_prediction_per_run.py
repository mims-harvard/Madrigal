"""Patient-level adverse-event prediction in the DFCI oncology cohort (Fig. 5i).

For each of 13 adverse events, a random forest predicts the event from patient covariates plus a
drug-regimen representation: Madrigal drug embeddings (one run per checkpoint and seed), Morgan
fingerprints, or a one-hot regimen encoding. Every per-run and per-fold AUROC is written out.
Requires the DFCI cohort table (`dfci/dfci_patient_data_filtered.pkl`), which is not distributed.

Run counts per outcome (13 outcomes):
  Madrigal      5 seeds x 5 checkpoints = 25 runs, each a 5-fold mean
                (also aggregated to 5 per-seed values, averaging over ckpts)
  Morgan FP     5 seeds                 =  5 runs, each a 5-fold mean
  One-hot       5 seeds                 =  5 runs, each a 5-fold mean

Outputs (aborts if the target files already exist):
  out_per_run_tissue_excl/dfci_per_run_aurocs.csv   long format, one row per run
  out_per_run_tissue_excl/dfci_per_fold_aurocs.csv  long format, one row per fold
  out_per_run_tissue_excl/dfci_per_run_summary.csv  mean/std per outcome per method

Run from Madrigal/notebooks/fig6/ with the .env file set up (see README.md).
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from madrigal.utils import BASE_DIR

# --- configuration ------------------------------------------------------------
TWOSIDES_CKPTS = [
    "all_train_seed0",
    "all_train_seed1",
    "all_train_seed2",
    "all_train_seed42",
    "all_train_seed99",
]
EPOCH = 700
SEEDS = [0, 1, 2, 42, 99]
N_SPLITS = 5
KFOLD_SEED = 42
PCA_DIM = 32
FP_BITS = 32
INCLUDE_TISSUE_TYPE = False  # tumour tissue type is near-collinear with the two-drug regimen; excluded
TOP_TISSUE_ONLY = 10         # (unused when INCLUDE_TISSUE_TYPE is False)
STANDARDIZE_SCORES = False

HEME_TISSUES = {"Myeloid", "Lymphoid", "Myeloma", "Leukemia", "Hematologic Other"}
# Drugs used in haematological malignancy are excluded because they confound haematological AE measurement.
HEME_DRUGS = {
    "Acalabrutinib", "Arsenic trioxide", "Azacitidine", "Bortezomib", "Busulfan",
    "Cytarabine", "Dasatinib", "Daunorubicin", "Decitabine", "Duvelisib",
    "Fludarabine", "Gilteritinib", "Ixazomib", "Lenalidomide", "Midostaurin",
    "Romidepsin", "Ruxolitinib", "Tretinoin", "Umbralisib", "Venetoclax",
    "Vincristine", "Imatinib", "Hydroxyurea", "Cyclophosphamide", "Methotrexate",
}

OUT_DIR = Path(os.getenv("OUT_DIR", "./out_per_run_tissue_excl"))


def standardise(df, scaler=None):
    """Z-score numeric (non-binary) columns; leave dummies as-is."""
    df = df.copy()
    num_mask = (df.dtypes == float) | (df.dtypes == int)
    num_cols = [c for c in df.columns[num_mask] if set(df[c].dropna().unique()) - {0, 1}]
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df, scaler
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def build_patient_features(data, include_regimen):
    """One-hot patient covariates; optionally append the one-hot drug regimen."""
    drop_cols = ["MALE"]
    cols = ["GENDER_NM"]
    if INCLUDE_TISSUE_TYPE:
        drop_cols += ["Other"]
        cols += ["ICD_BASED_TISSUE_TYPE_REDUCED"]
    drop_cols += ["OTHER"]
    cols += ["RACE"]
    if include_regimen:
        drop_cols += ["CARBOPLATIN+PACLITAXEL"]
        cols += ["FIRST_DRUG_REGIMEN"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=drop_cols)
    X = pd.DataFrame(ohe.fit_transform(data[cols]), columns=ohe.get_feature_names_out(), index=data.index)
    return pd.concat([data[["PALLIATIVE_INTENT", "AGE_AT_TREAT"]], X], axis=1)


def load_patient_data():
    data = pd.read_pickle("./dfci/dfci_patient_data_filtered.pkl")
    assert not data.index.has_duplicates
    data = data.query(
        "(drug_name_1 not in @HEME_DRUGS and drug_name_2 not in @HEME_DRUGS) "
        "and ICD_BASED_TISSUE_TYPE not in @HEME_TISSUES"
    )
    data = data[~data["FIRST_DRUG_REGIMEN"].str.contains("PEGYLATEDLIPOSOMALDOXORUBICIN")]
    rare = data["RACE"].value_counts()[data["RACE"].value_counts() < 100].index.values
    data.loc[data["RACE"].isin(rare), "RACE"] = "OTHER"
    top = data["ICD_BASED_TISSUE_TYPE"].value_counts().nlargest(TOP_TISSUE_ONLY).index
    if "UNSPECIFIED" in top:
        top = top.drop("UNSPECIFIED")
    data["ICD_BASED_TISSUE_TYPE_REDUCED"] = np.where(
        data["ICD_BASED_TISSUE_TYPE"].isin(top), data["ICD_BASED_TISSUE_TYPE"], "Other"
    )
    return data


def morgan_fps(data, n_bits=FP_BITS):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    meta = pd.read_pickle(os.path.join(BASE_DIR, "processed_data/drug_features/drug_metadata_ddi.pkl"))
    inds = np.unique(data[["drug_index_1", "drug_index_2"]].values.flatten())
    out = {}
    for ind, smi in zip(inds, meta["canonical_smiles"].loc[inds].values.tolist()):
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(n_bits, dtype=np.uint8)
        if mol is not None:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useChirality=True)
            DataStructs.ConvertToNumpyArray(bv, arr)
        out[ind] = arr.astype(float)
    return out


def load_embeddings(ckpt):
    """Load and PCA-reduce a checkpoint's drug embeddings (cached by caller)."""
    path = os.path.join(BASE_DIR, f"model_output/TWOSIDES/split_by_pairs/{ckpt}/all_drug_embeddings_full_{EPOCH}.pt")
    embeds = torch.load(path, map_location="cpu").numpy()
    if PCA_DIM < embeds.shape[1]:
        embeds = PCA(n_components=PCA_DIM, svd_solver="full").fit_transform(embeds)
    return embeds


def pair_frames(data, patient_X, mat, prefix):
    """Drug-1 / drug-2 feature blocks plus their swapped counterparts."""
    d1 = pd.DataFrame(mat[data["drug_index_1"].values, :], index=patient_X.index,
                      columns=[f"{prefix}_{i}_drug_1" for i in range(mat.shape[1])])
    d2 = pd.DataFrame(mat[data["drug_index_2"].values, :], index=patient_X.index,
                      columns=[f"{prefix}_{i}_drug_2" for i in range(mat.shape[1])])
    X = pd.concat([patient_X, d1, d2], axis=1)
    d2_as_1 = d2.rename(columns={c: c.replace("_drug_2", "_drug_1") for c in d2.columns})
    d1_as_2 = d1.rename(columns={c: c.replace("_drug_1", "_drug_2") for c in d1.columns})
    X_rev = pd.concat([patient_X, d2_as_1, d1_as_2], axis=1)
    return X, X_rev


def cv_scores(X, y, seed, X_rev=None):
    """5-fold CV; returns per-fold (auroc, auprc). Folds are shared across methods."""
    if STANDARDIZE_SCORES:
        X = X.astype(float)
        if X_rev is not None:
            X_rev = X_rev.astype(float)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=KFOLD_SEED)
    aurocs, auprcs = [], []
    for tr, va in skf.split(X.index.values, y):
        if X_rev is not None:
            merged_tr, scaler = standardise(pd.concat([X.iloc[tr], X_rev.iloc[tr]], axis=0))
            X_tr = np.concatenate([merged_tr.iloc[:len(tr)].values.astype(float),
                                   merged_tr.iloc[len(tr):].values.astype(float)], axis=0)
            y_tr = np.tile(y[tr], 2)
        else:
            scaled_tr, scaler = standardise(X.iloc[tr])
            X_tr, y_tr = scaled_tr.values.astype(float), y[tr]

        clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=4,
                                     random_state=seed, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        if X_rev is not None:
            merged_va = standardise(pd.concat([X.iloc[va], X_rev.iloc[va]], axis=0), scaler)
            X_va = np.concatenate([merged_va.iloc[:len(va)].values.astype(float),
                                   merged_va.iloc[len(va):].values.astype(float)], axis=0)
            pred = clf.predict_proba(X_va)[:, 1].reshape(2, -1).mean(0)
        else:
            pred = clf.predict_proba(standardise(X.iloc[va], scaler).values.astype(float))[:, 1]

        aurocs.append(roc_auc_score(y[va], pred))
        auprcs.append(average_precision_score(y[va], pred))
    return aurocs, auprcs


def main():
    for f in ("dfci_per_run_aurocs.csv", "dfci_per_fold_aurocs.csv", "dfci_per_run_summary.csv"):
        if (OUT_DIR / f).exists():
            sys.exit(f"ERROR: {OUT_DIR / f} exists; refusing to overwrite. Move it aside first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_patient_data()
    aes = [c[len("AE_"):] for c in data.columns if c.startswith("AE_")]
    patient_X = build_patient_features(data, include_regimen=False)
    patient_X_ohe = build_patient_features(data, include_regimen=True)
    fps = morgan_fps(data)
    fp_mat = np.zeros((max(fps) + 1, FP_BITS))
    for ind, v in fps.items():
        fp_mat[ind] = v
    print(f"patients={data.shape[0]} regimens={data['FIRST_DRUG_REGIMEN'].nunique()} "
          f"outcomes={len(aes)} patient_features={patient_X.shape[1]}", flush=True)

    embed_cache = {c: load_embeddings(c) for c in TWOSIDES_CKPTS}
    print(f"loaded {len(embed_cache)} checkpoints, embedding dim {next(iter(embed_cache.values())).shape[1]}", flush=True)

    run_rows, fold_rows = [], []
    for ae in aes:
        y = data[f"AE_{ae}"].values
        for seed in SEEDS:
            # Madrigal: one run per checkpoint
            for ckpt in TWOSIDES_CKPTS:
                X, X_rev = pair_frames(data, patient_X, embed_cache[ckpt], "madrigal")
                au, ap = cv_scores(X, y, seed, X_rev)
                run_rows.append(dict(outcome=ae, method="madrigal", seed=seed, ckpt=ckpt,
                                     auroc=np.mean(au), auprc=np.mean(ap)))
                fold_rows += [dict(outcome=ae, method="madrigal", seed=seed, ckpt=ckpt,
                                   fold=i, auroc=a, auprc=p) for i, (a, p) in enumerate(zip(au, ap))]

            # Morgan fingerprint baseline
            X, X_rev = pair_frames(data, patient_X, fp_mat, "morgan")
            au, ap = cv_scores(X, y, seed, X_rev)
            run_rows.append(dict(outcome=ae, method="morgan_fp", seed=seed, ckpt=None,
                                 auroc=np.mean(au), auprc=np.mean(ap)))
            fold_rows += [dict(outcome=ae, method="morgan_fp", seed=seed, ckpt=None,
                               fold=i, auroc=a, auprc=p) for i, (a, p) in enumerate(zip(au, ap))]

            # One-hot regimen baseline
            au, ap = cv_scores(patient_X_ohe, y, seed, None)
            run_rows.append(dict(outcome=ae, method="one_hot_regimen", seed=seed, ckpt=None,
                                 auroc=np.mean(au), auprc=np.mean(ap)))
            fold_rows += [dict(outcome=ae, method="one_hot_regimen", seed=seed, ckpt=None,
                               fold=i, auroc=a, auprc=p) for i, (a, p) in enumerate(zip(au, ap))]
        print(f"  done {ae}", flush=True)

    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUT_DIR / "dfci_per_run_aurocs.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "dfci_per_fold_aurocs.csv", index=False)

    # per-seed values: Madrigal averaged over checkpoints -> 5 per outcome per method
    per_seed = runs.groupby(["outcome", "method", "seed"])[["auroc", "auprc"]].mean().reset_index()
    summary = per_seed.groupby(["outcome", "method"])[["auroc", "auprc"]].agg(["mean", "std", "count"])
    summary.to_csv(OUT_DIR / "dfci_per_run_summary.csv")
    print(summary.round(4).to_string())
    print(f"\nwrote {len(runs)} runs and {len(fold_rows)} fold scores to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
