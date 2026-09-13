<h1 align="center">
  Madrigal: A Unified Multimodal Model for Predicting Drug Combination Effects
</h1>

## 👀 Overview of Madrigal

Madrigal is an open-source model for predicting drug combination outcomes from multimodal preclinical data. This repository contains the code for the model described in our [paper](https://arxiv.org/abs/2503.02781) and on the [project page](https://zitniklab.hms.harvard.edu/projects/Madrigal/).

## 🚀 Preparations

1. Clone this repository and install the `madrigal` package (see [Installing `madrigal`](#installing-madrigal)).
2. Choose a directory for the data and checkpoints, and create a `.env` file in the root of this repository that points to it (see [Setting up data and checkpoint directories](#setting-up-data-and-checkpoint-directories)).
3. Download the data and model checkpoints from our [Harvard Dataverse repository](https://doi.org/10.7910/DVN/ZFTW3J) into that directory.

To make predictions for a few drug pairs with the released checkpoints, run [quick_predictions.ipynb](notebooks/quick_predictions.ipynb) after steps 1-3. The notebook computes the model's predicted scores for each query. It can also look up the corresponding prediction scores used throughout the paper, which rank a score among those of all DrugBank drug pairs for the same outcome. Because that ranking requires scoring every drug pair, the prediction scores are distributed precomputed (see Requirements below).

## Detailed instructions

### Installing `madrigal`
Create the conda environment with `mamba env create -f env_new.yaml`. This takes under an hour (see the [mamba installation guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) if you do not have mamba yet). The environment targets CUDA 11.7 and gcc 9.2; for other CUDA or PyTorch versions, see [Adapting the environment](#adapting-the-environment).

Then activate the environment and install the package in editable mode:
```
mamba activate madrigal_env
cd /path/to/Madrigal
python -m pip install -e .
```
Check the install:
```
python -c "import madrigal; print('Imported')"
```

### Adapting the environment
`env_new.yaml` uses PyTorch 1.13.1, which needs CUDA below 12.0, and installs the PyTorch Geometric extensions (`pyg-lib`, `torch-scatter`, `torch-cluster`) from a wheel index built for CUDA 11.7. To install for other versions:

1. Check your PyTorch and CUDA versions: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
2. Point the `--find-links` line in `env_new.yaml` to the matching index, for example `https://data.pyg.org/whl/torch-2.1.0+cu121.html`. All indices are listed at [data.pyg.org/whl](https://data.pyg.org/whl/).
3. Update `pytorch` and `cudatoolkit` in the conda section to match. If the three extensions have no build for your versions, remove their version pins.

With CUDA 12 or later, `pytorch=2.1.0` with `pytorch-geometric<2.4.0` also works; we tested it with CUDA 12.8 and gcc 14.2. `env_locked_linux64.yaml` has the full environment we used for the paper.

### Setting up data and checkpoint directories
The data and checkpoints on Dataverse are organized as the code expects:
```
Madrigal_Data (BASE_DIR)
|-- processed_data
|  |-- drug_combination_data
|  |  |-- DrugBank
|  |  |  |-- drugbank_ddi_directed.tsv, drugbank_ddi_directed_label_map.pkl
|  |  |  |-- split_by_drugs_atc, split_by_drugs_random, split_by_drugs_targets, split_by_pairs
|  |  |  |  |-- train/validation/test tables
|  |  |-- TWOSIDES (same layout)
|  |-- drug_features
|  |  |-- drug_metadata_ddi.pkl, drug_metadata_nash.pkl (drug metadata; key to all other files)
|  |  |-- str (torchdrug-generated molecular graphs)
|  |  |-- kg (PyG-generated knowledge graphs)
|  |  |-- cv (cell viability table)
|  |  |-- tx (transcriptomics table and embeddings)
|-- model_output
|  |-- pretrain
|  |  |-- DrugBank/checkpoint_1000.pt, TWOSIDES/checkpoint_1000.pt (modality alignment)
|  |-- DrugBank
|  |  |-- split_by_pairs/all_train_seed{0,1,2,42,99} (fine-tuned on all DrugBank outcomes; used for inference)
|  |  |-- split_by_drugs_random/{madrigal,no_pretrain,structure_only_no_pretrain}_seed42 (used in the Fig. 2 notebooks)
|  |-- TWOSIDES
|  |  |-- split_by_pairs/all_train_seed{0,1,2,42,99} (fine-tuned on all TWOSIDES outcomes)
|-- raw_data (external datasets used in the notebooks)
```

Then add a `.env` file to the repository root with these paths, each ending in `/`:
```
PROJECT_DIR=/path/to/Madrigal/
BASE_DIR=/path/to/Madrigal_Data/
DATA_DIR=/path/to/Madrigal_Data/processed_data/
ENCODER_CKPT_DIR=/path/to/Madrigal/modality_pretraining/
CL_CKPT_DIR=/path/to/Madrigal_Data/model_output/pretrain/
```

## Notebooks

Inference with the released checkpoints:
- [generate_embeddings.ipynb](notebooks/generate_embeddings.ipynb): compute drug embeddings and predicted scores for a checkpoint, convert them to normalized ranks, and aggregate the ranks across the five checkpoints into prediction scores. [`notebooks/generate_embeddings_and_scores/`](notebooks/generate_embeddings_and_scores/) contains the same steps as command-line scripts.
- [quick_predictions.ipynb](notebooks/quick_predictions.ipynb): predicted scores and prediction scores for specific (outcome, drug A, drug B) queries.
- [normalize_scores.py](notebooks/normalize_scores.py): rank normalization for one checkpoint, from the raw-score tensor written by `generate_embeddings.ipynb` or `generate_embeddings_and_scores.py`.

Analyses in the paper:

| Notebook | Content | Requirement |
|-|-|-|
| [fig1_pretrained_embeds](notebooks/fig1/fig1_pretrained_embeds.ipynb) | UMAP of modality-specific embeddings before and after modality alignment (Fig. 1d) | Dataverse |
| [fig2_model_analyses](notebooks/fig2/fig2_model_analyses.ipynb) | Test performance as a function of similarity to training drugs (Fig. 2e,f) | Dataverse |
| [fig2_modality_ablations](notebooks/fig2/fig2_modality_ablations.ipynb) | Outcome-stratified performance of Madrigal and its ablations, and performance by available modalities (Fig. 2g) | Dataverse |
| [fig3_self_combo](notebooks/fig3/fig3_self_combo.ipynb) | Organ-specific adverse effects of individual drugs vs. FDA concern levels (Fig. 3a-c) | Dataverse |
| [fig3_transporter_mediated_ddis](notebooks/fig3/fig3_transporter_mediated_ddis.ipynb) | Transporter-, carrier-, and enzyme-mediated DDIs (Fig. 3d-h) | Dataverse, prediction scores |
| [fig4_clinical_trials_combos](notebooks/fig4/fig4_clinical_trials_combos.ipynb) | Comparison with adverse event data from clinical trials (Fig. 4b) | Dataverse, ToolUniverse |
| [fig4_parpi](notebooks/fig4/fig4_parpi.ipynb) | Safety of clinically tested cancer drug combinations, including PARP inhibitors (Fig. 4c) | Dataverse, prediction scores |
| [fig5_t2d_mash](notebooks/fig5/fig5_t2d_mash.ipynb) | Combinations for type 2 diabetes, heart failure, and MASH (Fig. 4d-f) | Dataverse, prediction scores |
| [fig6_beataml](notebooks/fig6/beataml/) | Drug combination synergy in BeatAML (Fig. 5b); `run_finetune_beataml.sh` then `plot_beataml.py` | Dataverse (the BeatAML cohort data are not distributed) |
| [fig6_PDX](notebooks/fig6/fig6_PDX.ipynb) | Drug combination efficacy in patient-derived xenografts (Fig. 5c-e) | Dataverse |
| [fig6_clinical_validation_dfci](notebooks/fig6/fig6_clinical_validation_dfci.ipynb) | Adverse events in the DFCI oncology cohort; [`dfci_patient_prediction_per_run.py`](notebooks/fig6/dfci_patient_prediction_per_run.py) computes the patient-level AUROCs (Fig. 5i, Supplementary Table S18) | Dataverse (the patient-level data are not distributed) |
| [discussions_proteomics_analysis](notebooks/discussions/discussions_proteomics_analysis.ipynb) | Correlation with proteomics data (Supplementary Fig. S9) | Dataverse |
| [discussions_combomatch](notebooks/discussions/discussions_combomatch.ipynb) | Predictions for ComboMATCH drug pairs (Supplementary Fig. S9) | Dataverse, prediction scores |
| [LM_decoder](LM_decoder/) | Language-model decoder over textual outcome descriptions (Supplementary Fig. S10) | Dataverse |

### Requirements
- [Harvard Dataverse](https://doi.org/10.7910/DVN/ZFTW3J): processed data, raw data used in the analyses, and model checkpoints.
- [Prediction scores](https://drive.google.com/file/d/1_TuoEoAthaZJe93b3zlxmwP88csEQ9sQ/view?usp=sharing): precomputed prediction scores for all DrugBank outcomes and drug pairs (80 GB). Place the file at `model_output/DrugBank/split_by_pairs/DrugBank_drugs_normalized_ranks.npy`, or regenerate it with `generate_embeddings.ipynb`.
- [ToolUniverse](https://github.com/mims-harvard/ToolUniverse): used to retrieve clinical trial adverse event data.
- Arial: the figure code loads it from `notebooks/arial.ttf`. Add a copy there to reproduce the figure fonts.

## 🛠️ Training

Madrigal is trained in three stages. The scripts below are SLURM job scripts: edit the `#SBATCH` header and `base` path for your machine, or run the Python command in them directly.

| Stage | Script | Configuration |
|-|-|-|
| Modality adaptation of each encoder | [`modality_pretraining/`](modality_pretraining/) | [`configs/chemcpa/`](configs/chemcpa/) (transcriptomics) |
| Contrastive modality alignment | [`scripts/cl_pretrain/run_pretrain_{drugbank,twosides}.sh`](scripts/cl_pretrain/) | [`configs/cl_pretrain/`](configs/cl_pretrain/) |
| Fine-tuning on drug combination outcomes (train/validation/test splits, with the ablation models of Fig. 2) | [`scripts/ddi_finetune/finetune_{drugbank,twosides}_scale.sh`](scripts/ddi_finetune/) | [`configs/ddi_finetune/`](configs/ddi_finetune/) |
| Fine-tuning on all drug combination outcomes (the released checkpoints, used for inference) | [`scripts/ddi_finetune/finetune_{drugbank,twosides}_all_train.sh`](scripts/ddi_finetune/) | [`configs/ddi_finetune/`](configs/ddi_finetune/) |

One fine-tuning run on DrugBank takes about an hour on a single GPU.

For example, fine-tuning on DrugBank with `split_by_drugs_targets` and seed 0:
```
python train_ddi_batch.py --checkpoint=DrugBank/checkpoint_1000.pt --finetune_mode="str_str+random_sample" --split_method=split_by_drugs_targets --repeat=None --seed=0 --from_yaml=configs/ddi_finetune/DrugBank/bottleneck_all_available.yaml --run_name=madrigal_seed0
```
`--checkpoint` is the modality-alignment checkpoint, relative to `CL_CKPT_DIR`. Each run writes its checkpoints and log to `model_output/<dataset>/<split>/<timestamp>_<run_name>/`. To fine-tune from an alignment checkpoint you trained yourself, pass `--checkpoint=<dataset>/<split>/<timestamp>_<run_name>/checkpoint_1000.pt`.

## 🌟 Using your own dataset

Using your own dataset takes some changes to the code:
- Arguments in `madrigal/parse_args.py`:
  - `data_source`: which dataset directory is loaded, and the training and evaluation setup for it.
  - `split_method`: which split directory is loaded, and how evaluation is done.
  - `task`: `binary`, `multiclass`, or `multilabel`.
  - `loss_fn_name`: `bce`, or `ce` for `multiclass`. The other choices are not implemented.
- Data files, in the same format as the released ones:
  - Drugs
    - Metadata: all other drug files follow its order.
    - Structure: molecular graphs built with `torchdrug`, in metadata order.
    - Knowledge graph: `PyG` `HeteroData` objects, with drug nodes indexed in metadata order.
    - Cell viability and transcriptomics: tables.
    - The transcriptomics encoder reads `drug_features/tx/embeddings/rdkit2D_embeddings_combined_all_normalized.parquet`, so regenerate that file for your drugs.
  - Drug combination outcomes
    - Tables of (label_indexed, head (drug 1), tail (drug 2), negs*). What the negative columns mean depends on the split.
    - A mapping from outcome label index to outcome.

## Known issues
1. Import `torchdrug` after `torch_geometric`.
2. `torchdrug>=0.2.0.post1` is required; earlier versions have an [issue](https://github.com/DeepGraphLearning/torchdrug/issues/148) with the LR scheduler.
3. `torchdrug` needs `torch-scatter` and `torch-cluster`. Import errors from these packages usually mean their wheels do not match your PyTorch and CUDA versions (see [Adapting the environment](#adapting-the-environment)).

## ⚖️ License

The code is released under the MIT License.

Each third-party dataset used in the paper keeps the terms of its source. Check the terms before reusing the data, especially for commercial purposes.

- DrugBank (release 2023-01-04): drug metadata, SMILES, and the drug-drug interaction table with its descriptions. DrugBank's academic license is non-commercial.
- TWOSIDES (2019-11-13): drug-drug interactions derived from FAERS.
- PrimeKG (Harvard Dataverse, v2): the knowledge graph, CC0 1.0.
- PRISM Repurposing 19Q4 (DepMap): cell viability profiles.
- Connectivity Map, CMap 2020: transcriptomics profiles.
- MUV (MoleculeNet): pretraining of the structure encoder.
- UniProt ID mapping (December 2022), Open Targets, ChEMBL, DrugCentral, the FDA Orange Book, CancerDrugs_DB, DILIrank, DICTrank, DIQTA: drug and target annotations used in the analyses.
- CDCDB (AACT, 2024-04-16) and ClinicalTrials.gov: clinical trial arms and adverse events.
- PDXE (Gao et al., 2015) and the HCT116 proteome data of Mitchell et al. (2023): supplementary tables of the original papers.

## Citation
If you use Madrigal, please cite our preprint:
```
@article{Huang2025.arXiv:2503.02781,
  author = {Huang, Yepeng and Su, Xiaorui and Ullanat, Varun and Moon, Intae and Liang, Ivy and Clegg, Lindsay and Olabode, Damilola and Johnson, Ruthie and Ho, Nicholas and Gibbs, Megan and Gusev, Alexander and John, Bino and Zitnik, Marinka},
  title = {Multimodal AI predicts clinical outcomes of drug combinations from preclinical data},
  journal = {arXiv preprint arXiv:2503.02781},
  year = {2025},
  doi = {10.48550/arXiv.2503.02781},
  URL = {https://arxiv.org/abs/2503.02781},
}
```
