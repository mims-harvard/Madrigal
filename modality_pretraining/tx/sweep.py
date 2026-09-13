import torch, os
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint

from seml.config import generate_configs, read_config

from madrigal.chemcpa.chemCPA.experiments_run import TxAdaptingExperimentWrapper
from madrigal.utils import PROJECT_DIR

exp = TxAdaptingExperimentWrapper(init_all=False)

assert os.path.exists(
    PROJECT_DIR+"configs/chemcpa/chemcpa_tx_adapting_configs_sweep.yaml"
), "config file not found"
_, _, experiment_config = read_config(
    PROJECT_DIR+"configs/chemcpa/chemcpa_tx_adapting_configs_sweep.yaml"
)
# we take the first config generated
configs = generate_configs(experiment_config)
args = configs[0]

exp.init_dataset(**args["dataset"])
exp.init_drug_embedding(embedding=args["model"]["embedding"])

exp.init_model(
    hparams=args["model"]["hparams"],
    additional_params=args["model"]["additional_params"],
    load_pretrained=args["model"]["load_pretrained"],
    append_ae_layer=args["model"]["append_ae_layer"],
    pretrained_model_path=args["model"]["pretrained_model_path"],
    pretrained_model_ckpt=args["model"]["pretrained_model_ckpt"],
    use_drugs=args["model"]["use_drugs"]
)
exp.update_datasets()

all_res = [exp.train(**args["training"])]

for i in range(1, len(configs)):
    args = configs[i]
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_ckpt=args["model"]["pretrained_model_ckpt"],
        use_drugs=args["model"]["use_drugs"]
    )
    exp.update_datasets()
    all_res.append(exp.train(**args["training"]))
