from pathlib import Path
from pprint import pprint

from seml.config import generate_configs, read_config

from .experiments_run import TxAdaptingExperimentWrapper

if __name__ == "__main__":
    exp = TxAdaptingExperimentWrapper(init_all=False)

    # this is how seml loads the config file internally
    assert Path(
        "../manual_run.yaml"
    ).exists(), "config file not found"
    seml_config, slurm_config, experiment_config = read_config(
        "../manual_run.yaml"
    )
    # we take the first config generated
    configs = generate_configs(experiment_config)
    if len(configs) > 1:
        raise Exception("More than one config generated from the yaml file")
    args = configs[0]
    pprint(args)

    exp.seed = 42
    
    # loads the dataset splits
    exp.init_dataset(**args["dataset"])
    
    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_ckpt=args["model"]["pretrained_model_ckpt"],
        use_drugs=args["model"]["use_drugs"],
    )
    
    # setup the torch DataLoader
    exp.update_datasets()

    res = exp.train(**args["training"])
