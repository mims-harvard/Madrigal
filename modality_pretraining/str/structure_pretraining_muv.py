import torch
from torchdrug import data, datasets
import os
from torchdrug import core, models, tasks, utils
from torchdrug import data, utils
from torchdrug.core import Registry as R
import os
import sys
import logging
from itertools import islice
import argparse
import pickle

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
from torchdrug import data, utils
from torchdrug.core import Registry as R

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols


parser = argparse.ArgumentParser(description='Structure pretraining')

parser.add_argument('--gin_hidden_dims', type=int, nargs='+', default=[128, 128, 128], help='GIN hidden dimensions')  # or [256, 256, 256, 256]
parser.add_argument('--gin_edge_input_dim', type=int, default=18, help='GIN edge input dimension, should be 18 if we use the default edge feature')
parser.add_argument('--gin_num_mlp_layer', type=int, default=3, help='number of MLP layers (AGG) for each GIN layer')
parser.add_argument('--gin_eps', type=float, default=0, help='GIN initial eps')
parser.add_argument('--gin_batch_norm', action='store_true', help='whether to use batch norm in GIN')
parser.add_argument('--gin_actn', type=str, default='relu', help='GIN activation', choices=['relu', 'leaky_relu', 'selu', 'gelu', 'tanh', 'sigmoid', 'softplus'])
parser.add_argument('--gin_readout', type=str, default='mean', help='GIN readout', choices=['sum', 'mean'])
parser.add_argument('--feature_dim', type=int, default=128, help='input feature dimension to transformer (i.e. output feature dimension of view encoders, position embedder and CLS embedder)')

args = parser.parse_args()


@R.register("datasets.MUV_filt")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class MUV_filt(data.MoleculeDataset):
    """
    Subset of PubChem BioAssay by applying a refined nearest neighbor analysis.

    Statistics:
        - #Molecule: 93,087
        - #Classification task: 17

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    target_fields = ["MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                     "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"]

    def __init__(self, path, verbose=1, **kwargs):

        self.load_csv(path, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)
        

dataset = MUV_filt("muv_filtered_torchdrug_new.csv")

lengths = [int(0.9 * len(dataset)), int(0.05 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

model = models.GraphIsomorphismNetwork(input_dim=dataset.node_feature_dim, hidden_dims=args.gin_hidden_dims+[args.feature_dim],
                                       edge_input_dim=args.gin_edge_input_dim, num_mlp_layer=args.gin_num_mlp_layer, 
                                       eps=args.gin_eps, batch_norm=args.gin_batch_norm, activation=args.gin_actn, 
                                       readout=args.gin_readout)

task = tasks.PropertyPrediction(model, task=dataset.tasks, criterion="bce", metric=("auprc", "auroc"))
                                
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0], batch_size=10000)
solver.train(num_epoch=500)
solver.evaluate("valid")

torch.save(model.state_dict(), "GIN_256x4_muv.pt")
solver.save("GIN_256x4_full_muv.pth")