import math
import pandas as pd
import numpy as np
from pprint import pformat
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss

## PyTorch Geometric
from torch_geometric.nn import HANConv, RGCNConv, HGTConv, HeteroLinear
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torchdrug import models

from ..utils import (
    MOL_DIM, 
    CELL_LINES, 
    CELL_LINES_CAPITALIZED,
    NON_TX_MODALITIES, 
    NUM_NON_TX_MODALITIES,
    NUM_MODALITIES,
    DATA_DIR,
    ENCODER_CKPT_DIR,
)
from ..chemcpa.chemCPA.model import TxAdaptingComPert

TX_INPUT_DIM = 978
actn2actfunc = {'relu': nn.ReLU(inplace=True), 'leakyrelu': nn.LeakyReLU(inplace=True), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'selu': nn.SELU(inplace=True), 'softplus': nn.Softplus(), 'gelu': nn.GELU(), None: nn.Identity()}


import os
def env_flag_is_true(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "t", "yes", "y"}


### Encoders
# TorchDrug
"""
Using TorchDrug models
"""

# KG Encoding
class HAN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, heads: int, metadata, negative_slope: float = 0.2, dropout: float = 0):
        """
        After HAN, we apply a linear layer to get the final embeddings of drugs.
        """
        super(HAN, self).__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(HANConv(in_channels, hidden_channels, metadata, heads, negative_slope=negative_slope, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(HANConv(hidden_channels, hidden_channels, metadata, heads, negative_slope=negative_slope, dropout=dropout))
        
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:  # metadata = Tuple[List[node_types], List[Tuple[src, relation, dst]]]
            if node_type != 'drug':
                continue
            self.lin_dict[node_type] = nn.Linear(hidden_channels, out_channels)
            
    def forward(self, x_dict, edge_index_dict):
        out = self.convs[0](x_dict, edge_index_dict)
        for i in range(1, len(self.convs)):
            out = self.convs[i](out, edge_index_dict)
            if i < len(self.convs) - 1:
                out = {node_type: x.relu_() for node_type, x in out.items()}
        
        return {'drug':self.lin_dict['drug'](out['drug'])}
    

class HGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, metadata, group='sum'):
        super(HGT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(HGTConv(in_channels, hidden_channels, metadata, num_heads, group=group))
        for _ in range(num_layers-1):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group=group)
            self.convs.append(conv)

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:  # metadata = Tuple[List[node_types], List[Tuple[src, relation, dst]]]
            self.lin_dict[node_type] = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.convs[0](x_dict, edge_index_dict)
        for i in range(1, len(self.convs)):
            out = self.convs[i](out, edge_index_dict)
            if i < len(self.convs) - 1:
                out = {node_type: x.relu_() for node_type, x in out.items()}

        return {node_type:self.lin_dict[node_type](x) for node_type, x in out.items()}


class RGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, num_types: int, num_relations: int, num_bases: int, aggr: str = "mean", actn: str = 'relu'):
        super(RGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases, aggr=aggr))
        for _ in range(num_layers-1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases, aggr=aggr))

        self.actn = actn2actfunc[actn]
        self.lin = HeteroLinear(hidden_channels, out_channels, num_types)
    
    def forward(self, x, edge_index, node_type, edge_type):
        out = self.actn(self.convs[0](x, edge_index, edge_type))
        for i in range(1, len(self.convs) - 1):
            out = self.convs[i](out, edge_index, edge_type)
            if i < len(self.convs) - 1:
                out = self.actn(out)
        
        return self.lin(out, node_type)


# perturbation encoding
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, output_dim: int, p: float, norm: str, actn: str, order: str = 'nd'):
        super(MLPEncoder, self).__init__()
        self.n_layer = len(hidden_dims) - 1
        self.in_dim = in_dim
        
        try:
            actn = actn2actfunc[actn]
        except:
            print(actn)
            raise NotImplementedError

        # input layer
        layers = [nn.Linear(self.in_dim, hidden_dims[0]), actn]
        # hidden layers
        for i in range(self.n_layer):
            layers += self.compose_layer(
                in_dim=hidden_dims[i], out_dim=hidden_dims[i+1], norm=norm, actn=actn, p=p, order=order
            )
        # output layers
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.fc = nn.Sequential(*layers)

    def compose_layer(
        self,
        in_dim: int,
        out_dim: int,
        norm: str,
        actn: nn.Module,
        p: float = 0.0,
        order: str = 'nd'
    ):
        norm2normlayer = {'bn': nn.BatchNorm1d(in_dim), 'ln': nn.LayerNorm(in_dim), None: None, 'None': None}  # because in_dim is only fixed here
        try:
            norm = norm2normlayer[norm]
        except:
            print(norm)
            raise NotImplementedError
        # norm --> dropout or dropout --> norm
        if order == 'nd':
            layers = [norm] if norm is not None else []
            if p != 0:
                layers.append(nn.Dropout(p))
        elif order == 'dn':
            layers = [nn.Dropout(p)] if p != 0 else []
            if norm is not None:
                layers.append(norm)
        else:
            print(order)
            raise NotImplementedError

        layers.append(nn.Linear(in_dim, out_dim))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output


class VAE(nn.Module):
    def __init__(self, vae_encoder_params: dict, hidden_dim: int, latent_dim: int, vae_decoder_params: dict):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = MLPEncoder(**vae_encoder_params)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLPEncoder(**vae_decoder_params)  # while it is an MLPEncoder object, it is actually a decoder

    def encode(self, x):
        h = F.relu(self.encoder(x))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, recon, mu, logvar


# helper functions
def get_str_encoder(str_encoder_name, str_encoder_hparams, embed_dim, atom_dim, use_modality_pretrain=True):
    if str_encoder_name == 'gat':
        str_encoder = models.GraphAttentionNetwork(input_dim=atom_dim, hidden_dims=str_encoder_hparams['gat_hidden_dims']+[embed_dim], edge_input_dim=str_encoder_hparams['gat_edge_input_dim'], num_head=str_encoder_hparams['gat_att_heads'], negative_slope=str_encoder_hparams['gat_negative_slope'], batch_norm=str_encoder_hparams['gat_batch_norm'], activation=str_encoder_hparams['gat_actn'], readout=str_encoder_hparams['gat_readout'])
    elif str_encoder_name == 'gin':        
        str_encoder = models.GraphIsomorphismNetwork(input_dim=atom_dim, hidden_dims=str_encoder_hparams['gin_hidden_dims']+[embed_dim], edge_input_dim=str_encoder_hparams['gin_edge_input_dim'], num_mlp_layer=str_encoder_hparams['gin_num_mlp_layer'], eps=str_encoder_hparams['gin_eps'], batch_norm=str_encoder_hparams['gin_batch_norm'], activation=str_encoder_hparams['gin_actn'], readout=str_encoder_hparams['gin_readout'])
    
    if use_modality_pretrain:
        print('Using pretrained structure encoder')
        pre_trained_path = ENCODER_CKPT_DIR + 'str/GIN_256x4_muv.pt'
        state_dict = torch.load(pre_trained_path, map_location=torch.device('cpu'))
        for k, v in list(state_dict.items()):
            if k.startswith('model.'):
                state_dict[k[len('model.'):]] = v
                del state_dict[k]
            elif not k.startswith('layer'):
                del state_dict[k]
        str_encoder.load_state_dict(state_dict)
    
    return str_encoder


def get_kg_encoder(kg_encoder_name, kg_encoder_hparams, embed_dim, all_kg_data, use_modality_pretrain=True):
    # NOTE: need to pass all_kg_data.metadata() into the instantiation of big model class, and feed that to HAN/HGT's instantiation
    if 'han' in kg_encoder_name:
        kg_encoder = HAN(in_channels=all_kg_data.x_dict['drug'].shape[1], hidden_channels=kg_encoder_hparams['han_hidden_dim'], out_channels=embed_dim, num_layers=kg_encoder_hparams['han_num_layers'], heads=kg_encoder_hparams['han_att_heads'], metadata=all_kg_data.metadata(), negative_slope=kg_encoder_hparams['han_negative_slope'], dropout=kg_encoder_hparams['han_dropout'])
    elif 'hgt' in kg_encoder_name:
        kg_encoder = HGT(in_channels=all_kg_data.x_dict['drug'].shape[1], hidden_channels=kg_encoder_hparams['hgt_hidden_dim'], out_channels=embed_dim, num_layers=kg_encoder_hparams['hgt_num_layers'], num_heads=kg_encoder_hparams['hgt_att_heads'], metadata=all_kg_data.metadata(), group=kg_encoder_hparams['hgt_group'])
    
    if use_modality_pretrain:
        print('Using pretrained KG encoder')
        pre_trained_path = ENCODER_CKPT_DIR + 'kg/hgt_best.pt'
        kg_encoder.load_state_dict(torch.load(pre_trained_path, map_location=torch.device('cpu')))
    
    return kg_encoder


def get_tabular_mod_encoder(mod_encoder_name, mod_encoder_hparams, embed_dim, use_modality_pretrain=True, mod="cv"):
    assert mod_encoder_name == 'mlp'
    mod_encoder = MLPEncoder(mod_encoder_hparams['cv_input_dim'], mod_encoder_hparams['cv_mlp_hidden_dims'], embed_dim, mod_encoder_hparams['cv_mlp_dropout'], mod_encoder_hparams['cv_mlp_norm'], mod_encoder_hparams['cv_mlp_actn'], mod_encoder_hparams['cv_mlp_order'])
    
    if use_modality_pretrain:
        print(f'Using pretrained {mod} encoder')
        pre_trained_path = ENCODER_CKPT_DIR + f'{mod}/{mod}_model_ae.pt'
        mod_encoder.load_state_dict(torch.load(pre_trained_path, map_location=torch.device('cpu')))
        
    return mod_encoder


def _resolve_tx_adapter(pretrained_model_ckpt):
    """Path of the transcriptomics adapter checkpoint."""
    path = ENCODER_CKPT_DIR + f"tx/{pretrained_model_ckpt}"
    if not os.path.exists(path):
        path = ENCODER_CKPT_DIR + "tx/tx_adapter.pt"
    return path


def get_tx_encoder(tx_encoder_name, tx_encoder_hparams, embed_dim, use_modality_pretrain=True):
    if tx_encoder_name == 'chemcpa':
        hparams=tx_encoder_hparams["model"]["hparams"]
        additional_params=tx_encoder_hparams["model"]["additional_params"]
        append_ae_layer=tx_encoder_hparams["model"]["append_ae_layer"]
        pretrained_model_ckpt=tx_encoder_hparams["model"]["pretrained_model_ckpt"]
        use_drugs=tx_encoder_hparams["model"]["use_drugs"]
        
        drug_metadata = pd.read_pickle(DATA_DIR+"drug_features/drug_metadata_ddi.pkl")
        smiles = drug_metadata['canonical_smiles'].values
        df = pd.read_parquet(DATA_DIR+"drug_features/tx/embeddings/rdkit2D_embeddings_combined_all_normalized.parquet")
        emb = torch.from_numpy(df.loc[smiles].values).float()
        emb = torch.nn.Embedding.from_pretrained(emb, freeze=True)
        
        if not use_modality_pretrain:
            tx_encoder = TxAdaptingComPert(
                num_genes=TX_INPUT_DIM,
                num_drugs=drug_metadata.shape[0],
                covariate_names_unique={"cell_iname":CELL_LINES_CAPITALIZED},
                **additional_params,
                hparams=hparams,
                drug_embeddings=emb,
                append_layer_width=None,
                use_drugs=use_drugs,
                disable_adv=True,
            )
            cell_lines_lowercase = np.array([cell_line.lower() for cell_line in CELL_LINES_CAPITALIZED])
            
        else:
            print('Using pretrained TX encoder')
            (
                state_dict,
                _,
                cov_embeddings_state_dicts,
                model_config,
                history,
            ) = torch.load(_resolve_tx_adapter(pretrained_model_ckpt), map_location=torch.device('cpu'))
            assert model_config['use_drugs'] == use_drugs, 'Need determine whether to use drug representations or not in chemCPA'
            assert len(cov_embeddings_state_dicts) == 1
            for key in list(state_dict.keys()):
                # remove all components which we will train from scratch, disable (adversary) or initialize explicitly (drug embeddings)
                if key.startswith("adversary_") or key == "drug_embeddings.weight":
                    state_dict.pop(key)
            for k, v in list(additional_params.items()):
                if k in model_config.keys():
                    additional_params[k] = model_config[k]
            tx_encoder = TxAdaptingComPert(
                num_genes=TX_INPUT_DIM,
                num_drugs=drug_metadata.shape[0],
                covariate_names_unique=model_config["covariate_names_unique"],
                **additional_params,
                hparams=model_config["hparams"],
                drug_embeddings=emb,
                append_layer_width=None,
                use_drugs=use_drugs,
                disable_adv=True,
            )
        
            incomp_keys = tx_encoder.load_state_dict(state_dict, strict=False)
            for embedding, state_dict in zip(
                tx_encoder.covariates_embeddings, cov_embeddings_state_dicts
            ):
                embedding.load_state_dict(state_dict)
            incomp_keys_info = {
                "Missing keys": incomp_keys.missing_keys,  # NOTE: We expect to miss `covariates_embeddings.0.weight` here, as it is loaded with the for loop after the `tx_encoder.load_state_dict` call
                "Unexpected_keys": incomp_keys.unexpected_keys,  # NOTE: NOTHING should be here
            }
            print(
                "INCOMP_KEYS (make sure these contain what you expected):\n%s",
                pformat(incomp_keys_info, indent=4, width=1),
            )
            cell_lines_lowercase = np.array([cell_line.lower() for cell_line in model_config["covariate_names_unique"]["cell_iname"]])

        return tx_encoder, cell_lines_lowercase
    
    elif tx_encoder_name == 'mlp':
        tx_encoder = MLPEncoder(tx_encoder_hparams['tx_input_dim'], tx_encoder_hparams['tx_mlp_hidden_dims'], embed_dim, tx_encoder_hparams['tx_mlp_dropout'], tx_encoder_hparams['tx_mlp_norm'], tx_encoder_hparams['tx_mlp_actn'], tx_encoder_hparams['tx_mlp_order'])
        return tx_encoder


### Transformer Fusion
class TransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_tx_bottlenecks, transformer_num_layers, transformer_att_heads, transformer_head_dim, transformer_ffn_dim, transformer_dropout = 0.1, transformer_actn = 'relu', transformer_norm_first = False, transformer_batch_first = True, transformer_agg='mean'):
        super(TransformerFusion, self).__init__()
        """
        Transformer class that takes in our sequence of tokens and does attention. If transformer_agg='cls', concatenates the CLS token which is a learned parameter initialized from a normal distribution. Note that PyTorch's MHA directly splits the input into heads, so we need to pass the raw embedding from encoders through a linear layer to get the correct dimensions (n_heads * head_dim). Finally, we map the output of the Transformer back to the original embedding dimension.

        :param: hidden_dim: intermediate hidden dim
        :param: num_transformer_layers: number of transformer layers
        Reference: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        """
        self.batch_first = transformer_batch_first
        self.norm_first = transformer_norm_first
        self.latent_dim = transformer_head_dim * transformer_att_heads
        self.embed2latent = nn.Linear(embed_dim, self.latent_dim)  # linearly maps the embedding to the correct dimensions for MHA (n_heads * head_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=transformer_att_heads, dim_feedforward=transformer_ffn_dim, dropout=transformer_dropout, activation=transformer_actn, norm_first=transformer_norm_first, batch_first=transformer_batch_first)  # Sometimes we set batch_first to False to disable "fast path" (which leads to NestedTensor that can't be hooked) and enable the use of hook    
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers, enable_nested_tensor=False)  # NOTE: MUST set `enable_nested_tensor=False` or else it would lead to (1) truncation of output when one token is masked across batch, and (2) NestedTensor that can't be hooked.  See: https://discuss.pytorch.org/t/transformerencoder-src-key-padding-mask-truncates-output-dimension-when-some-token-positions-are-masked-across-batch/175316.
        self.latent2embed = nn.Linear(self.latent_dim, embed_dim)  # linearly maps the output of the Transformer back to the original embedding dimension
        self.transformer_agg = transformer_agg
        if self.transformer_agg == 'x-attn':
            # Zorro-style cross attention pooling, see https://github.com/lucidrains/zorro-pytorch/blob/main/zorro_pytorch/zorro_pytorch.py#L317 and https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder.forward
            self.x_attn_kv_norm = nn.LayerNorm(self.latent_dim)
            self.x_attn_query_norm = nn.LayerNorm(self.latent_dim)
            self.x_attn_mha_layer = nn.MultiheadAttention(
                embed_dim=self.latent_dim, 
                num_heads=transformer_att_heads, 
                dropout=transformer_dropout, 
                batch_first=transformer_batch_first
            )
            self.x_attn_dropout = nn.Dropout(transformer_dropout)
            self.x_attn_query = nn.Parameter(torch.randn(1, self.latent_dim))  # query vector for cross-attention
            self.x_attn_key_padding_mask = torch.zeros(1, NUM_MODALITIES + num_tx_bottlenecks, dtype=torch.bool)
            # READOUT_MASK_MISSING=off lets the readout also attend to missing-modality placeholder tokens.
            # Not stored in checkpoints: set it identically for training and inference.
            self.readout_mask_missing = os.getenv("READOUT_MASK_MISSING", "on").lower() not in {"off", "0", "false", "no"}
            if num_tx_bottlenecks > 0:
                # Tokens read by the cross-attention query; layout [non-TX, bottleneck, TX], True = excluded.
                # Not stored in checkpoints: READOUT_MODE must be set identically for training and inference.
                readout_mode = os.getenv("READOUT_MODE", "all")
                if readout_mode == "nontx_bt":      # non-TX + bottleneck
                    self.x_attn_key_padding_mask[:, -len(CELL_LINES):] = True
                elif readout_mode == "all":         # all tokens (default)
                    pass
                elif readout_mode == "nontx_tx":    # non-TX + TX, exclude the bottleneck tokens
                    self.x_attn_key_padding_mask[:, NUM_NON_TX_MODALITIES:NUM_NON_TX_MODALITIES + num_tx_bottlenecks] = True
                elif readout_mode == "nontx":       # non-TX only (exclude bottleneck + TX)
                    self.x_attn_key_padding_mask[:, NUM_NON_TX_MODALITIES:] = True
                elif readout_mode == "bt":          # bottleneck tokens only
                    self.x_attn_key_padding_mask[:, :NUM_NON_TX_MODALITIES] = True
                    self.x_attn_key_padding_mask[:, -len(CELL_LINES):] = True
                else:
                    raise ValueError(f"unknown READOUT_MODE: {readout_mode}")

        # adapted from https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
        def patch_attention(m):
            forward_orig = m.forward

            def wrap(*args, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False

                return forward_orig(*args, **kwargs)

            m.forward = wrap

        patch_attention(self.transformer_encoder.layers[-1].self_attn)  # NOTE: patching the attention layer to return the attention weights, for the use of hook in analyses
        
    def forward(self, fusion_sequence, fusion_mask, src_mask=None):
        # fusion_sequence is of size (batch_size, seq, hdim)
        batch_size = fusion_sequence.shape[0]  # grab batch size

        if not self.batch_first:
            fusion_sequence = fusion_sequence.transpose(0, 1)

        ## transformer encoder
        fusion_sequence = self.embed2latent(fusion_sequence)
        fusion_sequence = self.transformer_encoder(src = fusion_sequence, src_key_padding_mask = fusion_mask, mask = src_mask)
        
        if self.transformer_agg != 'x-attn':
            fusion_sequence = self.latent2embed(fusion_sequence)

        if self.transformer_agg == 'cls':
            if not self.batch_first:
                out = fusion_sequence[0, :, :]
            else:
                out = fusion_sequence[:, 0, :]  # returning the CLS embedding
        elif self.transformer_agg == 'x-attn':
            x_attn_query = self.x_attn_query.repeat(batch_size, 1, 1)
            if not self.batch_first:
                x_attn_query = x_attn_query.transpose(0, 1)
            x_attn_key_padding_mask = self.x_attn_key_padding_mask.repeat(batch_size, 1).to(fusion_mask.device)
            if self.readout_mask_missing:  # "read available only": exclude this sample's missing modalities from the readout
                x_attn_key_padding_mask = x_attn_key_padding_mask | fusion_mask
            # A fully masked readout row (e.g. tx_tx evaluation with a non-TX-only readout) would give an
            # undefined softmax; fall back to the mode mask, which always leaves at least one token.
            _all_masked = x_attn_key_padding_mask.all(dim=1)
            if _all_masked.any():
                _base_mode_mask = self.x_attn_key_padding_mask.repeat(batch_size, 1).to(fusion_mask.device)
                x_attn_key_padding_mask = x_attn_key_padding_mask.clone()
                x_attn_key_padding_mask[_all_masked] = _base_mode_mask[_all_masked]
            fusion_sequence = self.x_attn_kv_norm(fusion_sequence)
            if self.norm_first:
                x_attn_query = self.x_attn_query_norm(x_attn_query)
            out = self.x_attn_mha_layer(
                query=x_attn_query,
                key=fusion_sequence,
                value=fusion_sequence,
                key_padding_mask=x_attn_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )[0]  # returns [1, batch_size, feat_dim]
            out = self.x_attn_dropout(out)
            out = out + x_attn_query
            if not self.norm_first:
                out = self.x_attn_query_norm(out)
            out = self.latent2embed(out)[0, :, :]
        elif self.transformer_agg == 'mean':  # NOTE: take only the mean of the non-masked tokens
            if not self.batch_first:
                fusion_sequence = fusion_sequence.transpose(0, 1)
            out = scatter_mean(fusion_sequence.reshape(-1, fusion_sequence.shape[-1]), ((torch.arange(fusion_mask.shape[0])+1).unsqueeze(-1).to(fusion_mask.device) * (~fusion_mask).long()).reshape(-1), dim=0)[1:]
        elif self.transformer_agg == 'max':
            if not self.batch_first:
                fusion_sequence = fusion_sequence.transpose(0, 1)
            out = scatter_max(fusion_sequence.reshape(-1, fusion_sequence.shape[-1]), ((torch.arange(fusion_mask.shape[0])+1).unsqueeze(-1).to(fusion_mask.device) * (~fusion_mask).long()).reshape(-1), dim=0)[0][1:]
        else:
            raise NotImplementedError

        return out


### Unimodal projector
class MLPAdaptor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, output_dim: int, p: float, norm: str, actn: str, order: str = 'nd'):
        super(MLPAdaptor, self).__init__()
        self.n_layer = len(hidden_dims) - 1
        self.in_dim = in_dim
        
        try:
            actn = actn2actfunc[actn]
        except:
            print(actn)
            raise NotImplementedError

        # input layer
        layers = [nn.Linear(self.in_dim, hidden_dims[0]), actn]
        # hidden layers
        for i in range(self.n_layer):
            layers += self.compose_layer(
                in_dim=hidden_dims[i], out_dim=hidden_dims[i+1], norm=norm, actn=actn, p=p, order=order
            )
        # output layers
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.fc = nn.Sequential(*layers)

    def compose_layer(
        self,
        in_dim: int,
        out_dim: int,
        norm: str,
        actn: nn.Module,
        p: float = 0.0,
        order: str = 'nd'
    ):
        norm2normlayer = {'bn': nn.BatchNorm1d(in_dim), 'ln': nn.LayerNorm(in_dim), None: None, 'None': None}  # because in_dim is only fixed here
        try:
            norm = norm2normlayer[norm]
        except:
            print(norm)
            raise NotImplementedError
        # norm --> dropout or dropout --> norm
        if order == 'nd':
            layers = [norm] if norm is not None else []
            if p != 0:
                layers.append(nn.Dropout(p))
        elif order == 'dn':
            layers = [nn.Dropout(p)] if p != 0 else []
            if norm is not None:
                layers.append(norm)
        else:
            print(order)
            raise NotImplementedError

        layers.append(nn.Linear(in_dim, out_dim))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output


### Decoder
class Symmetric(nn.Module):
    def forward(self, W):
        return W.triu() + W.triu(1).transpose(-1, -2)

class BilinearDDIScorer(nn.Bilinear):
    def __init__(self, input_dim1: int, input_dim2: int, output_dim: int):
        """
        Extend torch.nn.Bilinear to accommodate our need of outputting a tensor of shape (num_label, input_n1, input_n2) (note that for binary, num_label=1). No bias term is used.
        :param: input_dim1: Drug feature dimension after fusion, left
        :param: input_dim2: Drug feature dimension after fusion, right
        :param: output_dim: Number of labels (for binary, output_dim = 1)
        Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Bilinear
        """
        super(BilinearDDIScorer, self).__init__(in1_features=input_dim1, in2_features=input_dim2, out_features=output_dim)

    def bilinear(self, input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor):
        # weight is [num_labels, dim1, dim2]. broadcasting applies to the first dim.
        return torch.matmul(torch.matmul(input1, weight), input2.T)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, label_range: tuple = None):
        if label_range is not None:
            assert len(label_range) == 2
            out = self.bilinear(input1, input2, self.weight[label_range[0]:label_range[1], :, :])
        else:
            out = self.bilinear(input1, input2, self.weight)
        return out


### Position Encoding
class PositionEncodingSinusoidal(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 19, num_tx_bottlenecks: int = 0, transformer_agg: str = 'cls'):
        """
        Sinusoidal positional encoding for Transformer. They are added to the input embeddings before dropout.
        When num_attention_bottlenecks is not None, the positional encoding is only added to str, kg, cv, tx-bottleneck tokens. Otherwise, it is added to all tokens.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # swapping batch_size and seq_len
        pe = torch.transpose(pe, 0, 1)
        
        input_seq_len = NUM_MODALITIES + num_tx_bottlenecks
        if transformer_agg == 'cls':
            input_seq_len += 1
        if max_len < input_seq_len:
            pe_modified = torch.zeros((1, input_seq_len, d_model))
            pe_modified[:, :max_len, :] = pe
            pe = pe_modified
            
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)
    
    
class PositionEncodingLearnable(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 19, num_tx_bottlenecks: int = 0, transformer_agg: str = 'cls'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(1, self.max_len, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x[:, :self.max_len, :] += self.pe
        return self.dropout(x)


### Madrigal Model
class MadrigalEncoder(nn.Module):
    def __init__(self, all_kg_data, feat_dim, str_encoder_name, str_encoder_hparams, kg_encoder_name, kg_encoder_hparams, cv_encoder_name, cv_encoder_hparams, tx_encoder_name, tx_encoder_hparams, num_tx_bottlenecks, pos_emb_dropout, transformer_fusion_hparams, proj_hparams, fusion='transformer_uni_proj', str_node_feat_dim=MOL_DIM, use_modality_pretrain=True, normalize=False, pos_emb_type='learnable', adapt_before_fusion=False, **kwargs):
        """
        :param: all_kg_data: For HAN/HGT, a `HeterData` object. For RGCN, a `Data` object.
        :param: feat_dim: The dimension of the desired sembeddings.
        """
        super(MadrigalEncoder, self).__init__()
        
        self.embed_dim = feat_dim
        self.fusion = fusion
        self.normalize = normalize
        self.adapt_before_fusion = adapt_before_fusion
        self.use_tx_basal = kwargs.get('use_tx_basal', False)  # NOTE: this is only experimented for contrastive learning, not for downstream tasks
        
        # Initializing components
        # str encoder
        self.str_encoder = get_str_encoder(str_encoder_name, str_encoder_hparams, self.embed_dim, str_node_feat_dim, use_modality_pretrain)
        
        # kg encoder
        self.kg_encoder_name = kg_encoder_name
        self.kg_encoder = get_kg_encoder(kg_encoder_name, kg_encoder_hparams, self.embed_dim, all_kg_data, use_modality_pretrain)
        
        # get tabular mod encoder
        self.cv_encoder = get_tabular_mod_encoder(cv_encoder_name, cv_encoder_hparams, self.embed_dim, use_modality_pretrain, mod="cv")
        
        tab_mod_encoder_hparams_dict = kwargs.get('tab_mod_encoder_hparams_dict', None)
        self.tabular_mod_encoders = nn.ModuleDict()
        if tab_mod_encoder_hparams_dict is not None and len(tab_mod_encoder_hparams_dict) > 1:
            for mod, mod_encoder_hparams in tab_mod_encoder_hparams_dict.items():
                if mod == 'cv':
                    continue
                self.tabular_mod_encoders[mod] = get_tabular_mod_encoder(cv_encoder_name, mod_encoder_hparams, self.embed_dim, False, mod=mod)

        # get tx encoder
        if tx_encoder_name == 'mlp':
            self.tx_encoder_dict = nn.ModuleDict({cell_line : get_tx_encoder(tx_encoder_name, tx_encoder_hparams, self.embed_dim, use_modality_pretrain=False) for cell_line in CELL_LINES})
        elif tx_encoder_name == 'chemcpa':
            self.tx_encoder_dict = None
            self.tx_encoder, cell_lines = get_tx_encoder(tx_encoder_name, tx_encoder_hparams, self.embed_dim, use_modality_pretrain)
            self.tx_cell_line_onehot_encoder = OneHotEncoder(sparse=False)
            self.tx_cell_line_onehot_encoder.fit(cell_lines.reshape(-1, 1))  # NOTE: The cell line orders in modality availability and here (for cell line embeddings in chemCPA) are different, but it doesn't matter since these two components are separate
        else:
            raise NotImplementedError
        
        # enable using attention bottleneck
        self.num_tx_bottlenecks = num_tx_bottlenecks
        # SRC_MASK=off removes the non-TX<->TX attention block (ablation; set identically for training and inference).
        self.use_src_mask = os.getenv("SRC_MASK", "on").lower() not in {"off", "0", "false", "no"}
        self.transformer_agg = transformer_fusion_hparams['transformer_agg']
        self.transformer_att_heads = transformer_fusion_hparams['transformer_att_heads']
        self.tx_pe = env_flag_is_true("TX_PE")
        pos_emb_max_len = 0
        if self.num_tx_bottlenecks == 0:
            if not self.tx_pe:
                pos_emb_max_len += NUM_NON_TX_MODALITIES
            else:
                pos_emb_max_len += NUM_MODALITIES
        else:
            # add pos enc to the non-TX modalities; optionally also to the bottleneck tokens.
            # seq order is [non-TX(3), bottleneck(K), TX(16)], so extending max_len by K gives each
            # bottleneck token a distinct positional code (breaks their permutation symmetry).
            pos_emb_max_len += NUM_NON_TX_MODALITIES
            if env_flag_is_true("BT_POS_ENC"):
                pos_emb_max_len += self.num_tx_bottlenecks
        if self.transformer_agg == 'cls':
            pos_emb_max_len += 1
        if self.num_tx_bottlenecks > 0:
            self.tx_bottleneck_tokens = nn.Parameter(torch.randn(self.num_tx_bottlenecks, self.embed_dim))

        # Positional encoding
        if pos_emb_type == 'learnable':
            self.pos_encoder = PositionEncodingLearnable(
                d_model=self.embed_dim, 
                dropout=pos_emb_dropout, 
                max_len=pos_emb_max_len, 
                num_tx_bottlenecks=num_tx_bottlenecks, 
                transformer_agg=self.transformer_agg
            )
        elif pos_emb_type == 'sinusoidal':
            self.pos_encoder = PositionEncodingSinusoidal(
                d_model=self.embed_dim, 
                dropout=pos_emb_dropout, 
                max_len=pos_emb_max_len, 
                num_tx_bottlenecks=num_tx_bottlenecks, 
                transformer_agg=self.transformer_agg
            )
        else:
            raise NotImplementedError
        
        # Fusion
        self.transformer = TransformerFusion(
            self.embed_dim, 
            self.num_tx_bottlenecks,
            **transformer_fusion_hparams,
        )
        if self.transformer_agg == 'cls':
            self.cls = nn.Parameter(torch.randn(1, self.embed_dim))  # CLS token is added to the input sequence before the linear mapping

        # Unimodal projection (in CL and before fusion)
        self.uni_projector = MLPAdaptor(self.embed_dim, proj_hparams['proj_hidden_dims'], self.embed_dim, proj_hparams['proj_dropout'], proj_hparams['proj_norm'], proj_hparams['proj_actn'], proj_hparams['proj_order'])
        
        # Unimodal fuser (as fusion method for CL)
        if fusion == 'transformer_uni_proj':
            self.uni_fuser = MLPAdaptor(self.embed_dim, proj_hparams['proj_hidden_dims'], self.embed_dim, proj_hparams['proj_dropout'], proj_hparams['proj_norm'], proj_hparams['proj_actn'], proj_hparams['proj_order'])

    def encode(self, batch_drugs, batch_masks, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=False, **kwargs):
        ## Encoders
        # structure
        str_out_raw = self.str_encoder(batch_mols, batch_mols.node_feature.float())
        str_out = str_out_raw["graph_feature"]
        
        # kg
        kg_data = batch_kg['data']
        kg_drug_index_map = batch_kg['drug_index_map']
        
        if 'han' in self.kg_encoder_name or 'hgt' in self.kg_encoder_name:
            # NOTE: The output of this encoder is more than the drug nodes needed for us, but we don't need to slice it since it will be masked out later
            kg_out_valid = self.kg_encoder(kg_data.x_dict, kg_data.edge_index_dict)['drug']
        elif 'rgcn' in self.kg_encoder_name:
            kg_out_valid = self.kg_encoder(kg_data.node_embeddings, kg_data.edge_index, kg_data.node_type, kg_data.edge_type)[:(kg_data.node_type==0).sum().item()]  # NOTE: The first (==0) `drug_num` nodes are drugs
        
        # pad drugs not in KG with dummy embeddings here because they will be masked out later
        kg_out = torch.randn((max(batch_drugs.max().item()+1, kg_drug_index_map.max().item() + 1), self.embed_dim), device=kg_out_valid.device)
        kg_out[kg_drug_index_map] = kg_out_valid
        kg_out = kg_out[batch_drugs]
        
        assert torch.isin(batch_drugs[batch_masks[:, 1] == 0], kg_drug_index_map).all()
        
        # cv
        cv_out = self.cv_encoder(batch_cv)
        
        # other tabular mods
        other_tabular_mod_out = []
        if len(self.tabular_mod_encoders) > 0:
            for mod in NON_TX_MODALITIES[3:]:
                batch_mod = kwargs.get(mod, None)
                assert batch_mod is not None, f"Missing {mod} in the input batch"
                other_tabular_mod_out.append(self.tabular_mod_encoders[mod](batch_mod))
        
        # tx
        if self.tx_encoder_dict is not None:
            tx_out_all_cell_lines = [self.tx_encoder_dict[cell_line](batch_tx_dict[cell_line]['sigs']) for cell_line in CELL_LINES]
        else:
            all_sigs = torch.cat([batch_tx_dict[cell_line]['sigs'] for cell_line in CELL_LINES], dim=0)
            all_drugs = torch.cat([batch_tx_dict[cell_line]['drugs'] for cell_line in CELL_LINES], dim=0)
            all_dosages = torch.cat([batch_tx_dict[cell_line]['dosages'] for cell_line in CELL_LINES], dim=0)
            all_cell_lines = np.concatenate([batch_tx_dict[cell_line]['cell_lines'] for cell_line in CELL_LINES], axis=0)
            all_cell_lines_oh = torch.from_numpy(self.tx_cell_line_onehot_encoder.transform(all_cell_lines.reshape(-1, 1))).long().to(all_sigs.device)
            _, _, tx_out_cell_line = self.tx_encoder.predict(
                genes=all_sigs,  # [batch_size * num_cells, num_genes]
                drugs_idx=all_drugs,  # didn't the change the naming in chemcpa, but their drugs_idx correpsonds to our drugs, while their drugs refer to one-hot encodings
                dosages=all_dosages,  # [batch_size * num_cells]
                covariates=[all_cell_lines_oh],
                return_latent_basal=self.use_tx_basal,
                return_latent_treated=(not self.use_tx_basal),
            )
            tx_out_all_cell_lines = list(torch.split(tx_out_cell_line, split_size_or_sections=tx_out_cell_line.shape[0]//len(CELL_LINES), dim=0))
        
        # join all
        all_embeds = [str_out, kg_out, cv_out] + other_tabular_mod_out + tx_out_all_cell_lines
        
        ## process before fusion
        all_embeds = torch.stack(all_embeds, dim=1)  # [batch_size, num_modalities, embed_dim]
        if self.adapt_before_fusion and (not raw_encoder_output):
            all_embeds = self.uni_projector(all_embeds)
        
        if not raw_encoder_output:
            if self.fusion in {'transformer_uni_proj', 'transformer'}:
                if self.fusion == 'transformer_uni_proj':
                    # Since we feed unimodal into MLP and multimodal into transformer, we need to separate the embeddings and feed into fusion & projector separately
                    assert torch.all((~batch_masks).sum(dim = 1) > 0)
                    observed_multimodal_bool = (~batch_masks).sum(dim=1) > 1
                    batch_masks_fusion = batch_masks[observed_multimodal_bool]
                    batch_masks_uni = batch_masks[~observed_multimodal_bool]
                    unimodal_mod_indices = torch.where(batch_masks_uni==0)[1]  # NOTE: This line is correct only because each row only has one 0 (unimodal) in batch_masks_uni. 
                    
                    # [BATCH_SIZE, NUM_MODALITY (SEQ_LEN), INPUT_DIM]
                    fusion_sequence = all_embeds[observed_multimodal_bool, :, :]
                
                else:
                    batch_masks_fusion = batch_masks
                    fusion_sequence = all_embeds
                
                src_mask = None
                
                # apply attention bottleneck to ones that require fusion
                if self.num_tx_bottlenecks > 0:
                    fusion_sequence = torch.cat([
                        fusion_sequence[:, :NUM_NON_TX_MODALITIES, :],
                        self.tx_bottleneck_tokens.repeat(fusion_sequence.shape[0], 1, 1),
                        fusion_sequence[:, NUM_NON_TX_MODALITIES:, :],
                    ], dim=1)
                    tx_bottleneck_masks = torch.zeros(self.num_tx_bottlenecks, dtype=torch.bool, device=batch_masks_fusion.device).unsqueeze(0).repeat(batch_masks_fusion.shape[0], 1)  # NOTE: TX_BT tokens will always be there (even none of the cell lines are observed)
                    batch_masks_fusion = torch.cat([
                        batch_masks_fusion[:, :NUM_NON_TX_MODALITIES],
                        tx_bottleneck_masks,
                        batch_masks_fusion[:, NUM_NON_TX_MODALITIES:],
                    ], dim=1)
                    
                    src_mask = torch.zeros((fusion_sequence.shape[1], fusion_sequence.shape[1]), dtype=torch.bool, device=fusion_sequence.device)
                    if self.use_src_mask:  # SRC_MASK=off leaves this all-False (full attention, no bottleneck block)
                        mask_submat = torch.ones((NUM_NON_TX_MODALITIES, len(CELL_LINES)), dtype=torch.bool, device=fusion_sequence.device)
                        src_mask[:mask_submat.shape[0], -mask_submat.shape[1]:] = mask_submat  # lower left
                        src_mask[-mask_submat.shape[1]:, :mask_submat.shape[0]] = mask_submat.T  # upper right
                
                if self.transformer_agg == 'cls':
                    fusion_sequence = torch.cat([
                        self.cls.repeat(fusion_sequence.shape[0], 1, 1),
                        fusion_sequence,
                    ], dim=1)
                    batch_masks_fusion = torch.cat([
                        torch.zeros((batch_masks_fusion.shape[0], 1), dtype=torch.bool, device=batch_masks_fusion.device),
                        batch_masks_fusion,
                    ], dim=1)
                    
                    if self.num_tx_bottlenecks > 0:
                        # NOTE: CLS only takes value/key from non-tx and tx bottlenecks, BUT, it MUST attend to all (otherwise, some tokens could be nan)
                        # Even when we use MAX aggregation, there must be some placeholder [CLS] token that attends to all, otherwise, some tokens could be nan
                        cls_mask_row = torch.zeros((1, src_mask.shape[1]), dtype=torch.bool, device=fusion_sequence.device)
                        src_mask = torch.cat([
                            cls_mask_row,
                            src_mask,
                        ], dim=0)
                        cls_mask_col = torch.zeros((src_mask.shape[0], 1), dtype=torch.bool, device=fusion_sequence.device)
                        src_mask = torch.cat([
                            cls_mask_col,
                            src_mask,
                        ], dim=1)
                
                # NOTE: Only to experiment with the case where the TX_BT tokens do not attend to other tokens (k/v), but just take input from them (query them). Since some sample have all of their TX modalities will be missing, assigning mask separately for each sample is potentially more accurate, where the samples missing TX modalities completely will have the TX_BT tokens not querying any TX tokens.
                
                if self.normalize:
                    fusion_sequence = F.normalize(fusion_sequence, p=2, dim=-1)
                
                pos_enc_sequence = self.pos_encoder(fusion_sequence)  # position encoding
                z_fusion = self.transformer(pos_enc_sequence, fusion_mask=batch_masks_fusion, src_mask=src_mask)

                if self.fusion == 'transformer_uni_proj':
                    ## Unimodal "fusion" (projection)
                    uni_embeds = all_embeds[~observed_multimodal_bool, unimodal_mod_indices, :]
                    if self.normalize:
                        uni_embeds = F.normalize(uni_embeds, p=2, dim=-1)
                    z_uni = self.uni_fuser(uni_embeds)

                    ## Concatenate z_fusion and z_uni back per index
                    z = torch.empty_like(str_out, device=str_out.device)
                    z[observed_multimodal_bool] = z_fusion
                    z[~observed_multimodal_bool] = z_uni
                
                else:
                    z = z_fusion

            elif self.fusion == 'mean':
                if self.normalize:
                    all_embeds = F.normalize(all_embeds, p=2, dim=-1)
                z = scatter_mean(all_embeds.reshape(-1, all_embeds.shape[-1]), ((torch.arange(batch_masks.shape[0])+1).unsqueeze(-1).to(batch_masks.device) * (~batch_masks).long()).reshape(-1), dim=0)[1:]  # get mean of correct entries in all_embeds for each sample in the batch based on modality availability (~mask)
                
            elif self.fusion == 'add':
                if self.normalize:
                    all_embeds = F.normalize(all_embeds, p=2, dim=-1)
                z = scatter_add(all_embeds.reshape(-1, all_embeds.shape[-1]), ((torch.arange(batch_masks.shape[0])+1).unsqueeze(-1).to(batch_masks.device) * (~batch_masks).long()).reshape(-1), dim=0)[1:]
            
            else:
                raise NotImplementedError
        
        torch.cuda.empty_cache()
        
        
        if raw_encoder_output:
            uni_embeds = all_embeds[~batch_masks, :]
            if self.normalize:
                uni_embeds = F.normalize(uni_embeds, p=2, dim=-1)
            z = self.uni_projector(uni_embeds)

        return z

    def forward(self, batch_drugs, batch_masks, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=False, **kwargs):
        return self.encode(batch_drugs, batch_masks, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output, **kwargs)
        

    
    
class MadrigalMultilabel(nn.Module):
    def __init__(self, encoder, feat_dim, prediction_dim, prediction_dim_single_drug=None, normalize=False, use_single_drug=False):
        super(MadrigalMultilabel, self).__init__()
        self.encoder = encoder
        self.embed_dim = feat_dim
        self.normalize = normalize
        self.use_single_drug = use_single_drug
        self.decoder = BilinearDDIScorer(input_dim1 = self.embed_dim, input_dim2 = self.embed_dim, output_dim = prediction_dim)
        nn.utils.parametrize.register_parametrization(self.decoder, 'weight', Symmetric())

    def forward(self, batch_head, batch_tail, batch_head_mod_masks, batch_tail_mod_masks, batch_kg, label_range=None, single_drug=False):
        head_drugs = batch_head['drugs']
        head_mol_strs = batch_head['strs']
        head_cv = batch_head['cv']
        head_tx_all_cell_lines = batch_head['tx']
        head_masks = batch_head_mod_masks
        
        tail_drugs = batch_tail['drugs']
        tail_mol_strs = batch_tail['strs']
        tail_cv = batch_tail['cv']
        tail_tx_all_cell_lines = batch_tail['tx']
        tail_masks = batch_tail_mod_masks
        
        z_head = self.encoder(head_drugs, head_masks, head_mol_strs, batch_kg, head_cv, head_tx_all_cell_lines)
        z_tail = self.encoder(tail_drugs, tail_masks, tail_mol_strs, batch_kg, tail_cv, tail_tx_all_cell_lines)
        if self.normalize:
            z_head = F.normalize(z_head)  # same as z_head / torch.norm(z_head, dim=1, keepdim=True)
            z_tail = F.normalize(z_tail)
        
        pred_scores = self.decoder(z_head, z_tail, label_range)  # [num_labels, num_heads, num_tails]
        
        return pred_scores
    
