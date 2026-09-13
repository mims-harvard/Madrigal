import torch
import torch.nn as nn
import torch.nn.functional as F
from madrigal.utils import get_loss_fn, get_model, PROJECT_DIR
from seml.config import generate_configs
from seml.config import read_config

class MultiHeadAttentionStack(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, dropout_rate=0.1):
        super(MultiHeadAttentionStack, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        # Optional: Layer normalization can be added for each layer
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(num_layers)])

    def forward(self, x, key_padding_mask=None):
        attn_output_weights = []
        for i, attn_layer in enumerate(self.layers):
            x, attn_weight = attn_layer(x, x, x, key_padding_mask=key_padding_mask)
            # Apply dropout and normalization if necessary
            x = self.dropout(x)
            x = self.norm_layers[i](x)
            attn_output_weights.append(attn_weight)
        return x, attn_output_weights


class MadrigalLM(nn.Module):
    def __init__(
        self,
        encoder,
        lm_model,
        drug_project_dim,
        text_project_dim,
        mlp_dim,
        dropout,
        self_att,
        num_heads=4,
        num_att_layers=1,
        normalize=False,
    ):
        super(MadrigalLM, self).__init__()

        assert drug_project_dim == text_project_dim

        self.encoder = encoder
        self.normalize = normalize
        self.self_att = self_att

        if lm_model == "mistralai/Mistral-7B-v0.1":
            lm_emb_dim = 4096
        else:
            lm_emb_dim = 768

        self.drug_project = nn.Sequential(
            nn.Linear(128, drug_project_dim), nn.SiLU(), nn.Dropout(dropout)
        )

        self.text_project = nn.Sequential(
            nn.Linear(lm_emb_dim, text_project_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        
        
        
        if self.self_att:
            self.multihead_attn = nn.MultiheadAttention(drug_project_dim, num_heads)
            

        self.out_mlp = nn.Sequential(
            nn.Linear(text_project_dim + 2 * drug_project_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(
        self,
        batch_head,
        batch_tail,
        batch_head_mod_masks,
        batch_tail_mod_masks,
        batch_kg,
        embeddings,
    ):

        head_drugs = batch_head["drugs"]
        head_mol_strs = batch_head["strs"]
        head_cv = batch_head["cv"]
        head_tx_all_cell_lines = batch_head["tx"]
        head_masks = batch_head_mod_masks

        tail_drugs = batch_tail["drugs"]
        tail_mol_strs = batch_tail["strs"]
        tail_cv = batch_tail["cv"]
        tail_tx_all_cell_lines = batch_tail["tx"]
        tail_masks = batch_tail_mod_masks

        z_head = self.encoder(
            head_drugs,
            head_masks,
            head_mol_strs,
            batch_kg,
            head_cv,
            head_tx_all_cell_lines,
        )
        z_tail = self.encoder(
            tail_drugs,
            tail_masks,
            tail_mol_strs,
            batch_kg,
            tail_cv,
            tail_tx_all_cell_lines,
        )

        if self.normalize:
            z_head = F.normalize(z_head)
            z_tail = F.normalize(z_tail)

        z_head = self.drug_project(z_head)
        z_tail = self.drug_project(z_tail)
        z_text = self.text_project(embeddings)

        if self.self_att:
            all_out = torch.stack([z_text, z_head, z_tail], dim=0)
            
                
            attn_output, attn_output_weights = self.multihead_attn(
                all_out, all_out, all_out
            )
            out = self.out_mlp(
                torch.cat([attn_output[0], attn_output[1], attn_output[2]], dim=-1)
            )
        else:
            out = self.out_mlp(
                torch.cat([z_text, z_head, z_tail], dim=-1)
            )

        return out


def get_loss_fn(
    loss_fn_name, pos_weight, task="binary", loss_readout="mean", device="cuda"
):
    pos_weight = torch.tensor([pos_weight])
    if loss_fn_name == "bce":
        # NOTE: note that multiclass task can also use BCE loss with negative sampling
        loss_fn = nn.BCELoss(reduction=loss_readout)
    elif loss_fn_name == "bce_with_weight":
        loss_fn = nn.BCEWithLogitsLoss(
            reduction=loss_readout, pos_weight=pos_weight.to(device)
        )
    elif loss_fn_name == "ce" and task == "multiclass":
        loss_fn = nn.CrossEntropyLoss(reduction=loss_readout)
    else:
        raise NotImplementedError(
            "Loss function {} not implemented for task {}".format(loss_fn_name, task)
        )
    return loss_fn


def get_full_model(args, train_collator, all_kg_data):

    feature_dim = 128
    str_encoder = "gin"
    gin_edge_input_dim = 18
    gin_batch_norm = True
    kg_encoder = "hgt"
    hgt_num_layers = 2
    cv_encoder = "mlp"
    tx_encoder = "chemcpa"
    tx_chemcpa_config_path = PROJECT_DIR+"configs/chemcpa/chemcpa_finetune_configs.yaml"

    fusion = "transformer_uni_proj"
    num_attention_bottlenecks = 2

    loss_fn_name = "bce"
    task = "multiclass"
    evaluate_interval = 20
    finetune_mode = "str_random_sample"
    test = True
    normalize = False
    decoder_normalize = False
    device = torch.device("cuda")

    str_encoder_hparams = {
        "gin_hidden_dims": [128, 128, 128],
        "gin_edge_input_dim": 18,
        "gin_num_mlp_layer": 3,
        "gin_eps": 0,
        "gin_batch_norm": gin_batch_norm,
        "gin_actn": "relu",
        "gin_readout": "mean",
    }
    kg_encoder_hparams = {
        "hgt_hidden_dim": 128,
        "hgt_num_layers": 2,
        "hgt_att_heads": 4,
        "hgt_group": "sum",
    }

    cv_encoder_hparams = {
        "cv_input_dim": train_collator.cv_df.shape[0],
        "cv_mlp_hidden_dims": [512, 256],
        "cv_mlp_dropout": 0.2,
        "cv_mlp_norm": None,
        "cv_mlp_actn": "relu",
        "cv_mlp_order": "nd",
    }
    
    tab_mod_encoder_hparams_dict = {}
    tab_mod_encoder_hparams_dict["cv"] = cv_encoder_hparams

    _, _, experiment_config = read_config(tx_chemcpa_config_path)
    configs = generate_configs(experiment_config)
    assert len(configs) == 1
    tx_encoder_hparams = configs[0]

    proj_hparams = {
        "proj_hidden_dims": [512, 512],
        "proj_dropout": 0.2,
        "proj_norm": "ln",
        "proj_actn": "relu",
        "proj_order": "nd",
    }

    transformer_fusion_hparams = {
        "transformer_num_layers": 3,
        "transformer_att_heads": 4,
        "transformer_head_dim": 128,
        "transformer_ffn_dim": 512,
        "transformer_dropout": 0.2,
        "transformer_actn": "gelu",
        "transformer_norm_first": False,
    }

    transformer_fusion_hparams["transformer_batch_first"] = False
    transformer_fusion_hparams["transformer_agg"] = "x-attn"

    if args.use_pretrained:
        checkpoint_path = args.pretrained_path

    encoder, encoder_configs = get_model(
        all_kg_data=all_kg_data,
        feature_dim=128,
        prediction_dim=158,
        str_encoder_name=str_encoder,
        str_encoder_hparams=str_encoder_hparams,
        kg_encoder_name="hgt",
        kg_encoder_hparams=kg_encoder_hparams,
        cv_encoder_name="mlp",
        cv_encoder_hparams=cv_encoder_hparams,
        tx_encoder_name="chemcpa",
        tx_encoder_hparams=tx_encoder_hparams,
        num_attention_bottlenecks=2,
        pos_emb_type="learnable",
        pos_emb_dropout=0.2,
        transformer_fusion_hparams=transformer_fusion_hparams,
        proj_hparams=proj_hparams,
        fusion=fusion,
        normalize=False,
        decoder_normalize=False,
        checkpoint_path=checkpoint_path,
        frozen=False,
        device=torch.device("cuda"),
        encoder_only=True,
        finetune_mode="str_random_sample",
        str_node_feat_dim=train_collator.str_node_feat_dim,
        logger=None,
        use_modality_pretrain=args.use_pretrained,
        adapt_before_fusion=False,
        use_pretrained_adaptor=True,
        tab_mod_encoder_hparams_dict=tab_mod_encoder_hparams_dict,
    )

    model = MadrigalLM(
        encoder,
        args.lm_model,
        args.drug_project_dim,
        args.text_project_dim,
        args.mlp_dim,
        args.dropout,
        args.self_att,
        args.num_heads,
        args.num_layers,
        args.normalize,
    ).to(device)
    

    loss_fn = get_loss_fn(loss_fn_name=args.loss, pos_weight=args.pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd
    )

    return model, loss_fn, optimizer
