import argparse, os, yaml
from pathlib import Path
from .utils import DATA_DIR, BASE_DIR, PROJECT_DIR

def create_parser(stage: str):
    parser = argparse.ArgumentParser(description='Madrigal')
    
    # Common args
    parser.add_argument('--from_yaml', type=str, default=None, help='whether to load args from yaml file')
    parser.add_argument('--debug', action='store_true', help='whether to run in debug mode or not')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--data_source', type=str, default='TWOSIDES', help='DDI data source', choices=['ONSIDES', 'TWOSIDES', 'DrugBank'])
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the best model and results')
    parser.add_argument('--path_base', type=str, default=None, help='Data directory')
    parser.add_argument('--split_method', type=str, default='split_by_triplets', choices=['split_by_drugs_random', 'split_by_drugs_atc', 'split_by_drugs_targets', 'split_by_drugs_taxonomy', 'split_by_pairs', 'split_by_triplets', "split_by_drugs_twosides_matched_random"])
    parser.add_argument('--feature_dim', type=int, default=128, help='input feature dimension to transformer (i.e. output feature dimension of view encoders, position embedder, tx bottlenecks, and CLS embedder)')
    parser.add_argument('--use_modality_pretrain', action='store_true', help='whether to use first stage pretraining')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # Structural encoder args
    parser.add_argument('--str_encoder', type=str, default='gin', help='which structural encoder to use', choices=['gat', 'gin', 'grover', 'transformer-m'])
    
    parser.add_argument('--gat_hidden_dims', type=int, nargs='+', default=[128, 128, 128], help='GAT hidden dimensions')
    parser.add_argument('--gat_edge_input_dim', type=int, default=18, help='GAT edge input dimension, should be 18 if we use the default edge feature')
    parser.add_argument('--gat_att_heads', type=int, default=4, help='GAT attention heads')
    parser.add_argument('--gat_negative_slope', type=float, default=0.2, help='GAT negative slope')
    parser.add_argument('--gat_batch_norm', action='store_true', help='whether to use batch norm in GAT')
    parser.add_argument('--gat_actn', type=str, default='relu', help='GAT activation', choices=['relu', 'leaky_relu', 'selu', 'gelu', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--gat_readout', type=str, default='mean', help='GAT readout', choices=['sum', 'mean'])

    parser.add_argument('--gin_hidden_dims', type=int, nargs='+', default=[128, 128, 128], help='GIN hidden dimensions')  # or [256, 256, 256, 256]
    parser.add_argument('--gin_edge_input_dim', type=int, default=18, help='GIN edge input dimension, should be 18 if we use the default edge feature')
    parser.add_argument('--gin_num_mlp_layer', type=int, default=3, help='number of MLP layers (AGG) for each GIN layer')
    parser.add_argument('--gin_eps', type=float, default=0, help='GIN initial eps')
    parser.add_argument('--gin_batch_norm', action='store_true', help='whether to use batch norm in GIN')
    parser.add_argument('--gin_actn', type=str, default='relu', help='GIN activation', choices=['relu', 'leaky_relu', 'selu', 'gelu', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--gin_readout', type=str, default='mean', help='GIN readout', choices=['sum', 'mean'])

    # KG encoder args
    parser.add_argument('--kg_encoder', type=str, default='hgt', help='which structural encoder to use', choices=[
        'hgt',  # HGT (the underlying KG is the same as HAN w/o metapaths)
    ])
    parser.add_argument('--kg_sampling_num_neighbors', type=int, default=None, help='number of neighbors to sample for each node for each edge type in the KG')
    parser.add_argument('--kg_sampling_num_layers', type=int, default=None, help='layers to sample for each node for each edge type in the KG')
    
    parser.add_argument('--han_att_heads', type=int, default=4, help='HAN attention heads')
    parser.add_argument('--han_num_layers', type=int, default=1, help='HAN number of layers')
    parser.add_argument('--han_hidden_dim', type=int, default=128, help='HAN hidden dimension')
    parser.add_argument('--han_negative_slope', type=float, default=0.2, help='HAN negative slope')
    parser.add_argument('--han_dropout', type=float, default=0.2, help='HAN dropout')
    
    parser.add_argument('--hgt_num_layers', type=int, default=2, help='HGT number of layers')
    parser.add_argument('--hgt_hidden_dim', type=int, default=128, help='HGT hidden dimension')
    parser.add_argument('--hgt_att_heads', type=int, default=4, help='HGT num att heads')
    parser.add_argument('--hgt_group', type=str, default='sum', help='HGT group', choices=['sum', 'mean', 'max'])
    
    # Cv encoder args
    parser.add_argument('--cv_encoder', type=str, default='mlp', help='which cv encoder to use', choices=['mlp'])
    parser.add_argument("--cv_mlp_hidden_dims", type=int, nargs='+', default=[512, 256], help="cv encoder hidden dims")
    parser.add_argument("--cv_mlp_dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--cv_mlp_norm", type=str, default=None, help="normalization layer", choices=['bn', 'ln', None])
    parser.add_argument("--cv_mlp_actn", type=str, default='relu', help="activation type", choices=['relu', 'gelu', 'selu', 'leakyrelu', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument("--cv_mlp_order", type=str, default='nd', help="order of normalization and dropout")
    
    # Ts encoder args
    parser.add_argument('--tx_encoder', type=str, default='chemcpa', help='which tx encoder to use', choices=['mlp', 'chemcpa'])
    
    parser.add_argument("--tx_chemcpa_config_path", type=str, default=None, help="chemCPA config path")
    
    parser.add_argument("--tx_mlp_hidden_dims", type=int, nargs='+', default=[512, 256], help="tx encoder hidden dims")
    parser.add_argument("--tx_mlp_dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--tx_mlp_norm", type=str, default=None, help="normalization layer", choices=['bn', 'ln', None])
    parser.add_argument("--tx_mlp_actn", type=str, default='relu', help="activation type", choices=['relu', 'gelu', 'selu', 'leakyrelu', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument("--tx_mlp_order", type=str, default='nd', help="order of normalization and dropout")

    # fusion args
    parser.add_argument('--fusion', type=str, default='transformer', help='which fusion to use', choices=['concat_mlp', 'add', 'transformer', 'transformer_uni_proj', 'mean']) 
    parser.add_argument('--normalize', action='store_true', help='whether or not normalize the modality embeddings before feeding into transformer')  # NOTE: This `normalize` is different from `decoder_normalize` which is for normalizing inputs to the bilinear decoder
    
    # position embedder args
    parser.add_argument('--pos_emb_type', type=str, default='learnable', help='position embedder type', choices=['learnable', 'sinusoidal'])
    parser.add_argument('--pos_emb_dropout', type=float, default=0.2, help='position embedder dropout')

    # Transformer fusion args
    parser.add_argument('--transformer_att_heads', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--transformer_head_dim', type=int, default=128, help='Transformer head dimensions')
    parser.add_argument('--transformer_num_layers', type=int, default=3, help='Transformer number of layers')
    parser.add_argument('--transformer_ffn_dim', type=int, default=512, help='Transformer feed forward net dimension')
    parser.add_argument('--transformer_dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--transformer_actn', type=str, default='gelu', help='Transformer activation function', choices=['relu', 'gelu'])
    parser.add_argument('--transformer_norm_first', action='store_true', help='Transformer norm first')
    parser.add_argument('--transformer_batch_first', action='store_true', help='Transformer batch first, this affects whether PyTorch would use `fast path` or not, and thus impacts training efficiency.')
    parser.add_argument('--transformer_not_batch_first', action='store_true', help='Wrapper for Transformer batch first.')
    parser.add_argument('--transformer_agg', type=str, default='x-attn', help='aggregation of token representations for the sequence representation', choices=['mean', 'max', 'cls', 'x-attn'])
    parser.add_argument('--num_attention_bottlenecks', type=int, default=0, help='number of attention bottleneck in transformer fusion, 0 = no bottleneck')

    # Unimodal projector args
    parser.add_argument("--proj_hidden_dims", type=int, nargs='+', default=[512, 512], help="projector hidden dims")
    parser.add_argument("--proj_dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--proj_norm", type=str, default='ln', help="normalization layer", choices=['bn', 'ln', None])
    parser.add_argument("--proj_actn", type=str, default='relu', help="activation type", choices=['relu', 'gelu', 'selu', 'leakyrelu', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument("--proj_order", type=str, default='nd', help="order of normalization and dropout")


    parser.add_argument('--evaluate_interval', type=int, default=10, help='evaluate interval')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--not_drop_last', action='store_true', help='whether to not drop last batch in dataloader')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epoch num')
    parser.add_argument('--repeat', type=str, default=None, help='if not None, will load dataset based on split_method and the specified repeat index (like {split_method}_{repeat}), so that we can do multiple runs of the same type of split')

    if stage == 'train':
        # Training args
        parser.add_argument('--loss_fn_name', type=str, default='bce', help='loss function', choices=['bce', 'focal', 'hinge', 'kepler', 'ce'])  # Only BCE is implemented.
        parser.add_argument('--task', type=str, default='multilabel', help='Classification task', choices=['binary', 'multiclass', 'multilabel'])
        parser.add_argument('--num_epochs', type=int, default=600, help='epoch num')
        parser.add_argument('--batch_size', type=int, default=None, help='batch size')
        parser.add_argument('--num_negative_samples_per_pair', type=int, default=None, help='number of negative samples per positive sample during training')  # NOTE: None here means using fixed negatives, i.e. negatives are not sampled during training on-the-fly.
        parser.add_argument('--negative_sampling_probs_type', type=str, default='uniform', help='negative sampling probability distribution type', choices=['uniform', 'degree', 'degree_w2v'])
        
        parser.add_argument('--structure_encoder_lr', type=float, default=1e-4, help='learning rate for structure encoder')
        parser.add_argument('--kg_encoder_lr', type=float, default=1e-4, help='learning rate for KG encoder')
        parser.add_argument('--perturb_encoders_lr', type=float, default=1e-4, help='learning rate for cv/ts encoders')
        parser.add_argument('--fusion_lr', type=float, default=1e-4, help='learning rate for transformer encoder fusion')
        parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for bilinear decoder')
        
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for AdamW')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for AdamW')
        parser.add_argument('--wd', type=float, default=1e-2, help='weight decay for AdamW')
        parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for RAdam and AdamW')
        
        parser.add_argument('--loss_readout', type=str, default='mean', help='readout for loss', choices=['mean', 'sum'])
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer', choices=['radam', 'adamw'])
        parser.add_argument('--checkpoint', type=str, default=None)
        parser.add_argument('--finetune_mode', type=str, default='str_random_sample', help='Conduct subset sampling in training', choices=[
            'ablation_str_str', 
            'ablation_kg_kg_subset', 
            'ablation_kg_kg_padded', 
            'ablation_cv_cv_padded', 
            'ablation_tx_tx_padded',
            'ablation_str_random_str+kg_full_sample', 
            'ablation_str_random_str+cv_full_sample',
            'ablation_str_random_str+tx_full_sample',
            'ablation_str_random_str+kg+cv_full_sample', 
            'ablation_str_random_str+kg+tx_full_sample',
            'ablation_str_random_str+cv+tx_full_sample',
            'str_full', 
            'full_full', 
            'double_random', 
            'str_random_sample', 
            'str_str+random_sample', 
            'full_str+random_sample'
        ])  # NOTE: 'ablation_str_str', 'ablation_kg_kg's, 'ablation_cv_cv's, 'ablation_str_random_str+kg_full_sample', 'ablation_str_random_str+cv_full_sample', 'ablation_str_random_str+kg+cv_full_sample' (i.e. str_random but treat each drug's full as only str+kg) are ablations; anything with 'padded' are baselines.
        parser.add_argument('--test', action='store_true', help='Report test performance')
        parser.add_argument('--no_test', action='store_true', help='Not report test performance')
        parser.add_argument('--frozen', action='store_true', help='Freeze the encoder')
        parser.add_argument('--intermediate_figs_savedir', type=str, default=None, help='Directory for saving intermediate figures')
        parser.add_argument('--decoder_normalize', action='store_true', help='normalize the decoder input')
        parser.add_argument('--train_with_str_str', action='store_true', help='whether to train with str-str pairs')
        parser.add_argument('--group_tx_in_sampling', action='store_true', help='In *_random_sample finetune modes, treat all TX cell lines as a SINGLE meta-modality during modality subsampling (a sampled subset includes either all of a drug\'s available TX cell lines or none).')
        parser.add_argument('--adapt_before_fusion', action='store_true', help='whether to adapt the view embeddings before fusion')
        parser.add_argument('--use_pretrained_adaptor', action='store_true', help='whether to use pretrain adaptor for adaptation of modality encodings before fusion')
        
        parser.add_argument("--dataset_ratio", type=str, default="1_1_1", help="dataset ratio for TWOSIDES, DrugBank, and ONSIDES_OFFSIDES")
        parser.add_argument("--use_drugbank", action='store_true', help="whether to use DrugBank dataset")
        parser.add_argument("--use_single_drug", action='store_true', help="whether to use ONSIDES-OFFISDES dataset")
        parser.add_argument("--loss_ratio_single_drug", type=float, default=10., help="loss ratio for single drug")

        args = parser.parse_args()
        args = process_args(args, 'train')
    
    elif stage == 'pretrain':
        parser.add_argument('--pretrain_loss_func', type=str, default='infonce', help='pretraining loss function', choices=['triplet_margin', 'infonce']) 
        parser.add_argument('--save_checkpoints', type=int, default=100, help='Save checkpoints per this many epochs')
        parser.add_argument('--str_sim_threshold', type=float, default=0.95, help='Threshold for structure similarity (jaccard), above which a pair of drugs won\'t be negative samples mutually')
        parser.add_argument('--kg_sim_threshold', type=float, default=0.95, help='Threshold for KG similarity (cosine similarity), above which a pair of drugs won\'t be negative samples mutually')
        parser.add_argument('--perturb_sim_threshold', type=float, default=0.95, help='Threshold for perturbation similarity (pearson correlation), above which a pair of drugs won\'t be negative samples mutually')
        parser.add_argument('--too_hard_neg_mask', action='store_true', help='whether or not to mask highly similar negative samples in the loss function')
        parser.add_argument('--extra_str_neg_mol_num', default=0, type=int, help='Number of extra negative samples to be sampled from the structure database (ChEMBL)') 
        parser.add_argument('--shared_predictor', action='store_true', help='whether or not to share the predictor between two views')
        parser.add_argument('--use_tx_basal', action='store_true', help='whether or not to use tx basal instead of after adding cell line embeddings')
        parser.add_argument('--raw_encoder_output', action='store_true', help='whether or not to use raw encoder output instead of the output after projection')

        # Pretraining args
        parser.add_argument('--pretrain_num_epochs', default=5000, type=int, help='number of total epochs to run')
        parser.add_argument('--pretrain_start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
        
        parser.add_argument('--pretrain_lr', default=1e-4, type=float, help='initial (base) learning rate')
        parser.add_argument('--pretrain_str_encoder_lr', default=1e-4, type=float, help='initial (base) learning rate for structure encoder')
        parser.add_argument('--pretrain_kg_encoder_lr', default=1e-4, type=float, help='initial (base) learning rate for KG encoder')
        parser.add_argument('--pretrain_perturb_encoder_lr', default=1e-4, type=float, help='initial (base) learning rate for ts/cv encoders')
        
        parser.add_argument('--pretrain_wd', default=1e-2, type=float, help='weight decay (default: 1e-6)')
        parser.add_argument('--pretrain_eps', default=1e-8, type=float, help='epsilon (for AdamW)')
        parser.add_argument('--pretrain_beta1', type=float, default=0.9, help='beta1 for AdamW', choices=[0.1, 0.5, 0.9])
        parser.add_argument('--pretrain_beta2', type=float, default=0.999, help='beta2 for AdamW', choices=[0.9, 0.999])
        parser.add_argument('--pretrain_momentum', default=0.9, type=float, help='momentum (for SGD/LARS)')
        
        parser.add_argument('--pretrain_batch_size', default=1000, type=int)
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--pretrain_optimizer', default='adamw', type=str, choices=['lars', 'adamw', 'radam'], help='optimizer to use')
        parser.add_argument('--pretrain_mode', type=str, default='str_center_uni', choices=['double_random_comb', 'double_random_uni', 'str_center_comb', 'str_center_uni', 'str_kg'])
        parser.add_argument('--pretrain_tx_downsample_ratio', type=float, default=1.0, help='ratio for downsampling tx cell line modality in contrastive pretraining')
        parser.add_argument('--pretrain_unbalanced', action='store_true', help='If true, use unbalanced dataset for pretraining')  # default is balanced (for str_center mode)

        # moco/simclr configs:
        parser.add_argument('--moco_mlp_dim', default=512, type=int, help='hidden dimension in MLPs (default: 4096)')
        parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating momentum encoder (default: 0.99)')
        parser.add_argument('--moco_m_cos', action='store_true', help='gradually increase moco momentum to 1 with a half-cycle cosine schedule')
        parser.add_argument('--moco_t', default=0.1, type=float, help='softmax temperature (default: 1.0)')

        args = parser.parse_args()
        args = process_args(args, 'pretrain')

    return args


def get_hparams(args, stage):
    hparams = {
        'str_encoder': args.str_encoder,
        'kg_encoder': args.kg_encoder,
        'cv_encoder': args.cv_encoder,
        'tx_encoder': args.tx_encoder,
        'feature_dim': args.feature_dim,
        'use_modality_pretrain': args.use_modality_pretrain, 
        
        'gat_hidden_dims': args.gat_hidden_dims,        
        'gat_edge_input_dim': args.gat_edge_input_dim,
        'gat_att_heads': args.gat_att_heads,
        'gat_negative_slope': args.gat_negative_slope,
        'gat_batch_norm': args.gat_batch_norm,
        'gat_actn': args.gat_actn,
        'gat_readout': args.gat_readout,

        'gin_hidden_dims': args.gin_hidden_dims,
        'gin_edge_input_dim': args.gin_edge_input_dim,
        'gin_num_mlp_layer': args.gin_num_mlp_layer,
        'gin_batch_norm': args.gin_batch_norm,
        'gin_actn': args.gin_actn,
        'gin_readout': args.gin_readout,
        'gin_eps': args.gin_eps,
        
        'kg_sampling_num_neighbors': args.kg_sampling_num_neighbors,
        'kg_sampling_num_layers': args.kg_sampling_num_layers,
        
        'han_num_layers': args.han_num_layers,
        'han_att_heads': args.han_att_heads,
        'han_hidden_dim': args.han_hidden_dim,
        'han_negative_slope': args.han_negative_slope,
        'han_dropout': args.han_dropout,
        
        'hgt_hidden_dim': args.hgt_hidden_dim,
        'hgt_att_heads': args.hgt_att_heads,
        'hgt_num_layers': args.hgt_num_layers,
        'hgt_group': args.hgt_group,
        
        'tx_mlp_hidden_dims': args.tx_mlp_hidden_dims,
        'tx_mlp_dropout': args.tx_mlp_dropout,
        'tx_mlp_actn': args.tx_mlp_actn,
        'tx_mlp_norm': args.tx_mlp_norm,
        'tx_mlp_order': args.tx_mlp_order,
        'tx_chemcpa_config_path': args.tx_chemcpa_config_path,
        
        'cv_mlp_hidden_dims': args.cv_mlp_hidden_dims,
        'cv_mlp_dropout': args.cv_mlp_dropout,
        'cv_mlp_actn': args.cv_mlp_actn,
        'cv_mlp_norm': args.cv_mlp_norm,
        'cv_mlp_order': args.cv_mlp_order,
        
        'fusion': args.fusion,
        'pos_emb_type': args.pos_emb_type,
        'pos_emb_dropout': args.pos_emb_dropout,
        'normalize': args.normalize,
        'num_attention_bottlenecks': args.num_attention_bottlenecks,
        'transformer_att_heads': args.transformer_att_heads,
        'transformer_head_dim': args.transformer_head_dim,
        'transformer_num_layers': args.transformer_num_layers,
        'transformer_ffn_dim': args.transformer_ffn_dim,
        'transformer_dropout': args.transformer_dropout,
        'transformer_actn': args.transformer_actn,
        'transformer_norm_first': args.transformer_norm_first,
        'transformer_agg': args.transformer_agg,

        "proj_hidden_dims": args.proj_hidden_dims,
        "proj_dropout": args.proj_dropout,
        "proj_norm": args.proj_norm,
        "proj_actn": args.proj_actn,
        "proj_order": args.proj_order,
        
        "evaluate_interval": args.evaluate_interval,
        "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
    }

    if stage=='train':
        hparams.update({
            'num_epochs': args.num_epochs,
            'checkpoint': args.checkpoint,
            'split_method': args.split_method,
            'finetune_mode': args.finetune_mode,
            'structure_encoder_lr': args.structure_encoder_lr,
            'kg_encoder_lr': args.kg_encoder_lr,
            'perturb_encoders_lr': args.perturb_encoders_lr,
            'fusion_lr': args.fusion_lr,
            'decoder_lr': args.decoder_lr,
            'beta1': args.beta1,
            'beta2': args.beta2,
            'wd': args.wd,
            'eps': args.eps,
            'loss_readout': args.loss_readout,
            'optimizer': args.optimizer,
            'decoder_normalize': args.decoder_normalize,
            'train_with_str_str': args.train_with_str_str,
            'group_tx_in_sampling': args.group_tx_in_sampling,
            'adapt_before_fusion': args.adapt_before_fusion,
            'use_pretrained_adaptor': args.use_pretrained_adaptor,
            
            "dataset_ratio": args.dataset_ratio,
            "use_drugbank": args.use_drugbank,
            "use_single_drug": args.use_single_drug,
            "loss_ratio_single_drug": args.loss_ratio_single_drug,
        })

    elif stage=='pretrain':
        hparams.update({
            'save_checkpoints': args.save_checkpoints,
            'str_sim_threshold': args.str_sim_threshold,
            'kg_sim_threshold': args.kg_sim_threshold,
            'perturb_sim_threshold': args.perturb_sim_threshold,
            'too_hard_neg_mask': args.too_hard_neg_mask,
            'extra_str_neg_mol_num': args.extra_str_neg_mol_num,
            'shared_predictor': args.shared_predictor,
            'use_tx_basal': args.use_tx_basal,
            'raw_encoder_output': args.raw_encoder_output,
            
            'pretrain_mode': args.pretrain_mode,
            'pretrain_num_epochs': args.pretrain_num_epochs,
            'pretrain_lr': args.pretrain_lr,
            'pretrain_wd': args.pretrain_wd,
            'pretrain_eps': args.pretrain_eps,
            'pretrain_beta1': args.pretrain_beta1,
            'pretrain_beta2': args.pretrain_beta2,
            'pretrain_momentum': args.pretrain_momentum,
            'feature_dim': args.feature_dim,
            'pretrain_optimizer': args.pretrain_optimizer,
            'pretrain_tx_downsample_ratio': args.pretrain_tx_downsample_ratio,
            
            'moco_mlp_dim':args.moco_mlp_dim,
            'moco_m':args.moco_m,
            'moco_m_cos':args.moco_m_cos,
            'moco_t':args.moco_t,
        })
    
    else:
        raise NotImplementedError
    
    return hparams


def update_args_with_yaml(args):
    if args.from_yaml is None:
        return
    configs = yaml.safe_load(Path(PROJECT_DIR + args.from_yaml).read_text())
    for arg_name, v in configs.items():
        if arg_name in args.__dict__:
            setattr(args, arg_name, v)
        else:
            raise ValueError(f'arg_name {arg_name} not in args.__dict__')


def process_args(args, stage):
    if args.from_yaml is not None:
        update_args_with_yaml(args)
    
    if args.not_drop_last:
        args.drop_last = False
    else:
        args.drop_last = True
    
    
    # For dirs
    if args.repeat in {'none', 'None'}:
        args.repeat = None
        
    if args.path_base is None:
        args.path_base = DATA_DIR
    if args.save_dir is None and stage=='train':
        if args.repeat is not None:
            args.save_dir = BASE_DIR + f'model_output/{args.data_source}/{args.split_method}/{args.repeat}/'
        else:
            args.save_dir = BASE_DIR + f'model_output/{args.data_source}/{args.split_method}/'
    elif args.save_dir is None and stage=='pretrain':
        if args.repeat is not None:
            args.save_dir = BASE_DIR + f'model_output/pretrain/{args.data_source}/{args.split_method}/{args.repeat}/'
        else:
            args.save_dir = BASE_DIR + f'model_output/pretrain/{args.data_source}/{args.split_method}/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        
    # Correspondence between different datasets & splits
    if args.data_source == 'DrugBank':
        assert stage=='pretrain' or args.task == 'multiclass'
    elif args.data_source == 'TWOSIDES':
        assert stage=='pretrain' or args.task == 'multilabel'
        
    # Tx encoder
    if args.tx_encoder == 'chemcpa':
        assert args.tx_chemcpa_config_path is not None
    
    # If in test mode (need to collect final embeds and attention weights), don't use batch first
    if stage == 'train' and not args.no_test:
        args.test = True
    if not args.transformer_not_batch_first:
        args.transformer_batch_first = True
    if stage == 'train' and args.test:
        args.transformer_batch_first = False

    # Different pretrain modes
    if stage=='pretrain' and args.pretrain_mode == 'double_random':
        args.pretrain_unbalanced = True
        
    if args.kg_sampling_num_neighbors is not None:
        if 'han' in args.kg_encoder:
            args.kg_sampling_num_layers = args.han_num_layers
        elif 'hgt' in args.kg_encoder:
            args.kg_sampling_num_layers = args.hgt_num_layers

    return args
