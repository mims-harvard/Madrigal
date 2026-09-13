import math, os, time, wandb
from datetime import datetime
import numpy as np


import torch

from madrigal.parse_args import create_parser, get_hparams
from madrigal.models.simclr import SimCLR_Madrigal
from madrigal.evaluate.evaluate import evaluate_final_embeds, evaluate_pt, stacked_inst_dist_topk_accuracy
from madrigal.evaluate.eval_utils import save_embeds
from madrigal.data.data import get_pretrain_data
from madrigal.evaluate.eval_utils import draw_umap_plot
from madrigal.utils import (
    NON_TX_MODALITIES,
    set_seed,
    AverageMeter,
    ProgressMeter,
    pretrain_modality_subset_sampler,
    save_checkpoint,
    get_root_logger,
    get_str_encoder_hparams,
    get_kg_encoder_hparams,
    get_cv_encoder_hparams,
    get_tx_encoder_hparams,
    get_transformer_fusion_hparams,
    get_proj_hparams,
    get_model,
    LARS,
    adjust_learning_rate,
    to_device,
    SEED,
)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(train_loader, model, all_train_subset_masks, hard_negative_mask, optimizer, epoch, pretrain_mode, unbalanced, all_extra_molecules, extra_mol_str_masks, extra_mol_num, hparams, wandb, logger, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        logger,
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    for i, batch_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_drug_indices, batch_data = to_device(batch_data, device)
        
        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, hparams['pretrain_lr'], hparams['warmup_epochs'], hparams['pretrain_num_epochs'])
        learning_rates.update(lr)
        
        # Get two (subset sampling) masks for each drug in the batch.  The mask is fed straight to the encoder, so it must already be aligned with the drugs.
        batch_mask1, batch_mask2 = pretrain_modality_subset_sampler([all_train_subset_masks[drug_ind] for drug_ind in batch_drug_indices.tolist()], pretrain_mode=pretrain_mode, unbalanced=unbalanced)
        batch_mask1, batch_mask2 = to_device(batch_mask1, device), to_device(batch_mask2, device)
        if hard_negative_mask is not None:
            batch_hard_negative_mask = to_device(hard_negative_mask[batch_drug_indices.tolist(), :][: , batch_drug_indices.tolist()], device)
        else:
            batch_hard_negative_mask = None
            
        # Get extra negative molecules for the batch (if applicable)
        if all_extra_molecules is not None and extra_mol_num > 0:
            batch_extra_mols = all_extra_molecules[np.random.choice(all_extra_molecules.shape[0], extra_mol_num, replace=False)]
            batch_extra_mols = to_device(batch_extra_mols, device)
        else:
            batch_extra_mols = None
        
        optimizer.zero_grad()
        
        # compute output
        _, _, (logits, labels, loss) = model(batch_drug_indices.to(device), batch_mask1, batch_mask2, batch_hard_negative_mask, batch_data, batch_extra_mols, extra_mol_str_masks)
        losses.update(loss.item(), len(batch_drug_indices))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
            with torch.no_grad():
                top1, top5 = stacked_inst_dist_topk_accuracy(logits, labels, topk=(1, 5))

        wandb.log({"train_loss": loss.item()}, step=epoch)
        wandb.log({'batch train top1 (CL-head)': top1.item(), 'batch train top5 (CL-head)': top5.item()}, step=epoch)  # Since in a batch the different drugs can be using different modality subsets (in `double_random` or `str_center` settings), we might see pretty stochastic results here.


# Adapted from https://github.com/facebookresearch/moco-v3/blob/main
def adjust_moco_momentum(epoch, hparams):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / hparams['pretrain_num_epochs'])) * (1. - hparams['moco_m'])
    return m


def main(args, hparams, wandb, logger, output_dir, device = DEVICE):
    logger.info("Loading data...")
    train_drugs, val_drugs, pretrain_drugs, train_loader, val_loader, pretrain_loader, collator, masks, all_train_subset_masks, hard_negative_mask, all_extra_molecules, extra_mol_str_masks, kg_args = get_pretrain_data(args, hparams)
    # masks: numpy.ndarray, dtype = int (all drugs)
    # hard_negative_mask: numpy.ndarray, dtype = bool (all drugs)
    # all_train_subset_masks: list of torch.tensor, dtype = int (only train CL drugs)
    
    # create model
    logger.info("Creating model '{}'".format('SimCLR_Madrigal'))
    
    str_encoder_hparams = get_str_encoder_hparams(args, hparams)
    kg_encoder_hparams = get_kg_encoder_hparams(args, hparams)
    tab_mod_encoder_hparams_dict = {}
    for mod in NON_TX_MODALITIES[2:]:
        tab_mod_encoder_hparams_dict[mod] = get_cv_encoder_hparams(args, hparams, collator.tabular_mod_stores[mod].shape[-1])
    tx_encoder_hparams = get_tx_encoder_hparams(args, hparams, list(collator.tx_store_dict.values())[0]['sigs'].shape[-1])
    proj_hparams = get_proj_hparams(hparams)
    transformer_fusion_hparams = get_transformer_fusion_hparams(args, hparams)
    
    encoder, encoder_configs = get_model(
        all_kg_data=collator.kg_data, 
        feature_dim=args.feature_dim,
        prediction_dim=None,
        str_encoder_name=args.str_encoder, 
        str_encoder_hparams=str_encoder_hparams, 
        kg_encoder_name=args.kg_encoder,
        kg_encoder_hparams=kg_encoder_hparams,
        cv_encoder_name=args.cv_encoder,
        cv_encoder_hparams=tab_mod_encoder_hparams_dict["cv"],
        tx_encoder_name=args.tx_encoder,
        tx_encoder_hparams=tx_encoder_hparams,
        num_attention_bottlenecks=args.num_attention_bottlenecks,
        pos_emb_type=args.pos_emb_type,
        pos_emb_dropout=args.pos_emb_dropout,
        transformer_fusion_hparams=transformer_fusion_hparams,
        proj_hparams=proj_hparams, 
        fusion=args.fusion,
        normalize=args.normalize,
        decoder_normalize=None,
        checkpoint_path=None,
        frozen=None,
        device=device,
        encoder_only=True,
        finetune_mode=None,
        str_node_feat_dim=collator.str_node_feat_dim,
        logger=logger,
        use_modality_pretrain=args.use_modality_pretrain,
        use_tx_basal=args.use_tx_basal,
        tab_mod_encoder_hparams_dict=tab_mod_encoder_hparams_dict,
    )
    
    model = SimCLR_Madrigal(encoder, hparams['feature_dim'], hparams['moco_mlp_dim'], hparams['moco_t'], raw_encoder_output=hparams['raw_encoder_output'], shared_predictor=hparams['shared_predictor'])
    model.to(device)
    logger.info(model)
    
    # infer learning rate before changing batch size
    args.pretrain_lr = args.pretrain_lr * args.pretrain_batch_size / 512

    if args.pretrain_optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=hparams['pretrain_lr'], weight_decay=hparams['pretrain_wd'], momentum=hparams['pretrain_momentum'])
    elif args.pretrain_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['pretrain_lr'], weight_decay=hparams['pretrain_wd'], eps=hparams['pretrain_eps'], betas=(hparams['pretrain_beta1'], 0.999))
    elif args.pretrain_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams['pretrain_lr'], weight_decay=hparams['pretrain_wd'], momentum=hparams['pretrain_momentum'], nesterov=hparams['pretrain_nesterov'], dampening=hparams['pretrain_dampening'])


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.pretrain_start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    logger.info('Start pretraining')
    
    if args.save_checkpoints != 0:
        logger.info('Saving embeddings before pretraining')
        save_dir = output_dir + 'before_pretrain/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_outputs, val_outputs = save_embeds(model.base_encoder, train_drugs, val_drugs, masks, collator, save_dir, device, raw_encoder_output=hparams['raw_encoder_output'])

    param_track = {}
    wandb.watch(model.base_encoder, log='all', log_freq=500)
    for epoch in range(args.pretrain_start_epoch, args.pretrain_num_epochs):
        logger.info('Epoch: {}'.format(epoch))
        
        train_epoch(train_loader, model, all_train_subset_masks, hard_negative_mask, optimizer, epoch, args.pretrain_mode, args.pretrain_unbalanced, all_extra_molecules, extra_mol_str_masks, args.extra_str_neg_mol_num, hparams, wandb, logger, device)
        

        if (args.save_checkpoints != 0) and (epoch % args.save_checkpoints == 0) and (epoch > 0):
            model.eval()
            for name, param in model.named_parameters():
                param_track.setdefault(name, []).append(param.data.abs().max().item())
            
            logger.info('evaluate_pt train')
            all_train_outputs = evaluate_pt(model, train_drugs, masks, hard_negative_mask, collator, 'train', wandb, logger, device, epoch)
            logger.info('evaluate_pt val')
            all_val_outputs = evaluate_pt(model, val_drugs, masks, hard_negative_mask, collator, 'val', wandb, logger, device, epoch)
            
            logger.info('save checkpoint')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'encoder_configs': encoder_configs,
                'kg_args': kg_args,
            }, is_best=False, filename=output_dir+f'checkpoint_{epoch}.pt')
            
            # NOTE: Segmentation faults might occur here.
            logger.info('draw_umap_plot train')
            draw_umap_plot({indices_str:output['embeds'] for indices_str, output in all_train_outputs.items()}, None, {indices:output['drugs'].tolist() for indices, output in all_train_outputs.items()}, None, None, None, 'train_epoch', wandb, None, logger, epoch=epoch, raw_encoder_output=hparams['raw_encoder_output'])
            logger.info('draw_umap_plot val')
            draw_umap_plot({indices_str:output['embeds'] for indices_str, output in all_val_outputs.items()}, None, {indices:output['drugs'].tolist() for indices, output in all_val_outputs.items()}, None, None, None, 'val_epoch', wandb, None, logger, epoch=epoch, raw_encoder_output=hparams['raw_encoder_output'])

    if args.save_checkpoints != 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'encoder_configs': encoder_configs,
            'kg_args': kg_args,
        }, is_best=False, filename=output_dir+f'/checkpoint_{epoch}.pt')
        
    logger.info('Finished pretraining... Saving embeddings...')
    save_dir = output_dir + f'/after_pretrain_{epoch}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_outputs, val_outputs = save_embeds(model.base_encoder, train_drugs, val_drugs, masks, collator, save_dir, device, raw_encoder_output=hparams['raw_encoder_output'])

    
    logger.info('Evaluating final embeddings... Would likely take half an hour or so...')
    evaluate_final_embeds(train_outputs, val_outputs, save_dir, wandb, logger, epoch)
    

if __name__ == '__main__':
    args = create_parser('pretrain')
    hparams = get_hparams(args, 'pretrain')
    wandb.init(
        project='pretrain_debug' if args.debug else f'pretrain_{args.data_source}_{args.split_method}', 
        dir=args.save_dir,
        mode='offline' if args.debug else 'online',
        config=hparams,
    )
    wandb.run.name = args.run_name if args.run_name is not None else (wandb.run.name or wandb.run.id)
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = f"{args.save_dir}/{cur_time}_{wandb.run.name}/"
    if os.path.exists(output_dir):
        output_dir = f"{args.save_dir}/{cur_time}_{wandb.run.name}_{os.getpid()}/"
    os.makedirs(output_dir)
    logger = get_root_logger(f'{output_dir}/log.txt')
    
    logger.info("Args: {}".format(args))
    logger.info("hparams: {}".format(hparams))
    logger.info("wandb: {}".format(wandb.run.name))
    logger.info("log_dir_path: {}".format(output_dir))
    
    main(args, hparams, wandb, logger, output_dir)
    
