import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import wandb
from madrigal.evaluate.metrics import get_metrics_binary
from madrigal.utils import to_device
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from data import get_collators, get_dataloaders, get_datasets
from embeddings import get_embeddings_labels, get_embeddings_labels_batched
from model import get_full_model

from madrigal.utils import DATA_DIR, BASE_DIR

kg_encoder = 'hgt'
data_source = 'DrugBank'
repeat = None
kg_sampling_num_neighbors = None
kg_sampling_num_layers = None
num_workers = 0
device = 'cuda'


def main(args):
    project_name = f'lm_{data_source}_{args.split_method}_{repeat}'
    if args.use_wandb:
        run = wandb.init(
            project=project_name, 
            dir='./',
            mode='online'
        )
    else:
        run=None

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        if args.use_wandb:
            save_dir = run.name
            if not os.path.exists(os.path.join(args.save_path, save_dir)):
                os.makedirs(os.path.join(args.save_path, save_dir))
        
    if args.generate_embeddings:
        if args.paraphrase:
            embedding_dir = get_embeddings_labels_batched(args.lm_model, args.paraphrase, args.use_label, args.save_path)
        else:
            embedding_dir = get_embeddings_labels(args.lm_model, args.paraphrase, args.use_label, args.save_path)
    else:
        embedding_dir = args.embeddings_file
    
    print(f'Loading embedding file')
    
    if args.paraphrase:
        train_embedding_file = os.path.join(embedding_dir, f'train_descriptions_0_embeddings.pt')
        eval_embedding_file = os.path.join(embedding_dir, f'eval_descriptions_0_embeddings.pt')
    else:
        train_embedding_file = os.path.join(embedding_dir, f'train_label_embeddings.pt')
        eval_embedding_file = os.path.join(embedding_dir, f'eval_label_embeddings.pt')
    
    print(f'Loading data')
    drug_metadata = pd.read_pickle(os.path.join(DATA_DIR, 'drug_features/drug_metadata_ddi.pkl'))
    drug_metadata['view_str'] = 1  # all drugs must have structure (filtered already during preprocessing)

    # load all structure modality data
    all_molecules = torch.load(os.path.join(DATA_DIR, 'drug_features/str/all_molecules_torchdrug.pt'), map_location='cpu')

    # load all KG modality data
    all_kg_data = torch.load(os.path.join(DATA_DIR, f'drug_features/kg/KG_data_{kg_encoder}.pt'), map_location='cpu')

    # load perturbation data
    cv_df = pd.read_csv(DATA_DIR + 'drug_features/cv/cv_cp_data.csv', index_col=0)
    tx_df = pd.read_csv(DATA_DIR + 'drug_features/tx/tx_cp_data_dose_averaged.csv', index_col=0)


    # load label map
    with open(DATA_DIR + f'drug_combination_data/{data_source}/{data_source.lower()}_ddi_directed_label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
        
    print(f'Loading dataloaders')
    # load datasets
    train_dataset, eval_dataset = get_datasets(args.paraphrase,
                                               args.use_label,
                                               embedding_dir,
                                               train_embedding_file, 
                                               eval_embedding_file)

    train_kg_data = all_kg_data

    train_collator, eval_collator= get_collators(drug_metadata, all_molecules, all_kg_data, train_kg_data, cv_df, tx_df, 158)

    train_loader, eval_loader = get_dataloaders(train_dataset, eval_dataset, train_collator, eval_collator, args.train_batch_size, args.eval_batch_size, args.weighted_sampling)
    
    print(f'Loading model')
    model, loss_fn, optimizer = get_full_model(args, train_collator, all_kg_data)
    
    trainer(args, train_loader, eval_loader, model, loss_fn, optimizer, run)
    
    
def trainer(args, train_loader, eval_loader, model, loss_fn, optimizer, run):
    num_epochs = args.num_train_epochs
    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=1000)
        
    if args.use_wandb:
        save_dir = run.name
    elif not args.use_wandb or save_dir == '':
        save_dir = 'test'
    
    full_save_dir = os.path.join(args.save_path, save_dir) 
    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir)
        
    torch.save(model.state_dict(), full_save_dir + f'/lm_{data_source}_{args.split_method}_base.pt')

    
    print('\nTraining....')

    for epoch in range(num_epochs):
        print(f'Training at epoch {epoch}')
        loss_accum = 0
        for step, train_batch in enumerate(tqdm(train_loader)):

            model.train()
            optimizer.zero_grad()

            batch_head = train_batch['head']  
            batch_tail = train_batch['tail']
            batch_kg = train_batch['kg']
            head_masks_base = train_batch['head']['masks']  
            tail_masks_base = train_batch['tail']['masks']
            ddi_head_indices = train_batch['edge_indices']['head']
            ddi_tail_indices = train_batch['edge_indices']['tail']
            ddi_labels = train_batch['edge_indices']['label']
            ddi_pos_neg_samples = train_batch['edge_indices']['pos_neg']
            embeddings = train_batch['text_embeddings']
            
            if args.paraphrase:
                sample_descriptions = torch.randint(0, 10, (embeddings.size(dim=0),)) # B
                expanded_indices = sample_descriptions.view(-1, 1, 1).expand(-1, 1, embeddings.size(dim=-1)) #B,1,4096
                embeddings = torch.gather(embeddings, 1, expanded_indices).squeeze() #B,4096
                
            batch_head = to_device(batch_head, device)
            batch_tail = to_device(batch_tail, device)
            batch_kg = to_device(batch_kg, device)
            ddi_labels = to_device(ddi_labels, device)
            ddi_head_indices = to_device(ddi_head_indices, device)
            ddi_tail_indices = to_device(ddi_tail_indices, device)
            ddi_pos_neg_samples = to_device(torch.tensor(ddi_pos_neg_samples), device)
            embeddings = embeddings.to(device)

            if args.loss=='bce_with_weight':
                pred_pos_neg = model(batch_head, batch_tail, to_device(head_masks_base, device), 
                                     to_device(tail_masks_base, device), 
                                     batch_kg,  embeddings)

            else:
                pred_pos_neg = torch.sigmoid(model(batch_head, batch_tail, to_device(head_masks_base, device), 
                                                   to_device(tail_masks_base, device), 
                                                   batch_kg,  embeddings))

            loss = loss_fn(pred_pos_neg.float(), ddi_pos_neg_samples.unsqueeze(-1).float())

            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().cpu().item()

            if step % 500 == 0:
                print(f'Loss at {step} step of epoch {epoch}: {loss_accum/(step+1)}')
        
        if args.use_wandb:
            run.log({"loss": loss_accum/(step+1)}, step=epoch)

        print(f'Loss at epoch {epoch}: {loss_accum/(step+1)}')

        print(f'Evaluating')
        if args.paraphrase:
            _, _ = evaluate_paraphrased(eval_loader, 'eval', model, run, args.use_wandb, epoch, args.save_path)
        else:
            _, _ = evaluate(eval_loader, 'eval', model, run, args.use_wandb, epoch, args.save_path)
        
        torch.save(model.state_dict(), os.path.join(args.save_path, save_dir) + f'/lm_{data_source}_{args.split_method}_{epoch}.pt')
    
    
@torch.no_grad()
def evaluate_paraphrased(loader, name, model, run, use_wandb, epoch, save_path, eval_all=True):
    if use_wandb:
        save_dir = run.name
    elif not use_wandb or save_dir == '':
        save_dir = 'test'
    
    if eval_all:
        stop = 10
    else:
        stop = 1
        
    for i in range(stop):
        
        print(f'\nEvaluating on descriptions {i}...')
        
        true_labels = []
        pred_labels = []
        raw_scores = []
    
        for step, eval_batch in enumerate(tqdm(loader)):

            batch_head = eval_batch['head']  
            batch_tail = eval_batch['tail']
            batch_kg = eval_batch['kg']
            head_masks_base = eval_batch['head']['masks']  
            tail_masks_base = eval_batch['tail']['masks']
            ddi_head_indices = eval_batch['edge_indices']['head']
            ddi_tail_indices = eval_batch['edge_indices']['tail']
            ddi_labels = eval_batch['edge_indices']['label']
            ddi_pos_neg_samples = eval_batch['edge_indices']['pos_neg']
            embeddings = eval_batch['text_embeddings']
            
            embeddings = embeddings[:,i,:].squeeze()
            
            batch_head = to_device(batch_head, device)
            batch_tail = to_device(batch_tail, device)
            batch_kg = to_device(batch_kg, device)
            ddi_labels = to_device(ddi_labels, device)
            ddi_head_indices = to_device(ddi_head_indices, device)
            ddi_tail_indices = to_device(ddi_tail_indices, device)
            ddi_pos_neg_samples = to_device(torch.tensor(ddi_pos_neg_samples), device)
            embeddings = embeddings.to(device)

            raw_pos_neg = model(batch_head, batch_tail, to_device(head_masks_base, device), to_device(tail_masks_base, device), batch_kg, embeddings)
            pred_pos_neg = torch.sigmoid(raw_pos_neg)

            raw_scores.append(raw_pos_neg.detach().cpu().numpy())
            true_labels.append(ddi_pos_neg_samples.detach().cpu().numpy())
            pred_labels.append(pred_pos_neg.detach().cpu().numpy())
    
        preds = np.concatenate(pred_labels).ravel()
        ys = np.concatenate(true_labels).ravel()
        raw_scores = np.concatenate(raw_scores).ravel()
        
        np.save(os.path.join(save_path, save_dir) + f'/{name}_raw_scores_{epoch}_description_{i}.npy', raw_scores)
    
        metrics = get_metrics_binary(preds, ys, k=50)
        metrics_dict = {metric_name + name + f'_description_{i}': (None if metric_value == 'nan' else metric_value) for metric_value, metric_name in zip(*metrics)}
        if use_wandb:
            run.log(metrics_dict, step=epoch)
            
    return None, None
    

@torch.no_grad()
def evaluate(loader, name, model, run, use_wandb, epoch, save_path):
    
    print('\nEvaluating...')
    
    if use_wandb:
        save_dir = run.name
    elif not use_wandb or save_dir == '':
        save_dir = 'test'
    
    
    true_labels = []
    pred_labels = []
    raw_scores = []
    
    for step, eval_batch in enumerate(tqdm(loader)):
        
        batch_head = eval_batch['head']  
        batch_tail = eval_batch['tail']
        batch_kg = eval_batch['kg']
        head_masks_base = eval_batch['head']['masks']  
        tail_masks_base = eval_batch['tail']['masks']
        ddi_head_indices = eval_batch['edge_indices']['head']
        ddi_tail_indices = eval_batch['edge_indices']['tail']
        ddi_labels = eval_batch['edge_indices']['label']
        ddi_pos_neg_samples = eval_batch['edge_indices']['pos_neg']
        embeddings = eval_batch['text_embeddings']
        
        batch_head = to_device(batch_head, device)
        batch_tail = to_device(batch_tail, device)
        batch_kg = to_device(batch_kg, device)
        ddi_labels = to_device(ddi_labels, device)
        ddi_head_indices = to_device(ddi_head_indices, device)
        ddi_tail_indices = to_device(ddi_tail_indices, device)
        ddi_pos_neg_samples = to_device(torch.tensor(ddi_pos_neg_samples), device)
        embeddings = embeddings.to(device)
        
        raw_pos_neg = model(batch_head, batch_tail, to_device(head_masks_base, device), to_device(tail_masks_base, device), batch_kg, embeddings)
        pred_pos_neg = torch.sigmoid(raw_pos_neg)
        
        raw_scores.append(raw_pos_neg.detach().cpu().numpy())
        true_labels.append(ddi_pos_neg_samples.detach().cpu().numpy())
        pred_labels.append(pred_pos_neg.detach().cpu().numpy())
    
    preds = np.concatenate(pred_labels).ravel()
    ys = np.concatenate(true_labels).ravel()
    raw_scores = np.concatenate(raw_scores).ravel()
    
    np.save(os.path.join(save_path, save_dir) + f'/{name}_raw_scores_{epoch}.npy', raw_scores)
    
    metrics = get_metrics_binary(preds, ys, k=50)
    metrics_dict = {metric_name + name: (None if metric_value == 'nan' else metric_value) for metric_value, metric_name in zip(*metrics)}
    
    if use_wandb:
        run.log(metrics_dict, step=epoch)
    return preds, ys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Madrigal LM')
    
    # General args
    parser.add_argument('--split_method', type=str, default='split_by_classes')
    parser.add_argument('--use_wandb', action="store_true", default=False)
    parser.add_argument('--paraphrase', action="store_true", default=False)
    parser.add_argument('--use_label', action="store_true", default=False)
    parser.add_argument('--generate_embeddings', action="store_true", default=False)
    parser.add_argument('--embeddings_file', type=str, default=None)
    
    
    parser.add_argument('--num_negative_samples', type=int, default=2)
    parser.add_argument('--lm_model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--save_path', type=str, default='./')
    
    parser.add_argument('--use_pretrained', action="store_true", default=True)
    parser.add_argument('--pretrained_path', type=str, default='DrugBank/checkpoint_1000.pt')

    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay for training')
    
    parser.add_argument('--train_batch_size', type=int, default=3000, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=3000, help='Batch size for training')
    parser.add_argument('--num_train_epochs', type=int, default=20, help='Total number of epochs for training')
    
    parser.add_argument('--loss', type=str, default="bce_with_weight")
    parser.add_argument('--pos_weight', type=float, default=2.0)
    parser.add_argument('--weighted_sampling', action="store_true", default=False)
    
    parser.add_argument('--drug_project_dim', type=int, default=128)
    parser.add_argument('--text_project_dim', type=int, default=128)
    parser.add_argument('--mlp_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--self_att', action="store_true", default=False)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    
    
    args = parser.parse_args()
    
    main(args)
