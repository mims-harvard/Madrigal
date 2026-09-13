from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import sys
from torch import nn
import copy
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
import argparse

from madrigal.models import models
from madrigal.evaluate import metrics
from madrigal.utils import DATA_DIR

parser = argparse.ArgumentParser(description='KG pretraining')
parser.add_argument('--hgt_num_layers', type=int, default=2, help='HGT number of layers')
parser.add_argument('--hgt_hidden_dim', type=int, default=128, help='HGT hidden dimension')
parser.add_argument('--hgt_att_heads', type=int, default=4, help='HGT num att heads')
parser.add_argument('--hgt_group', type=str, default='sum', help='HGT group', choices=['sum', 'mean', 'max'])
parser.add_argument('--feature_dim', type=int, default=128, help='input feature dimension to transformer (i.e. output feature dimension of view encoders, position embedder and CLS embedder)')
args = parser.parse_args()

VIEWS_PATH = DATA_DIR+'drug_features/kg/KG_data_hgt.pt'

graph = torch.load(VIEWS_PATH)


edge_types = []
rev_edge_types = []
for edge in graph.edge_types:
    if 'rev' not in edge[1]:
        edge_types.append(edge)
        rev_edge_types.append(None)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected = False,
    disjoint_train_ratio=0.1,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=True,
    edge_types=edge_types,
    rev_edge_types=rev_edge_types, 
)
train_data, val_data, test_data = transform(graph)

def manually_add_rev_labels(data):
    for edge in data.edge_types:
        if 'rev' in edge[1]:
            normal_edge = edge[1].replace("rev_", "")
            normal = (edge[2], normal_edge, edge[0])
            normal_edge_index = data[normal].edge_label_index
            data[edge]['edge_label_index'] = torch.roll(normal_edge_index, 1, 0)
            data[edge]['edge_label'] = data[normal].edge_label

manually_add_rev_labels(train_data)
manually_add_rev_labels(val_data)
manually_add_rev_labels(test_data)

train_edge_label_index = [ (train_data[edge]['edge_label_index'],edge) for edge in train_data.edge_types]
val_edge_label_index = [ (val_data[edge]['edge_label_index'],edge) for edge in val_data.edge_types]
test_edge_label_index = [ (test_data[edge]['edge_label_index'],edge) for edge in test_data.edge_types]

train_labels = [ train_data[edge]['edge_label'] for edge in train_data.edge_types]
val_labels = [ val_data[edge]['edge_label'] for edge in val_data.edge_types]
test_labels = [ test_data[edge]['edge_label'] for edge in test_data.edge_types]

class HGTLinkPred(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, 
                 metadata, num_edge_types, group='sum'):
        super(HGTLinkPred, self).__init__()
        
        self.encoder = models.HGT(in_channels, hidden_channels, out_channels, 
                                  num_layers, num_heads, metadata, group)
        self.decoder = models.BilinearDDIScorer(out_channels, out_channels, 1)
        self.decoders = [self.decoder for i in range(num_edge_types)]
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, debug=False):
        z_dict = self.encoder(x_dict, edge_index_dict)
        preds = []
        for decoder, (edge_pred_index, edge_name) in zip(self.decoders, edge_label_index):
            i, r, j = edge_name
            node_in, node_out = edge_pred_index
            pred = decoder(z_dict[i], z_dict[j]).squeeze(0)
            pred = pred[node_in, node_out]
            preds.append(pred)
        return torch.cat(preds, 0)
    
    def save_checkpoint(self, PATH):
        torch.save(self.encoder.state_dict(), PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HGTLinkPred(in_channels=train_data.x_dict['drug'].shape[1],
                    hidden_channels=args.hgt_hidden_dim, 
                    out_channels=args.feature_dim, 
                    num_layers=args.hgt_num_layers, 
                    num_heads=args.hgt_att_heads,
                    group=args.hgt_group,
                    num_edge_types=len(train_edge_label_index),
                    metadata = train_data.metadata()).to(device)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    preds = model(train_data.x_dict, train_data.edge_index_dict, train_edge_label_index)
    targets = torch.cat(train_labels).to(device)
    loss = loss_fn(preds, targets) 
    loss.backward()
    optimizer.step()
    return float(loss)
    
@torch.no_grad()
def test(data, edge_label_index, labels):
    model.eval()
    preds = torch.sigmoid(model(data.x_dict, data.edge_index_dict, edge_label_index))
    targets = torch.cat(labels).to(device)
    
    _ = metrics.get_metrics_binary(preds.cpu().numpy(), targets.cpu().numpy(), k=50, verbose=True)

for epoch in range(1, 301):
    print(f'Training at epoch {epoch}')
    loss = train()
    print(f'Loss: {loss}')
    print('Validating')
    test(val_data, val_edge_label_index, val_labels)
    print('Testing')
    test(test_data, test_edge_label_index, test_labels)
    if epoch % 100 == 0:
        model.save_checkpoint(f'hgt_2_{epoch}.pt')
