import pandas as pd
import numpy as np
from madrigal.models.models import MLPEncoder
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

class AE(nn.Module):
    def __init__(self, ae_encoder_params: dict, ae_decoder_params: dict):
        super(AE, self).__init__()
        self.encoder = MLPEncoder(**ae_encoder_params)
        self.decoder = MLPEncoder(**ae_decoder_params)  # while it is an MLPEncoder object, it is actually a decoder

    def encode(self, x):
        h = F.relu(self.encoder(x))
        return h

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        h = F.relu(self.encode(x))
        recon = self.decode(h)
        return h, recon
    
    def loss(self, recons, x):
        recons_loss= F.mse_loss(recons, x)
        return recons_loss


class CVDataset(Dataset):

    def __init__(self, data_file):
        self.df = pd.read_csv(data_file, index_col=0)
        
    def __len__(self) -> int:
        return self.df.shape[1]

    def __getitem__(self, index):
        item = self.df.iloc[:,index]
        return torch.tensor(item)

dataset = CVDataset('cv_no_test.csv')

train_length=int(0.9* len(dataset))
test_length=len(dataset)-train_length

train_dataset, test_dataset = torch.utils.data.random_split(dataset,(train_length,test_length))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_params ={'in_dim': 559, 'hidden_dims': [512, 256], 'output_dim': 128, 'p':0.2, 
                 'norm': None, 'actn': 'relu', 'order': 'nd'}
decoder_params = {'in_dim': 128, 'hidden_dims': [256, 512], 'output_dim': 559, 'p':0.2, 
                 'norm': None, 'actn': 'relu', 'order': 'nd'}

model =  AE(encoder_params, ae_decoder_params = decoder_params)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_fn = nn.MSELoss(reduction='sum')

def train(x):
    model.train()
    optimizer.zero_grad()
    z, recon = model(x)
    loss = model.loss(recon, x)
    loss.backward()
    optimizer.step()
    return loss
    
@torch.no_grad()
def test(test_loader):
    model.eval()
    losses = []
    for x in tqdm(test_loader):
        z, recon = model(x.float())
        loss = model.loss(recon, x)
        losses.append(loss.detach().cpu().numpy())
    print(f'Loss: {np.array(losses).ravel().mean()}')
    return np.array(losses).ravel().mean()

best_loss= 100
for epoch in range(1, 200):
    print(f'\nTraining at epoch {epoch}')
    losses = []
    for batch in tqdm(train_loader):
        loss = train(batch.float())
        losses.append(loss.detach().cpu().numpy())
    print(f'Loss: {np.array(losses).ravel().mean()}')

    print('\nTesting')
    test_loss = test(test_loader)
    if test_loss < best_loss:
        print('Saving checkpoint')
        torch.save(model.encoder.state_dict(), 'cv_model_ae.pt')
        best_loss = test_loss


@torch.no_grad()
def plot_test(test_loader):
    xs = []
    preds = []
    model.eval()
    for x in tqdm(test_loader):
        z, recon = model(x.float())
        xs.append(x.detach().cpu().numpy())
        preds.append(recon.detach().cpu().numpy())
    return np.array(xs), np.array(preds)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
x,pred = plot_test(test_loader)

arr = np.concatenate([x,pred])
np.save('cv_arr.npy', arr)
