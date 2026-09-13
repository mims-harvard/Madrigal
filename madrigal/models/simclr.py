"""
Following the implementation in https://github.com/sthalles/SimCLR/blob/master/, Q and K are concatenated together rather than being separate as in MoCo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import NUM_NON_TX_MODALITIES


class SimCLR_Madrigal(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=1024, T=1.0, raw_encoder_output=False, shared_predictor=False):
        super(SimCLR_Madrigal, self).__init__()
        self.base_encoder = base_encoder
        self.T = T
        self.raw_encoder_output = raw_encoder_output  # whether to use modality encoder outputs or the alreayd projected embeddings
        self.shared_predictor = shared_predictor  # whether to use the same predictor for both streams
        
        # add mlp projection head
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # lazy initialize
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.uni_projector.fc[-1].weight.shape[0]  # get last dim of xW^T, same as self.base_encoder.transformer.latent2embed.weight.shape[0]
        assert hidden_dim == dim, f"Hidden dim of the encoder ({hidden_dim}) should be the same as the dim of the projection head ({dim})."
        
        # NOTE: projectors OMITTED

        # predictor
        if self.shared_predictor:
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
        else:
            self.predictor_1 = self._build_mlp(2, dim, mlp_dim, dim)
            self.predictor_2 = self._build_mlp(2, dim, mlp_dim, dim)


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    def contrastive_loss(self, aug1, aug2, batch_too_hard_neg_mask):
        assert aug1.shape[0] == aug2.shape[0]
        features = torch.cat([aug1, aug2], dim=0)
        labels = torch.cat([torch.arange(aug1.shape[0])] * 2, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # mask out the scores from the pairs that should not be negatives (too similar)
        if batch_too_hard_neg_mask is not None:
            similarity_matrix.masked_fill_(batch_too_hard_neg_mask.repeat(2, 2), -1e9)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1).to(features.device)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives

        # select only the negatives the negatives

        
        
        logits = similarity_matrix / self.T

        return logits, labels, torch.nn.CrossEntropyLoss()(logits, labels)

    def forward(self, drug_indices, batch_mask_1, batch_mask_2, batch_too_hard_neg_mask, batch_data, batch_extra_mols, batch_extra_masks):
        """
        Input:
            drug_indices
            mask1: first tensor of subsets of views of images
            mask2: second tensor of subsets of views of images
            m: moco momentum
        Output:
            loss
        """
        batch_mols, batch_kg, batch_cv, batch_tx_dict = batch_data
        other_tabular_mod_data = {}
        
        # compute features
        if self.shared_predictor:
            aug_1 = self.predictor(self.base_encoder(drug_indices, batch_mask_1, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output, **other_tabular_mod_data))  # raw encoder output is [batch_size, seq_len, hidden_dim], then select the corresponding modality output for each compound
            aug_2 = self.predictor(self.base_encoder(drug_indices, batch_mask_2, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output, **other_tabular_mod_data))
        else:
            aug_1 = self.predictor_1(self.base_encoder(drug_indices, batch_mask_1, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output, **other_tabular_mod_data))  # raw encoder output is [batch_size, seq_len, hidden_dim], then select the corresponding modality output for each compound
            aug_2 = self.predictor_2(self.base_encoder(drug_indices, batch_mask_2, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output, **other_tabular_mod_data))

        torch.cuda.empty_cache()

        return aug_1, aug_2, self.contrastive_loss(aug_1, aug_2, batch_too_hard_neg_mask)

