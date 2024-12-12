import torch
import torch.nn as nn
from encoders import DrugEncoderWithSkipConnect, SMILESTransformerEncoder
from fingers import FingerFNN
from attention import Attention


class ComNet_DDI(nn.Module):
    def __init__(self, device, node_dim, edge_dim, hidden_dim=64, dropout=0.2, rmodule_dim=86):
        super(ComNet_DDI, self).__init__()

        self.drug_encoder = DrugEncoderWithSkipConnect(node_dim, edge_dim, dropout, hidden_dim)
        self.finger_encoder = FingerFNN(in_dim=1024, dropout=dropout, hidden_dim=hidden_dim)
        self.smiles_transformer = SMILESTransformerEncoder(66, device, max_len=100, d_model=hidden_dim, dropout=dropout)
        self.relation_embedding = nn.Embedding(rmodule_dim, hidden_dim * 6)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 6, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self.attention = Attention(hidden_dim)
        self.device = device
        self.hidden_dim = hidden_dim

        self.to(device)

    def forward(self, inputs):
        h_data, t_data, head_fins, head_input_ids, head_mask, tail_fins, tail_input_ids, tail_mask, rels, smiles_h, smiles_t = inputs

        head_features = self._process_data(h_data, head_fins, head_input_ids, head_mask)
        tail_features = self._process_data(t_data, tail_fins, tail_input_ids, tail_mask)

        pair_embedding = torch.cat([head_features, tail_features], dim=-1)

        relation_embedding = self.relation_embedding(rels)
        embedding = pair_embedding * relation_embedding
        logit = self.fc_layers(embedding).sum(-1)

        return logit

    def _process_data(self, drug_data, finger_data, input_ids, attention_mask):
        drug_embedding = self.drug_encoder(drug_data)
        smiles_embedding = self.smiles_transformer(input_ids, attention_mask).mean(dim=1)
        finger_embedding = self.finger_encoder(finger_data)
        attention_score = self.attention(drug_embedding, smiles_embedding, finger_embedding)
        combined_embedding = self._apply_attention(attention_score, drug_embedding, smiles_embedding, finger_embedding)
        return combined_embedding

    def _apply_attention(self, attention_score, drug_embedding, smiles_embedding, finger_embedding):
        attention_weighted_emb = (attention_score.unsqueeze(dim=2) *
                                  torch.stack([drug_embedding, smiles_embedding, finger_embedding], dim=1))
        return attention_weighted_emb.reshape(-1, 3 * self.hidden_dim)
