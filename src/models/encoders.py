import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool


def create_positional_encoding(dim, max_len, device):
    pe = torch.zeros(max_len, dim, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class SMILESTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, device, d_model=64, nhead=1, num_encoder_layers=1, dim_feedforward=128,
                 max_len=100, dropout=0.2):
        super(SMILESTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.positional_encoding = create_positional_encoding(d_model, max_len, device)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x += self.positional_encoding[:x.size(1)]
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        return output.permute(1, 0, 2)


class DrugEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, dropout, hidden_dim=64):
        super(DrugEncoder, self).__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(in_dim, hidden_dim),
            GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim),
            GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        ])
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x
        for conv in self.conv_layers:
            x = self._apply_conv_layer(conv, x, data)
        x = global_max_pool(x, data.batch)
        x = self._apply_fc_layers(x)
        return x

    def _apply_conv_layer(self, conv, x, data):
        return F.relu(conv(x, data.edge_index, data.edge_attr))

    def _apply_fc_layers(self, x):
        x = self.relu(self.fc_1(x))
        x = self.dropout(x)
        return self.fc_2(x)


class DrugEncoderWithSkipConnect(nn.Module):
    def __init__(self, in_dim, edge_dim, dropout, hidden_dim=64):
        super(DrugEncoderWithSkipConnect, self).__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(in_dim, in_dim),
            GATConv(in_dim, in_dim, edge_dim=edge_dim),
            GATConv(in_dim, in_dim, edge_dim=edge_dim)
        ])
        self.fc_1 = nn.Linear(in_dim, in_dim)
        self.fc_2 = nn.Linear(in_dim, in_dim)
        self.fc_g1 = nn.Linear(in_dim, hidden_dim)
        self.fc_g2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.mol_bias = nn.Parameter(torch.rand(1, in_dim))
        torch.nn.init.uniform_(self.mol_bias, -0.2, 0.2)

    def forward(self, data):
        mol_x = data.x
        mol_batch = data.batch
        mol_n = mol_x.size(0)

        for i, conv in enumerate(self.conv_layers):
            mol_x = self._apply_conv_layer(conv, mol_x, data, i)
            if i > 0:
                mol_x = self._apply_skip_connection(mol_x, mol_n)

        mol_x = global_max_pool(mol_x, mol_batch)
        mol_x = self._apply_fc_layers(mol_x)
        return self.fc_g2(mol_x)

    def _apply_conv_layer(self, conv, x, data, layer_idx):
        if layer_idx < len(self.conv_layers) - 1:
            return F.relu(conv(x, data.edge_index, data.edge_attr))
        return conv(x, data.edge_index, data.edge_attr)

    def _apply_skip_connection(self, mol_x, mol_n):
        mol_z = torch.sigmoid(self.fc_1(mol_x) + self.fc_2(mol_x) + self.mol_bias.expand(mol_n, -1))
        return mol_z * mol_x + (1 - mol_z) * mol_x

    def _apply_fc_layers(self, x):
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        return x