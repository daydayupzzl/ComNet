import torch
import torch.nn as nn


class FingerFNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(FingerFNN, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )

    def forward(self, data):
        return self.lin(data.float())