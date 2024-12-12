import torch
import torch.nn as nn


def _project_layer(in_size, hidden_size):
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1, bias=False)
    )


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()
        self.project_x = _project_layer(in_size, hidden_size)
        self.project_xt = _project_layer(in_size, hidden_size)
        self.project_xq = _project_layer(in_size, hidden_size)

    def forward(self, x, xt, xq):
        a = torch.cat((
            self.project_x(x),
            self.project_xt(xt),
            self.project_xq(xq)
        ), dim=1)
        return torch.softmax(a, dim=1)
