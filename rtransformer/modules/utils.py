import copy

import numpy as np
import torch
import torch.nn as nn


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


def clones(module, n):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    _subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(_subsequent_mask) == 0
