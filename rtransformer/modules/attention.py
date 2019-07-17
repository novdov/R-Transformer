import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtransformer.modules import utils


def scaled_dot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled-dot attention"""

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, num_head: int, d_model: int, seq_len: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.subsequent_mask = utils.subsequent_mask(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nbatches = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.num_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (x, x, x))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = scaled_dot_attention(
            query, key, value, mask=self.subsequent_mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_head * self.d_k)
        return self.linears[-1](x)
