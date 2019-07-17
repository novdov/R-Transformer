import torch
import torch.nn as nn

from rtransformer.modules.attention import MultiHeadedAttention
from rtransformer.modules.layer import (
    LocalRNNLayer,
    PositionwiseFeedForward,
    SublayerConnection
)
from rtransformer.modules.utils import clones


class RTransformerBlock(nn.Module):
    """R-Transformer block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_rnn_layer: int,
        num_head: int,
        window_size: int,
        seq_len: int,
        dropout: float,
        rnn_type: str,
    ):
        super().__init__()
        self.local_rnn_layers = clones(
            LocalRNNLayer(d_model, window_size, seq_len, dropout, rnn_type),
            num_rnn_layer,
        )
        self.sublayer_connection = clones(SublayerConnection(d_model, dropout), 2)
        self.multi_head_attention = MultiHeadedAttention(num_head, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_ff, d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.local_rnn_layers:
            x = layer(x)
        x = self.sublayer_connection[0](x, self.multi_head_attention)
        x = self.sublayer_connection[1](x, self.ffn)
        return x
