import torch
import torch.nn as nn

from rtransformer.modules.rnn import LocalRNN


class LocalRNNLayer(nn.Module):
    """Local RNN layer with local RNN and residual connection."""

    def __init__(
        self,
        d_model: int,
        window_size: int,
        seq_len: int,
        dropout: float,
        rnn_type: str,
    ):
        super().__init__()
        self.local_rnn = LocalRNN(d_model, window_size, dropout, seq_len, rnn_type)
        self.sublayer_connection = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sublayer_connection(x, self.local_rnn)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Networks"""

    def __init__(self, d_ff: int, d_model: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
