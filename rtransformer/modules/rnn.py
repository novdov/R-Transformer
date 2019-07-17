from typing import Type

import torch
import torch.nn as nn


def _get_rnn(rnn_type: str) -> Type[nn.Module]:
    rnn_type = rnn_type.lower()
    if rnn_type == "rnn":
        return nn.RNN
    elif rnn_type == "lstm":
        return nn.LSTM
    elif rnn_type == "gru":
        return nn.GRU
    else:
        raise ValueError(f"Not supported rnn: {rnn_type}")


class LocalRNN(nn.Module):
    """Local RNN processes only M size sequences."""

    def __init__(
        self,
        hidden_size: int,
        window_size: int,
        dropout: float,
        seq_len: int,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.window_size = window_size
        self.rnn = _get_rnn(rnn_type)(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        padding_size = self.window_size - 1
        # This indices works like windows in CNN
        # Consider padded sequences for making sub-sequences
        # len(indices) = (seq_len + padding_size) * window_size
        indices = [
            i
            for j in range(padding_size, seq_len + padding_size)
            for i in range(j - padding_size, j + 1)
        ]
        self.indices = torch.tensor(indices, dtype=torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_sub_sequences(x)
        batch_size, seq_len, window_size, d_model = x.size()
        # output: (batch_size * seq_len, window_size, d_model)
        output, _ = self.rnn(x.view(-1, window_size, d_model))
        hidden_state = output[0][:, -1, :].view(batch_size, seq_len, d_model)
        return hidden_state

    def get_sub_sequences(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = sequence.size()
        # padding by (M - 1) position where M is window size
        # (batch_size, window_size - 1, d_model)
        padding = (
            torch.zeros((self.window_size - 1, d_model))
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        sequence = torch.cat([padding, sequence], dim=1)
        sub_sequences = sequence.index_select(dim=1, index=self.indices)
        # (batch_size, seq_len, window_size, d_model)
        sub_sequences = sub_sequences.reshape(batch_size, seq_len, self.window_size, -1)
        return sub_sequences
