import torch
import torch.nn as nn

from rtransformer.model import RTransformer


class WordLanguageModel(nn.Module):
    """Word-level language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_block: int,
        num_rnn_layer: int,
        num_head: int,
        window_size: int,
        seq_len: int,
        dropout: float,
        rnn_type: str,
    ):
        super().__init__()
        self.rtransformer = RTransformer(
            d_model,
            d_ff,
            num_block,
            num_rnn_layer,
            num_head,
            window_size,
            seq_len,
            dropout,
            rnn_type,
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding_dropout(self.embedding(x))
        preds = self.rtransformer(embedded)
        preds = self.fc(preds)
        return preds
