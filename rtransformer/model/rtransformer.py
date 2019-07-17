import torch
import torch.nn as nn

from rtransformer.modules.block import RTransformerBlock
from rtransformer.modules.utils import clones


class RTransformer(nn.Module):
    """Complete R-Transformer model."""

    def __init__(
        self,
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
        layers = clones(
            RTransformerBlock(
                d_model,
                d_ff,
                num_rnn_layer,
                num_head,
                window_size,
                seq_len,
                dropout,
                rnn_type,
            ),
            num_block,
        )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
