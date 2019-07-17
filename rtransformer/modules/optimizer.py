import torch


class TransformerLRScheduler:
    """Learning rate scheduler used in standard transformer."""

    def __init__(
        self, optimizer: torch.optim.Adam, d_model: int, warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.init_lr = self.d_model ** -0.5

        self._step = 0

    def step(self):
        self._step += 1

    def get_lr(self):
        arg1 = self.d_model ** -0.5
        arg2 = self._step * (self.warmup_steps ** -1.5)
        return self.init_lr * min(arg1, arg2)
