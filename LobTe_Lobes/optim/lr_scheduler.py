"""
File: lr_scheduler.py
Author: Ariel Hernán Curiale
Github: https://gitlab.com/Curiale
Description: Custom Learninig rate schedulers
"""


class TransformerScheduler:

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.step()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model**-0.5) * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    def step(self):
        """Learning rate scheduling per step"""

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
