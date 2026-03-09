"""
File: callbacks.py
Author: Ariel Hernán Curiale
Github: https://gitlab.com/Curiale
Description:
"""


class EarlyStopping:

    def __init__(
        self,
        model,
        patience=1,
        min_delta=0,
        mode="min",
        restore_best_weights=False,
    ):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode.lower()
        self.best_loss = float("inf")
        if mode.lower() == "max":
            self.best_loss = -self.best_loss
        self.best_weights = None
        self.restore_best_weights = restore_best_weights
        self.stop_training = False
        self.best_epoch = 0

    def on_epoch_end(self, epoch, loss_val):
        self.stop_training = False
        if self.mode == "min":
            if loss_val < self.best_loss:
                self.counter = 0
                self.best_loss = loss_val
                self.best_epoch = epoch
                if self.restore_best_weights:
                    self.best_weights = {
                        k: v.clone()
                        for k, v in self.model.state_dict().items()
                    }
            elif loss_val >= (self.best_loss + self.min_delta):
                self.counter += 1
                self.stop_training = self.counter >= self.patience
        else:
            if loss_val > self.best_loss:
                self.counter = 0
                self.best_loss = loss_val
                self.best_epoch = epoch
                if self.restore_best_weights:
                    self.best_weights = {
                        k: v.clone()
                        for k, v in self.model.state_dict().items()
                    }
            elif loss_val <= (self.best_loss + self.min_delta):
                self.counter += 1
                self.stop_training = self.counter >= self.patience

    def on_train_start(self):
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }

    def on_train_end(self):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
