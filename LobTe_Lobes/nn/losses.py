"""
File: losses.py
Author: Ariel Hernán Curiale
Github: https://gitlab.com/Curiale
Description:
"""

import torch


class Loss:
    """Wrapper class to add a name attribute to the pytorch loss functions"""

    def __init__(self, name, fx, bias_penalty=0):
        self.name = name
        self.fx = fx
        self.bias_penalty = bias_penalty

    def __call__(self, y_pred, y_true):
        loss_val = self.fx(y_pred, y_true)
        if self.bias_penalty > 0:
            # Add a bias penalty to the loss function to remove any linear or
            # proportional bias in the error. To this end we compute the
            # pearson correlation between the residuals and the true values. If
            # the correlation is high, it means that the model is making a
            # proportional error in its predictions. The penalty is added to
            # the loss function to remove this bias.
            res = y_pred - y_true
            # Centered residuals and true values
            res_centered = res - res.mean()
            y_centered = y_true - y_true.mean()
            # Estimate correlation (covariance / stds)
            cov = (res_centered * y_centered).mean()
            corr = cov / (res.std() * y_true.std() + 1e-6)
            # Add a proportional bias penalty
            penalty = torch.abs(corr)  # or use corr ** 2
            loss_val = loss_val + self.bias_penalty * penalty
        return loss_val
