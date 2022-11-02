# From https://github.com/ShannonAI/dice_loss_for_NLP/blob/master/loss/dice_loss.py
# flake8: noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional



class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, eps=0.):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.

        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n] or [bs]): Targets.

        Returns:
            torch tensor [bs]: Loss values.
        """
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)
        loss = loss.sum(-1)

        return loss


class NBMELoss(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.defice = device

        if self.config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif self.config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=config["smoothing"])
        elif self.config["name"] == "mse":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(f"Loss name {self.config['name']} not supported")

    def prepare(self, pred, y):
        y = y.view(-1)
        if self.config["name"] == "ce":
            pred = pred.view(y.size(0), -1)
        else:
            pred = pred.view(-1)
            y = y.float()
        return pred, y

    def forward(self, pred, y):
        pred, y = self.prepare(pred, y)
        loss = self.loss(pred, y)
        return loss.sqrt()
