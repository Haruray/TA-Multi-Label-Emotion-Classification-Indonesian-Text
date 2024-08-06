# base class for multi-label emotion classification
import torch.nn as nn
import torch


class MLECModel(nn.Module):

    def __init__(
        self,
        alpha=0.2,
        beta=0.1,
        device="cuda:0",
        name="MLECModel",
    ):
        super(MLECModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.name = name

    def forward(self, batch, device):
        pass

    def compute_pred(self, logits, threshold=0.5):
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()
