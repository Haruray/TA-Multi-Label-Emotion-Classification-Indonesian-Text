# base class for multi-label emotion classification
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


def lca_loss(y_hat, y_true, reduction="mean"):
    loss = torch.zeros(y_true.size(0)).cuda()
    for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
        y_z, y_o = (y == 0).nonzero(), y.nonzero()
        if y_o.nelement() != 0:
            output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
            num_comparisons = y_z.size(0) * y_o.size(0)
            loss[idx] = output.div(num_comparisons)
    return loss.mean() if reduction == "mean" else loss.sum()
