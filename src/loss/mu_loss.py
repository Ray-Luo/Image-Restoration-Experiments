import torch
import numpy as np
import torch.nn.L1Loss as l1_loss

MU = 5000.0

class MuLoss(torch.nn.Module):
    """ tonemapping HDR images using μ-law before computing loss """
    def __init__(self):
        super(MuLoss, self).__init__()

    def mu(self, x):
        return torch.log(1.0 + MU * (x + 1.0) / 2.0) / np.log(1.0 + MU)

    def forward(self, pred, gt):
        return l1_loss(self.mu(pred), self.mu(gt))
