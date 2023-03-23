import torch
from torch.nn import L1Loss

MU = 5000.0

class MuLoss:
    def __init__(self):
        self.loss = L1Loss()

    def mu(self, x):
        return torch.log(1.0 + MU * (x + 1.0) / 2.0) / torch.log(1.0 + MU)

    def __call__(self, pred, gt):
        return self.loss(self.mu(pred), self.mu(gt))
