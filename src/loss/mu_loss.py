import torch
from torch.nn import L1Loss

MU = 5000.0

class MuLoss:
    def __init__(self):
        self.loss = L1Loss()

    def mu(self, x):
        mu = torch.tensor(MU, dtype=torch.float32)
        x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        x_mu = ((x_mu + 1) / 2 * mu + 0.5)
        return x_mu

    def __call__(self, pred, gt):
        pred = pred * 4000.0
        gt = gt * 4000.0
        return self.loss(self.mu(pred), self.mu(gt))
