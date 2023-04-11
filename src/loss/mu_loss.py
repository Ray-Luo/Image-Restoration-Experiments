import torch
from torch.nn import L1Loss

MU = 5000.0

class MuLoss:
    def __init__(self):
        self.loss = L1Loss()

    def mu(self, x):
        mu = torch.tensor(MU, dtype=torch.float32)
        one = torch.tensor(1.0, dtype=torch.float32)
        x_mu = torch.log(torch.max(one + x * MU, torch.ones_like(x) * 1e-5)) / torch.log(one + MU)
        return x_mu

    def __call__(self, pred, gt):
        return self.loss(self.mu(pred), self.mu(gt))
