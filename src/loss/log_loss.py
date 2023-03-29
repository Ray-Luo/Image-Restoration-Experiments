import torch
from torch.nn import L1Loss


class LogLoss:
    def __init__(self):
        self.loss = L1Loss()

    def log(self, x):
        x = torch.log(torch.max(x, torch.ones_like(x) * 1e-5))
        x = x * 4000.0
        return x

    def __call__(self, pred, gt):
        return self.loss(self.log(pred), self.log(gt))
