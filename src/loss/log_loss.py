import torch
from torch.nn import L1Loss


class LogLoss:
    def __init__(self):
        self.loss = L1Loss()

    def log(self, x):
        return torch.log(torch.max(x, 1e-5))

    def __call__(self, pred, gt):
        return self.loss(self.log(pred), self.plogu(gt))
