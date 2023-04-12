import torch
from torch.nn import L1Loss


class LogLoss:
    def __init__(self):
        self.loss = L1Loss()

    def exp(self, x):
        x = torch.exp(x)
        return x

    def __call__(self, pred, gt):
        return self.loss(self.exp(pred), self.exp(gt))
