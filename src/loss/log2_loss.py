import torch
from torch.nn import L1Loss


class Log2Loss:
    def __init__(self):
        self.loss = L1Loss()

    def pow(self, x):
        return torch.pow(2, x)


    def __call__(self, pred, gt):
        return self.loss(self.pow(pred), self.pow(gt)) * 4000.0
