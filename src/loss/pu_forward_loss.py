import torch
from torch.nn import L1Loss

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428


class PUForwardLoss:
    def __init__(self):
        self.loss = L1Loss()

    def pu(self, x):
        # x *= 4000.0
        return torch.log2(A_COEFF * x + B_COEFF)

    def __call__(self, pred, gt):
        return self.loss(self.pu(pred), self.pu(gt))
