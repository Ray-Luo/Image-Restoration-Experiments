import torch
from torch.nn import L1Loss

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428


class PULoss:
    def __init__(self):
        self.loss = L1Loss()

    def pu(self, x):
        return (torch.pow(2.0, x) - B_COEFF) / A_COEFF

    def __call__(self, pred, gt):
        return self.loss(self.pu(pred), self.pu(gt)) * 4000.0
