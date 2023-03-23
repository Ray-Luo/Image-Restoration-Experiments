import torch
import torch.nn.L1Loss as l1_loss

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428

class PULoss(torch.nn.Module):
    def __init__(self):
        super(PULoss, self).__init__()

    def pu(self, x):
        return (torch.pow(2.0, x) - B_COEFF) / A_COEFF

    def forward(self, pred, gt):
        return l1_loss(self.pu(pred), self.pu(gt))
