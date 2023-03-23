import torch

class SmapeLoss:
    def __init__(self):
        pass

    def forward(self, pred, gt):
        return torch.abs(torch.abd(pred-gt)/(torch.abs(pred)+torch.abs(gt)+1e-5))
