import torch

class SmapeLoss:
    def __init__(self):
        pass

    def __call__(self, pred, gt):
        pred = pred * 4000.0
        gt = gt * 4000.0
        return torch.sum(torch.abs(torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+1e-5)))
