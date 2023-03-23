import torch

class SmapeLoss(torch.nn.Module):
    def __init__(self):
        super(SmapeLoss, self).__init__()

    def forward(self, pred, gt):
        return torch.abs(torch.abd(pred-gt)/(torch.abs(pred)+torch.abs(gt)+1e-5))
