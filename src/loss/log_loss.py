import torch
import torch.nn.L1Loss as l1_loss

class LogLoss(torch.nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()

    def log(self, x):
        return torch.log(torch.max(x, 1e-5))

    def forward(self, pred, gt):
        return l1_loss(self.log(pred), self.log(gt))
