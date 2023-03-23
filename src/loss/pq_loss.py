import torch
from torch.nn import L1Loss

L_MAX = 10000
N = 0.1593017578125
M = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875

class PQLoss:
    def __init__(self):
        self.loss = L1Loss()

    def pq(self, x):
        im_t = torch.pow(torch.maximum(x, torch.zeros_like(x)),1 / M)
        out = L_MAX * torch.pow(torch.maximum(im_t - C1, torch.zeros_like(x))/(C2 - C3 * im_t), 1 / N)
        return out

    def __call__(self, pred, gt):
        return self.loss(self.pq(pred), self.pq(gt))
