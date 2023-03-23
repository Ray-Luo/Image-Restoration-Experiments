import torch
import numpy as np
import torch.nn.L1Loss as l1_loss

L_MAX = 10000
N = 0.1593017578125
M = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875

class PQLoss(torch.nn.Module):
    def __init__(self):
        super(PQLoss, self).__init__()

    def pq(self, x):
        im_t = np.power(np.maximum(x,0),1 / M)
        out = L_MAX * np.power(np.maximum(im_t - C1, 0)/(C2 - C3 * im_t), 1 / N)
        return out

    def forward(self, pred, gt):
        return l1_loss(self.pu(pred), self.pu(gt))
