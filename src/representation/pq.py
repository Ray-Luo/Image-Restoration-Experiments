import torch
import numpy as np

L_MAX = 10000
N = 0.1593017578125
M = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875


class PQ:
    def __init__(self):
        pass

    def __call__(self, x):
        im_t = torch.pow(torch.clip(x, 0, L_MAX) / L_MAX, N)
        out = torch.pow((C2 * im_t + C1) / (1 + C3 * im_t), M)
        return out
