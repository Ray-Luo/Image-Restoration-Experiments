import torch

class Log2:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log2(torch.max(x, torch.ones_like(x) * 1e-5))
