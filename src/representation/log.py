import torch

class Log:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log(torch.max(x, torch.ones_like(x) * 1e-5))
