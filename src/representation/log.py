import torch

class Log:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log2(x)
