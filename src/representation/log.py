import torch

class Log:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torch.clamp(x, min=1e-5)
        x = torch.log(x)
        x = x / torch.max(x)
        return x
