import torch

class Log(torch.nn.Module):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        return torch.log2(x)
