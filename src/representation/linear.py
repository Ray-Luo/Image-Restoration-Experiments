import torch

class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x
