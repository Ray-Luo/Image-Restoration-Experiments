import torch

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428

class PU(torch.nn.Module):
    def __init__(self):
        super(PU, self).__init__()

    def forward(self, x):
        return torch.log2(A_COEFF * x) + B_COEFF
