import torch

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428

class PU:
    def __init__(self):
        pass

    def __call__(self, x):
        max = torch.log2(A_COEFF * 4000.0 + B_COEFF)
        return torch.log2(A_COEFF * x + B_COEFF) / max
