import torch
import numpy as np

MU = 5000.0

class MuTonemap(torch.nn.Module):
    """ tonemapping HDR images using μ-law before computing loss """
    def __init__(self, requires_grad=False):
        super(MuTonemap, self).__init__()

    def forward(self, x):
        return torch.log(1.0 + MU * (x + 1.0) / 2.0) / np.log(1.0 + MU)
