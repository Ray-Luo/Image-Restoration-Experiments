import torch
import numpy as np


def mu_tonemap(img):
    """ tonemapping HDR images using μ-law before computing loss """

    MU = 5000.0
    return torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)