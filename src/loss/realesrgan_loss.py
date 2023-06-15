# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from basicsr.losses.basic_loss import L1Loss, PerceptualLoss
from basicsr.losses.gan_loss import GANLoss
from torch import nn


class REALESRGANLosses(nn.Module):
    def __init__(self, pixel_opt, perceptual_opt, gan_opt):
        super(REALESRGANLosses, self).__init__()
        self.pixel_loss = L1Loss(**pixel_opt)
        self.perceptual_loss = PerceptualLoss(**perceptual_opt)
        self.gan_loss = GANLoss(**gan_opt)
