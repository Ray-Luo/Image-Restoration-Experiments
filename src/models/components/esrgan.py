import random

import numpy as np
import torch
from basicsr.archs.discriminator_arch import UNetDiscriminatorSN

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from torch import nn
from torch.nn import functional as F


class RealESRGANNet(nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_output_channels,
        degredation_params,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super(RealESRGANNet, self).__init__()
        self.net_g = RRDBNet(
            num_in_ch=num_input_channels,
            num_out_ch=num_output_channels,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )
        self.net_d = UNetDiscriminatorSN(
            num_in_ch=num_input_channels, num_feat=64, skip_connection=True
        )

        self.degredation_params = degredation_params
        self.scale = scale

        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def forward(self, batch, degrade=False):
        if type(batch) == dict:
            out = self.net_g(batch["lq"])
        # in case the model is used for simple inference without dataloader
        else:
            out = self.net_g(batch)
        # out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        return out
