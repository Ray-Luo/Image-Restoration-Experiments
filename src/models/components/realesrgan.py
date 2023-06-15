from basicsr.archs.discriminator_arch import UNetDiscriminatorSN

from basicsr.archs.rrdbnet_arch import RRDBNet
from torch import nn


class RealESRGANNet(nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_output_channels,
        loss,
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
        self.loss = loss

    def forward(self, x, degrade=False):
        out = self.net_g(x)
        return out
