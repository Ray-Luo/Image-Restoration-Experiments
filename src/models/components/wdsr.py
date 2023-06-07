import math

import torch
import torch.nn as nn
import torch.nn.init as init

# Wide Activation for Efficient and Accurate Image Super-Resolution (https://arxiv.org/abs/1808.08718)


class Block(nn.Module):
    def __init__(
        self, num_residual_units, kernel_size, width_multiplier, res_scale: float
    ) -> None:
        super(Block, self).__init__()
        weight_norm = torch.nn.utils.weight_norm
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)

        body = []
        body.append(conv)
        body.append(nn.ReLU(True))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x


class WDSR(nn.Module):
    def __init__(
        self,
        scale,
        image_mean,
        num_input_channels,
        num_residual_units,
        num_blocks,
        width_multiplier,
        **kwargs,
    ) -> None:
        super(WDSR, self).__init__()
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = num_input_channels
        num_outputs = scale * scale * num_input_channels

        self.image_mean = image_mean

        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        for _ in range(num_blocks):
            body.append(
                Block(
                    num_residual_units,
                    kernel_size,
                    width_multiplier,
                    res_scale=1 / math.sqrt(num_blocks),
                )
            )
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.body = nn.Sequential(*body)

        skip = []
        if num_inputs != num_outputs:
            conv = weight_norm(
                nn.Conv2d(
                    num_inputs,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2,
                )
            )
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        x = x - self.image_mean
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        x = x + self.image_mean
        return x
