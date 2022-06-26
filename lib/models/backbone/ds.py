import torch
import numpy as np
import torch.nn as nn
from .modules import *


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs=16, dw_chs=16, dw_k=3, dw_s=1, dw_p=1, se_chs=8, out_chs=16):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = (dw_s == 1 and in_chs == out_chs)

        self.conv_dw = nn.Conv2d(in_chs, dw_chs, dw_k, dw_s, dw_p,
                                 groups=dw_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(dw_chs)
        self.act1 = Swish()

        # Squeeze-and-excitation
        self.se = SqueezeExcite(dw_chs, se_chs)

        self.conv_pw = nn.Conv2d(dw_chs, out_chs, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)
        self.act2 = nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.se is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            x += residual
        return x
