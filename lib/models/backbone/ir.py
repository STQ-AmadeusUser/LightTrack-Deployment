import torch
import numpy as np
import torch.nn as nn
from .modules import *


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs=16, dw_chs=64, dw_k=7, dw_s=2, dw_p=3, se_chs=16, out_chs=24):
        super(InvertedResidual, self).__init__()
        self.has_residual = (in_chs == out_chs and dw_s == 1)

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_chs, dw_chs, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dw_chs)
        self.act1 = Swish()

        # Depth-wise convolution
        self.conv_dw = nn.Conv2d(dw_chs, dw_chs, dw_k, dw_s, dw_p,
                                 groups=dw_chs, bias=False)
        self.bn2 = nn.BatchNorm2d(dw_chs)
        self.act2 = Swish()

        # Squeeze-and-excitation
        self.se = SqueezeExcite(dw_chs, se_chs)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv2d(dw_chs, out_chs, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            x += residual

        return x