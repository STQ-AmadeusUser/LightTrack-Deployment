import torch
import torch.nn as nn
from .ds import DepthwiseSeparableConv
from .ir import InvertedResidual
from .modules import *


class ChildNet_FCN(nn.Module):

    def __init__(self):
        super(ChildNet_FCN, self).__init__()

        # Stem
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = Swish()

        # Middle stages (IR/ER/DS Blocks)
        blocks = []
        # stage 0
        blocks.append(nn.Sequential(
            DepthwiseSeparableConv(in_chs=16, dw_chs=16, dw_k=3, dw_s=1, dw_p=1, se_chs=8, out_chs=16)
        ))
        # stage 1
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=16, dw_chs=64, dw_k=7, dw_s=2, dw_p=3, se_chs=16, out_chs=24),
            InvertedResidual(in_chs=24, dw_chs=144, dw_k=7, dw_s=1, dw_p=3, se_chs=40, out_chs=24),
        ))
        # stage 2
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=24, dw_chs=96, dw_k=3, dw_s=2, dw_p=1, se_chs=24, out_chs=40),
            InvertedResidual(in_chs=40, dw_chs=160, dw_k=5, dw_s=1, dw_p=2, se_chs=40, out_chs=40),
            InvertedResidual(in_chs=40, dw_chs=240, dw_k=7, dw_s=1, dw_p=3, se_chs=64, out_chs=40),
            InvertedResidual(in_chs=40, dw_chs=240, dw_k=3, dw_s=1, dw_p=1, se_chs=64, out_chs=40),
        ))
        # stage 3
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=40, dw_chs=160, dw_k=7, dw_s=2, dw_p=3, se_chs=40, out_chs=80),
            InvertedResidual(in_chs=80, dw_chs=320, dw_k=3, dw_s=1, dw_p=1, se_chs=80, out_chs=80),
            InvertedResidual(in_chs=80, dw_chs=320, dw_k=7, dw_s=1, dw_p=3, se_chs=80, out_chs=80),
            InvertedResidual(in_chs=80, dw_chs=320, dw_k=7, dw_s=1, dw_p=3, se_chs=80, out_chs=80),
        ))
        # stage 4
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=80, dw_chs=480, dw_k=7, dw_s=1, dw_p=3, se_chs=120, out_chs=96),
            InvertedResidual(in_chs=96, dw_chs=384, dw_k=5, dw_s=1, dw_p=2, se_chs=96, out_chs=96),
            InvertedResidual(in_chs=96, dw_chs=576, dw_k=3, dw_s=1, dw_p=1, se_chs=144, out_chs=96),
        ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # architecture = [[0], [], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x
