import torch
import torch.nn as nn
import torch.nn.functional as F


class head_subnet(nn.Module):
    def __init__(self):
        super(head_subnet, self).__init__()
        cls_tower = []
        reg_tower = []

        cls_tower.append(SeparableConv2d_BNReLU(128, 256, 5, 1, 2))
        cls_tower.append(SeparableConv2d_BNReLU(256, 256, 5, 1, 2))
        cls_tower.append(SeparableConv2d_BNReLU(256, 256, 3, 1, 1))
        cls_tower.append(SeparableConv2d_BNReLU(256, 256, 3, 1, 1))
        cls_tower.append(SeparableConv2d_BNReLU(256, 256, 3, 1, 1))
        cls_tower.append(SeparableConv2d_BNReLU(256, 256, 3, 1, 1))

        reg_tower.append(SeparableConv2d_BNReLU(128, 192, 3, 1, 1))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 3, 1, 1))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 3, 1, 1))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 3, 1, 1))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 3, 1, 1))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 5, 1, 2))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 5, 1, 2))
        reg_tower.append(SeparableConv2d_BNReLU(192, 192, 5, 1, 2))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)
        self.cls_perd = cls_pred_head(inchannels=256)
        self.reg_pred = reg_pred_head(inchannels=192)

    def forward(self, inp):
        oup = {}
        # cls
        cls_feat = self.cls_tower(inp['cls'])
        oup['cls'] = self.cls_perd(cls_feat)
        # reg
        reg_feat = self.reg_tower(inp['reg'])
        oup['reg'] = self.reg_pred(reg_feat)
        return oup


class SeparableConv2d_BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_BNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.ReLU(self.BN(x))
        return x


class cls_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(cls_pred_head, self).__init__()
        self.cls_pred = nn.Conv2d(inchannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """mode should be in ['all', 'cls', 'reg']"""
        x = 0.1 * self.cls_pred(x)
        return x


class reg_pred_head(nn.Module):
    def __init__(self, inchannels=256, stride=16):
        super(reg_pred_head, self).__init__()
        self.stride = stride
        # reg head
        self.bbox_pred = nn.Conv2d(inchannels, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.bbox_pred(x)) * self.stride
        return x
