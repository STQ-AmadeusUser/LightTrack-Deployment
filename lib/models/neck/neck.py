import torch
import torch.nn as nn
import torch.nn.functional as F


class BN_adj(nn.Module):
    '''
    BN adjust layer before Correlation
    '''
    def __init__(self, num_channel):
        super(BN_adj, self).__init__()
        self.BN_z = nn.BatchNorm2d(num_channel)
        self.BN_x = nn.BatchNorm2d(num_channel)

    def forward(self, zf, xf):
        return self.BN_z(zf), self.BN_x(xf)


class PW_Corr_adj(nn.Module):
    def __init__(self, num_kernel=64, adj_channel=128):
        super(PW_Corr_adj, self).__init__()
        self.pw_corr = PWCA(num_kernel)
        self.adj_layer = nn.Conv2d(num_kernel, adj_channel, 1)

    def forward(self, kernel, search):
        '''
        stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16
        '''
        oup = {}
        corr_feat = self.pw_corr([kernel], [search])
        corr_feat = self.adj_layer(corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


class PWCA(nn.Module):
    """
    Pointwise Correlation & Channel Attention
    """

    def __init__(self, num_channel):
        super(PWCA, self).__init__()
        self.CA_layer = CAModule(channels=num_channel)

    def forward(self, z, x):
        z11 = z[0]
        x11 = x[0]
        # pixel-wise correlation
        corr = pixel_corr_mat(z11, x11)
        # channel attention
        opt = self.CA_layer(corr)

        return opt


class CAModule(nn.Module):
    """
    Channel attention module
    """

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


def pixel_corr_mat(z, x):
    """
    Pixel-wise correlation (implementation by matrix multiplication)
    The speed is faster because the computation is vectorized
    """
    b, c, h, w = x.size()
    z_mat = z.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)





