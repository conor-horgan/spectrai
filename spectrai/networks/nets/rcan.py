import math
import torch
from torch import nn
from spectrai.networks.layers import Upsampler

# ------------------------------------------------------------
#                    Hyperspectral RCAN
# ------------------------------------------------------------
# This code is adapted from: https://github.com/yulunzhang/RCAN
# The corresponding paper is:
# Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu, 
# Image Super-Resolution Using Very Deep Residual Channel Attention Networks
# ECCV, 2018
# Available from: https://arxiv.org/abs/1807.02758

class ChannelAttentionBlock(nn.Module):
    """Channel attention block for hyperspectral RCAN."""
    def __init__(self, channels, activation, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.chan_attn = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
                activation,
                nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.chan_attn(y)
        return x * y

class ResidualChannelAttentionBlock(nn.Module):
    """Residual channel attention block for hyperspectral RCAN."""
    def __init__(self, channels=500, kernel_size=3, reduction=16, bias=True, activation=nn.ReLU(True)):
        super(ResidualChannelAttentionBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size//2), bias=bias))
            if i == 0: modules_body.append(activation)
        modules_body.append(ChannelAttentionBlock(channels, activation, reduction))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    """Residual group for hyperspectral RCAN."""
    def __init__(self, channels=500, kernel_size=3, reduction=16, bias=True, activation=nn.ReLU(True), n_resblocks=6):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [ResidualChannelAttentionBlock(channels, kernel_size, reduction, bias=bias, activation=nn.ReLU(True)) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size//2), bias=bias))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Hyperspectral_RCAN(nn.Module):
    """Hyperspectral RCAN class."""
    def __init__(self, spectrum_length, scale, activation, kernel_size=3, reduction=16, bias=True, n_resblocks=16, n_resgroups=18):
        super(Hyperspectral_RCAN, self).__init__()       
        modules_head1 = [Upsampler(scale, spectrum_length, kernel_size, activation=False), nn.Conv2d(spectrum_length, spectrum_length, kernel_size, padding=(kernel_size//2), bias=bias)]
        modules_head2 = [nn.Conv2d(spectrum_length, int(spectrum_length/2), kernel_size, padding=(kernel_size//2), bias=bias)]

        modules_body = [ResidualGroup(int(spectrum_length/2), kernel_size, reduction, activation, n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(int(spectrum_length/2), int(spectrum_length/2), kernel_size, padding=(kernel_size//2), bias=bias))

        modules_tail = [nn.Conv2d(int(spectrum_length/2), int(spectrum_length/2), kernel_size, padding=(kernel_size//2), bias=bias)]
        modules_tail.append(nn.Conv2d(int(spectrum_length/2), spectrum_length, kernel_size, padding=(kernel_size//2), bias=bias))

        self.head1 = nn.Sequential(*modules_head1)
        self.head2 = nn.Sequential(*modules_head2)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head1(x)
        x1 = self.head2(x)

        res1 = self.body(x1)
        res1 += x1

        res2 = self.tail(res1)
        res2 += x

        return res2