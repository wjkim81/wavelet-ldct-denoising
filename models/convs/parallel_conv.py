import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs.common import ResBlock
from models.convs.attention import CBAM as CBAM

class ParallelBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, groups, bias=True):
        
        super(ParallelBlock, self).__init__()

        m = [
            nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups,
            padding=(kernel_size//2), bias=bias)
        ]
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)

class ParallelResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, groups,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,
        attn=False, reduction_ratio=16, pool_types=['avg', 'max'], spatial_attn=False):

        super(ParallelResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                padding=(kernel_size//2), groups=groups, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(out_channels))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.attn = attn

        if self.attn:
            self.cbam = CBAM(out_channels, reduction_ratio=reduction_ratio, pool_types=pool_types, spatial_attn=spatial_attn)

    def forward(self, x):
        res = x
        x = self.body(x).mul(self.res_scale)
        if self.attn:
            x = self.cbam(x)

        # print("x.size():", x.size())
        # print("res.size():", res.size())
        out = res + x
        return out