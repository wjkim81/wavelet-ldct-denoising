import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def dilated_conv(in_channels, out_channels, kernel_size, bias=True, dilation=2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size+(dilation-1)*2)//2,
        dilation = dilation,
        bias = True
    )

class MeanShift(nn.Module):
    def __init__(self, pixel_range, n_channels, rgb_mean=None, rgb_std=None, sign=-1):
        super(MeanShift, self).__init__()

        if rgb_mean is None and rgb_std is None:
            if n_channels == 1:
                rgb_mean = [0.5]
                rgb_std =[1.0]
            elif n_channels == 3:
                rgb_mean = (0.4488, 0.4371, 0.4040)
                rgb_std = (1.0, 1.0, 1.0)

        self.shifter = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
        std = torch.Tensor(rgb_std)
        self.shifter.weight.data = torch.eye(n_channels).view(n_channels, n_channels, 1, 1) / std.view(n_channels, 1, 1, 1)
        self.shifter.bias.data = sign * pixel_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        super(BasicBlock, self).__init__()

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

    def forward(self, x):
        return self.body(x).mul(self.res_scale)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res