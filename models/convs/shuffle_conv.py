"""
https://github.com/ericsun99/Shufflenet-v2-Pytorch/blob/master/ShuffleNetV2.py
"""
import torch
import torch.nn as nn


"""
Pointwise convolution
"""
def conv_1x1(in_channels, out_channels, groups=1, bn=False, act=nn.ReLU(inplace=True), bias=False):
    m = []
    m.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
    if bn:
        m.append(nn.BatchNorm2d(out_channels))
    if act is not None:
        m.append(nn.ReLU(inplace=True))

    return nn.Sequential(*m)

"""
Depthwise convolution
"""
def dwConv(in_channels, out_channels, kernel_size=3, padding=1, groups=None, bn=False, act=nn.ReLU(inplace=True), bias=False):
    if groups is None:
        groups = in_channels 

    m = []
    m.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
    if bn:
        m.append(nn.BatchNorm2d(out_channels))
    if act is not None:
        m.append(act)

    return nn.Sequential(*m)

def transposeDwConv(in_channels, out_channels, kernel_size=3, padding=1, groups=None, bn=False, act=nn.ReLU(inplace=True), bias=False):
    if groups is None:
        groups = in_channels
    m = []
    m.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
    if bn:
        m.append(nn.BatchNorm2d(out_channels))
    if act is not None:
        m.append(act)

    return nn.Sequential(*m)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, bn=False, act=nn.ReLU(True), mode=None, add='cat'):
        super(ShuffleBlock, self).__init__()
        self.mode = mode
        if self.mode == 'down':
            self.branch1 = nn.Sequential(
                dwConv(in_channels, in_channels, bn=bn, act=act, bias=bias),
                conv_1x1(in_channels, out_channels, bn=bn, act=act, bias=bias)
            )
            self.branch2 = nn.Sequential(
                conv_1x1(in_channels, out_channels, bn=bn, act=act, bias=bias),
                dwConv(out_channels, out_channels, bn=bn, act=act, bias=bias),
                conv_1x1(out_channels, out_channels, bn=bn, act=act, bias=bias)
            )
        elif self.mode == 'up':
            self.branch1 = nn.Sequential(
                transposeDwConv(in_channels, in_channels, bn=bn, act=act, bias=bias),
                conv_1x1(in_channels, out_channels, bn=bn, act=act, bias=bias)
            )
            self.branch2 = nn.Sequential(
                conv_1x1(in_channels, out_channels, bn=bn, act=act, bias=bias),
                transposeDwConv(out_channels, out_channels, bn=bn, act=act, bias=bias),
                conv_1x1(out_channels, out_channels, bn=bn, act=act, bias=bias)
            )
        else:
            assert in_channels == out_channels
            half_feats = in_channels // 2
            self.branch2 = nn.Sequential(
                conv_1x1(half_feats, half_feats, bn=bn, act=act, bias=bias),
                dwConv(half_feats, half_feats, bn=bn, act=act, bias=bias),
                conv_1x1(half_feats, half_feats, bn=bn, act=act, bias=bias),
            )

        # self.cat = True if add == 'cat' else False


    def forward(self, x):
        if self.mode == 'down':
            out = torch.cat(self.branch1(x), self.branch2(x), 1)
        elif self.mode == 'up':
            out = torch.cat(self.branch1(x), self.branch2(x), 1)
        else:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)

        out = channel_shuffle(x, 2)

        return out

    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)