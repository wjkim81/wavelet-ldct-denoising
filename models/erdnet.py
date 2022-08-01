"""
This code is implementing combination of shuffnet and edsr for the efficient network for denoising
ERDNet: Efficieint residual deep networks
by Wonjin Kim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from models.convs import shuffle_conv
from models.convs.shuffle_conv import ShuffleBlock as ShuffleBlock
from models.convs.shuffle_conv import channel_shuffle as channel_shuffle

def create_model(opt):
    return ERDNet(opt)

class ERDNet(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(ERDNet, self).__init__()

        n_shuffle_blocks = opt.n_shuffle_blocks
        n_feats = opt.n_feats
        n_shuffle_feats = n_feats // 2
        kernel_size = 3 
        self.n_channels = opt.n_channels
        self.bias = opt.bias
        self.bn = opt.batch_norm

        act = nn.ReLU(True)
        # self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]

        self.shift_mean = opt.shift_mean

        pix_range = 1.0
        self.sub_mean = common.MeanShift(pix_range, n_channels=self.n_channels)
        self.add_mean = common.MeanShift(pix_range, n_channels=self.n_channels, sign=1)

        # define head module
        m_head = [conv(self.n_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ShuffleBlock(
                n_shuffle_feats, n_shuffle_feats, bias=self.bias, bn=self.bn, act=act
            ) for _ in range(n_shuffle_blocks)
        ]
        m_body.append(conv(n_shuffle_feats, n_shuffle_feats, kernel_size))

        # define tail module
        m_tail = [
                conv(n_feats, self.n_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # if self.shift_mean and self.n_channels == 3:
        x = self.sub_mean(x)
        x = self.head(x)

        x1 = x[:, :(x.shape[1]//2), :, :]
        x2 = x[:, (x.shape[1]//2):, :, :]

        res = self.body(x2)
        # res += x
        x = torch.cat((x1, res), 1)
        x = channel_shuffle(x, 2)

        x = self.tail(x)
        # if self.shift_mean and self.out_channels == 3:
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
