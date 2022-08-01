"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
U-Net architecture is slightly different depending on each implementation.
I implemented U-Net based of the following paper:
'A performance comparison of convolutional neural network-based imagedenoising methods: The effect of loss functions on low-dose CT images'
Byeongjoon Kim, Minah Han, Hyunjung Shima), and Jongduk Baek
"""
# import os
# import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from models.common.unet import create_unet

from .base_model import BaseModel

class UNet(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Network parameters
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')
        if is_train:
            parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l2',
                help='loss function (l1, l2)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_unet(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out = self.net(self.x)

    def backward(self):
        self.loss = self.loss_criterion(self.out, self.target)
        mse_loss = self.mse_loss_criterion(self.out, self.target)
        self.psnr = 10 * torch.log10(1 / mse_loss)
        
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.8f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )



# def create_model(opt):
#     return UNetModel(opt)

# class single_conv(nn.Module):
#     def __init__(self, in_ch, out_ch, bn=False):
#         super(single_conv, self).__init__()
#         m_body = []
#         m_body.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
#         if bn: m_body.append(nn.BatchNorm2d(out_ch))
#         m_body.append(nn.ReLU(inplace=True))

#         self.conv = nn.Sequential(*m_body)

#         # self.conv = nn.Sequential(
#         #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
#         #     nn.ReLU(inplace=True)
#         # )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
        
#         self.double_conv = nn.Sequential(
#             single_conv(in_channels, out_channels, bn=bn),
#             single_conv(out_channels, out_channels, bn=bn)
#         )
#         # self.double_conv = nn.Sequential(
#         #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(mid_channels),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.ReLU(inplace=True)
#         # )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class UNetModel(nn.Module):
#     def __init__(self, opt):
#         super(UNetModel, self).__init__()
#         n_channels = opt.n_channels
#         bilinear = opt.bilinear

#         self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         # self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         # self.down4 = Down(512, 1024 // factor)
#         # self.up1 = Up(1024, 512 // factor, bilinear)
#         # self.up2 = Up(512, 256 // factor, bilinear)

#         self.convs = nn.Sequential(
#             single_conv(256, 256),
#             single_conv(256, 256),
#             single_conv(256, 256),
#             single_conv(256, 256),
#             single_conv(256, 256),
#             single_conv(256, 128),
#         )

#         self.up1 = Up(256, 128 // factor, bilinear)
#         self.up2 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_channels)

#         self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

#     def forward(self, x):
#         x = self.sub_mean(x)
#         res = x
#         x1 = self.inc(x)
#         # print('x1.shape:', x1.shape)
#         x2 = self.down1(x1)
#         # print('x2.shape:', x2.shape)
#         x3 = self.down2(x2)
#         # print('x3.shape:', x3.shape)
#         # x4 = self.down3(x3)
#         # x5 = self.down4(x4)
#         x = self.convs(x3)
#         # print('x.shape:', x.shape)

#         x = self.up1(x, x2)
#         # print('up1 x.shape:', x.shape)
#         x = self.up2(x, x1)
#         # print('up2 x.shape:', x.shape)
#         # x = self.up3(x, x2)
#         # x = self.up4(x, x1)
#         x = self.outc(x)
#         # print('outc x.shape:', x.shape)
#         out = x + res
#         out = self.add_mean(out)
#         return out
