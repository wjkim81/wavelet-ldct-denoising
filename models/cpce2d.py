"""
Low-Dose CT via Transfer Learning from a 2D Trained Network
https://github.com/hmshan/CPCE-3D
"""
# import os
# import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common

from .base_model import BaseModel

class CPCE2D(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_cpce2d(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
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
        self.psnr = 10 * torch.log10(1 / self.loss)
        
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



def create_cpce2d(opt):
    return CPCE2DModel(opt)

class CPCE2DModel(nn.Module):
    def __init__(self, opt):
        super(CPCE2DModel, self).__init__()
        n_channels = opt.n_channels

        self.conv1 = nn.Conv2d(n_channels, 32, 3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, bias=False)
        self.drelu1 = nn.ReLU(inplace=True)
        self.conv1x1_1 = nn.Conv2d(32 * 2, 32, 1, bias=False)
        self.relu1x1_1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(32, 32, 3, bias=False)
        self.drelu2 = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(32 * 2, 32, 1, bias=False)
        self.relu1x1_2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(32, 32, 3, bias=False)
        self.drelu3 = nn.ReLU(inplace=True)
        self.conv1x1_3 = nn.Conv2d(32 * 2, 32, 1, bias=False)
        self.relu1x1_3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(32, n_channels, 3, bias=False)
        self.drelu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.relu1(x)
        x = self.conv2(x1)
        x2 = self.relu2(x)
        x = self.conv3(x2)
        x3 = self.relu2(x)
        x = self.conv4(x3)
        x = self.relu4(x)
        
        x = self.deconv1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.drelu1(x)
        x = self.conv1x1_1(x)
        x = self.relu1x1_1(x)

        x = self.deconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.drelu2(x)
        x = self.conv1x1_2(x)
        x = self.relu1x1_2(x)

        x = self.deconv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.drelu3(x)
        x = self.conv1x1_3(x)
        x = self.relu1x1_3(x)
        
        x = self.deconv4(x)
        out = self.drelu4(x)

        return out
