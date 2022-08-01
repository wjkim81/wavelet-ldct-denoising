"""
Low-Dose CT via Transfer Learning from a 3d Trained Network
https://github.com/hmshan/CPCE-3D
"""
# import os
# import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.convs import common

from .base_model import BaseModel
from .cpce2d import create_cpce2d
from .wganvgg import create_D
from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss

class CPCE3D(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(depth=9)
        parser.add_argument('--n_d_train', type=int, default=1,
            help='number of discriminator training')
        parser.add_argument('--perceptual', dest='perceptual', action='store_true',
            help='use perceptual loss')
        parser.add_argument('--mse', dest='perceptual', action='store_false',
            help='use MSE loss')
        parser.add_argument('--perceptual_loss', type=str, default='cpce3',
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')

        if is_train:
            parser = parse_perceptual_loss(parser)

            parser.add_argument('--transfer_learning', default=False, action='store_true',
                help='transfer learning from cpce2d')
            # parser.add_argument('--cpce2d_path', type=str,
            #     default='../../data/denoising/checkpoints/mayo-20201015-0715-cpce2d/pth/epoch_best_n0074_loss0.00016992_psnr38.6772.pth',
            #     help='path to cpce2d')
            parser.set_defaults(perceptual=True)
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net_G']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net_G = create_G(opt)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            if opt.transfer_learning:
                if opt.datasets[0] == 'mayo3d':
                    cpce2d_path = '../../data/denoising/checkpoints/mayo-20201015-0715-cpce2d/pth/epoch_best_n0074_loss0.00016992_psnr38.6772.pth'
                elif opt.datasets[0] == 'phantom3d':
                    cpce2d_path = '../../data/denoising/checkpoints/mayo-20201015-0715-cpce2d/pth/epoch_best_n0074_loss0.00016992_psnr38.6772.pth'
                elif opt.dataset[0] == 'dahim3d':
                    cpce2d_path = '../../data/denoising/checkpoints/dahim-20210325-1912-cpce2d/pth/epoch_best_n0067_loss0.00153140_psnr28.5372.pth'

                cpce2d = create_cpce2d(opt)
                # cpce2d_pth = opt.cpce2d_path

                checkpoint = torch.load(cpce2d_path)
                state_dict_ae = checkpoint['net']
                cpce2d.load_state_dict(state_dict_ae)
                self.init_cpce3d_weights(cpce2d)

            self.net_G.to(self.device)
            self.net_D = create_D(opt).to(self.device)

            self.model_names = ['net_G', 'net_D']

            self.perceptual_loss_criterion = PerceptualLoss(opt)
            self.mse_loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.n_d_train = opt.n_d_train
            self.gp = True
            self.perceptual = opt.perceptual
        else:
            self.net_G.to(self.device)

    def init_cpce3d_weights(self, cpce2d):
        params_cpce3d = self.net_G.named_parameters()
        params_cpce2d = cpce2d.named_parameters()

        # print(self.net_G)
        # print(self.cpce2d)

        dict_params_cpce3d = dict(params_cpce3d)

        for name, param in params_cpce2d:
            # print('name:', name)
            if name in dict_params_cpce3d:
                # print('param.data.shape', param.data.shape)
                # print('dict_params_cpce3d[name].data.shape:', dict_params_cpce3d[name].data.shape)
                if len(param.data.shape) < len(dict_params_cpce3d[name].data.shape):
                    dict_params_cpce3d[name].data.fill_(0)
                    dict_params_cpce3d[name].data[:, :, 1].copy_(param.data)
                else:
                    dict_params_cpce3d[name].data.copy_(param.data)
                # print(dict_params_cpce3d[name].data[:, :, 0])
                # print(dict_params_cpce3d[name].data[:, :, 1])
                # print(dict_params_cpce3d[name].data[:, :, 2])

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)
            self.target = self.target[:, :, self.target.size(2) // 2]

    def forward(self):
        self.out = self.net_G(self.x)

    def gp_loss(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1))).to(self.device)
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.net_D(interp)
        fake_ = torch.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def p_loss(self, x, y):
        """
        percetual loss
        """
        # n, c, d, h, w = x.shape
        c_loss, p_loss = self.perceptual_loss_criterion(x, y)
        loss = c_loss + p_loss
        return loss

    def backward_D(self):
        fake = self.out
        d_real = self.net_D(self.target)
        d_fake = self.net_D(fake)
        loss = -torch.mean(d_real) + torch.mean(d_fake)
        loss_gp = self.gp_loss(self.target, fake) if self.gp else 0

        self.loss_d = loss + loss_gp
        self.loss_d.backward()

    def backward_G(self):
        fake = self.out
        d_fake = self.net_D(fake)
        loss = -torch.mean(d_fake)
        # loss = 0
        loss_p = self.p_loss(self.out, self.target) if self.perceptual else self.mse_loss_criterion(self.out, self.target)

        self.loss_g = loss + loss_p
        self.loss_g.backward()

        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        self.loss = mse_loss

    def optimize_parameters(self):
        self.set_requires_grad([self.net_D], True)
        for _ in range(self.n_d_train):
            self.optimizer_D.zero_grad()
            self.forward()
            self.backward_D()
            self.optimizer_D.step()

        self.set_requires_grad([self.net_D], False)
        self.optimizer_G.zero_grad()
        self.forward()
        self.backward_G()
        self.optimizer_G.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss_G: {:.8f}, Loss_D: {:.8f}, MSE_loss: {:.8f}, PSNR: {:.8f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss_g.item(), self.loss_d.item(), self.loss.item(), self.psnr.item())
        )


def create_G(opt):
    return CPCE3DModel(opt)

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(DownConv, self).__init__()
        m_body = []
        m_body.append(nn.Conv3d(in_ch, out_ch, 3, padding=0))
        if bn: m_body.append(nn.BatchNorm3d(out_ch))
        m_body.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(UpConv, self).__init__()
        self.convT = nn.ConvTranspose3d(in_ch, in_ch, 3, padding=0)
        self.one = nn.Conv3d(2*in_ch, out_ch, 1)
        self.leru = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.convT(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.leru(x)
        
        x = self.one(x)
        x = self.leru(x)

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, padding=0)
        self.leru = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.leru(self.conv(x))

class CPCE3DModel(nn.Module):
    def __init__(self, opt):
        super(CPCE3DModel, self).__init__()
        n_channels = opt.n_channels
        n_depth = opt.depth

        assert n_depth in [3, 5, 7, 9], "only depth with 3, 5, 7, and 9 is trainable now"

        self.conv1 = nn.Conv3d(n_channels, 32, 3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        
        if n_depth >= 3:
            self.conv2 = nn.Conv3d(32, 32, 3, bias=False)
        else:
            self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        
        if n_depth >= 5:
            self.conv3 = nn.Conv3d(32, 32, 3, bias=False)
        else:
            self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        

        if n_depth >= 9:
            self.conv4 = nn.Conv3d(32, 32, 3, bias=False)
        else:
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
        if x.dim() == 5: x = x.squeeze(2)
        x = self.conv2(x1)
        x2 = self.relu2(x)
        if x.dim() == 5: x = x.squeeze(2)
        x = self.conv3(x2)
        x3 = self.relu2(x)
        if x.dim() == 5: x = x.squeeze(2)

        x = self.conv4(x3)
        x = self.relu4(x)
        if x.dim() == 5: x = x.squeeze(2)
        
        x = self.deconv1(x)
        x = torch.cat([x, x3[:, :, x3.size(2)//2]], dim=1)
        x = self.drelu1(x)
        x = self.conv1x1_1(x)
        x = self.relu1x1_1(x)

        x = self.deconv2(x)
        x = torch.cat([x, x2[:, :, x2.size(2)//2]], dim=1)
        x = self.drelu2(x)
        x = self.conv1x1_2(x)
        x = self.relu1x1_2(x)

        x = self.deconv3(x)
        x = torch.cat([x, x1[:, :, x1.size(2)//2]], dim=1)
        x = self.drelu3(x)
        x = self.conv1x1_3(x)
        x = self.relu1x1_3(x)
        
        x = self.deconv4(x)
        out = self.drelu4(x)
        
        return out
