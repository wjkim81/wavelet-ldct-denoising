
"""
Enhanced Deep Residual Networks for Single Image Super-Resolution
https://github.com/thstkdgus35/EDSR-PyTorch
"""
import torch
import torch.nn as nn

import os
import datetime

from models.convs import common
from models.convs import attention
from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss

from .base_model import BaseModel

class EDSR(BaseModel):
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
        parser.add_argument('--in_skip', dest='in_skip', action='store_true',
            help='use in_skip')
        parser.add_argument('--no_in_skip', dest='in_skip', action='store_false',
            help='not use in_skip')
        parser.set_defaults(in_skip=True)
        parser.add_argument('--global_skip', action='store_true',
            help='use global_skip')
        
        parser.add_argument('--act', type=str, default='relu',
            help='activation function')
        parser.add_argument('--pre_train', type=str, default='',
            help='pre-trained model directory')
        parser.add_argument('--n_resblocks', type=int, default=32,
            help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=96,
            help='number of feature maps')
        parser.add_argument('--kernel_size', type=int, default=3,
            help='kernel size of convolution layer')
        parser.add_argument('--stride', type=int, default=1,
            help='stride of convolution and deconvolution layers')
        parser.add_argument('--bn', action='store_true',
            help='do batch normalization')
        parser.add_argument('--res_scale', type=float, default=1,
            help='residual scaling')

        # Attention map
        parser.add_argument('--attn', default=False, action='store_true',
            help='use channel attention')
        parser.add_argument('--spatial_attn', default=False, action='store_true',
            help='use spatial gate in attention')
        parser.add_argument('--reduction', type=int, default=16,
            help='reduction ratio in attention')
        parser.add_argument('--pool_type', nargs='+', default=['avg'], choices=['avg', 'max'],
            help='avg, max', )

        parser.add_argument('--each_attn', type=int, default=4,
            help='add channel attention for each number of resblocks')
        # parser.add_argument('--n_attnblocks', type=int, default=8,
        #     help='Number of attention blocks')

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')
        if is_train:
            # Loss
            parser = parse_perceptual_loss(parser)
            
        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.model + '-patch' + str(opt.patch_size)
        model_opt = model_opt + "-n_resblocks" + str(opt.n_resblocks) + "-n_feats" + str(opt.n_feats) + "-bn" +str(opt.bn)
        model_opt = model_opt + "-in_skip" + str(opt.in_skip) + "-global_skip" + str(opt.global_skip)
        model_opt = model_opt + "-cl_" + opt.content_loss

        if opt.perceptual_loss is not None:
            model_opt = model_opt + '-perceptual_loss' + '-' + opt.perceptual_loss + "-" + opt.style_loss

        if opt.attn:
            model_opt = model_opt + '-attn' + '-spatial' + str(opt.spatial_attn) + '-reduction' + str(opt.reduction)
            if 'avg' in opt.pool_type:
                model_opt = model_opt + 'avg'
            if 'max' in opt.pool_type:
                model_opt = model_opt + 'max'
        
        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir

    def __init__(self, opt):
        # super(WaveletDL, self).__init__()
        BaseModel.__init__(self, opt)

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        if self.perceptual_loss:
            self.loss_name = [
                'content_loss', 'style_loss', 'total_loss'
            ]
        else:
            self.loss_name = [
                'll_loss', 'high_loss', 'total_loss'
            ]

        self.model_names = ['net']
        self.var_name = ['x', 'out', 'target' ]

        # Create model
        self.net = create_model(opt).to(self.device)

        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()
                
            if self.perceptual_loss:
                self.perceptual_loss_criterion = PerceptualLoss(opt)

            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

        self.nc = opt.n_channels
        self.mse_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out = self.net(self.x)

    def backward(self):
        if self.perceptual_loss:
            self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.target, self.out)
            self.loss = self.content_loss + self.style_loss
        else:
            self.loss = self.content_loss_criterion(self.target, self.out)

        self.loss.backward()

        mse_loss = self.mse_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        if self.perceptual_loss:
            print("Content Loss: {:.8f}, Style Loss: {:.8f}".format(
                self.content_loss, self.style_loss)
            )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.8f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )



def create_model(opt):
    return EDSRModel(opt)

class EDSRModel(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(EDSRModel, self).__init__()

        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = 3 
        n_channels = opt.n_channels
        bn = opt.bn
        bias = not bn
        act = nn.ReLU(True)
        self.in_skip = opt.in_skip
        self.global_skip = opt.global_skip

        pix_range = 1.0
        self.sub_mean = common.MeanShift(pix_range, n_channels=n_channels)
        self.add_mean = common.MeanShift(pix_range, n_channels=n_channels, sign=1)

        if opt.attn: CBAM = attention.CBAM

        # define head module
        m_head = [conv(n_channels, n_feats, kernel_size)]

        # define body module
        # m_body = [
        #     common.ResBlock(
        #         conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
        #     ) for _ in range(n_resblocks)
        # ]
        m_body = []
        for i in range(n_resblocks):
            m_body.append(
                 common.ResBlock(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale)
            )
            if (i + 1) % opt.each_attn == 0 and opt.attn:
                m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
                m_body.append(
                    CBAM(n_feats, reduction_ratio=opt.reduction, pool_types=opt.pool_type, spatial_attn=opt.spatial_attn)
                )
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
                conv(n_feats, n_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        global_res = x
        
        x = self.head(x)
        
        res = x
        x = self.body(x)
        if self.in_skip: x += res

        x = self.tail(x)
        if self.global_skip: x += global_res

        out = self.add_mean(x)
        return out

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
