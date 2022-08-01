"""
This code implements wavelet domain convolutional neural network based on edsr codes
by Wonjin Kim
"""
import os
import datetime

import torch
import torch.nn as nn

from models.convs import common
from models.convs import attention
from models.loss.perceptual_loss import parse_wavelet_perceptual_loss, WaveletPerceptualLoss

from models.convs.wavelet import serialize_swt, unserialize_swt
from models.convs.wavelet import SWTForward, SWTInverse
from models.convs.norm import Norm2d, Norm2d_
from utils.utils import print_subband_loss

from .base_model import BaseModel

class WaveletDL(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Wavelet deep learning model specification
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--swt_lv', type=int, default=2,
            help='Level of stationary wavelet transform')

        # Network parameters
        parser.add_argument('--in_skip', dest='in_skip', action='store_true',
            help='use in_skip')
        parser.add_argument('--no_in_skip', dest='in_skip', action='store_false',
            help='not use in_skip')
        parser.set_defaults(in_skip=True)
        parser.add_argument('--global_skip', dest='global_skip', action='store_true',
            help='use global_skip')
        parser.add_argument('--no_global_skip', dest='global_skip', action='store_false',
            help='not use global_skip')
        parser.set_defaults(global_skip=True)
        
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
        parser.add_argument('--bn', dest='bn', action='store_true',
            help='do batch normalization')
        parser.add_argument('--no_bn', dest='bn', action='store_false',
            help='no batch normalization')
        parser.set_defaults(bn=False)
        parser.add_argument('--res_scale', type=float, default=1,
            help='residual scaling')

        # Attention map
        parser.add_argument('--attn', default=False, action='store_true',
            help='use channel attention')
        parser.add_argument('--spatial_attn', default=False, action='store_true',
            help='use spatial gate in attention')
        parser.add_argument('--reduction', type=int, default=16,
            help='reduction ratio in attention')
        parser.add_argument('--pool_types', nargs='+', default=['avg'], choices=['avg', 'max'],
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
            parser.add_argument('--ll_weight', type=float, default=0.2,
                help='weight of LL loss to high loss')
            parser = parse_wavelet_perceptual_loss(parser)
            parser.add_argument('--img_loss', default=False, action='store_true',
                help='include img loss')
            
        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date
        model_opt = model_opt + "-" + opt.model  + "-patch" + str(opt.patch_size)
        model_opt = model_opt + "-n_resblocks" + str(opt.n_resblocks)
        model_opt = model_opt + "-n_feats" + str(opt.n_feats) + "-bn" +str(opt.bn)
        model_opt = model_opt + "-swt_" + opt.wavelet_func + "_lv" + str(opt.swt_lv)
        model_opt = model_opt + "-cl_" + opt.content_loss + '-img_loss' + str(opt.img_loss)

        if opt.perceptual_loss is not None:
            model_opt = model_opt + '-perceptual_loss' + '-' + opt.perceptual_loss  + "-" + opt.style_loss

        if opt.attn:
            model_opt = model_opt + '-attn' + '-spatial' + str(opt.spatial_attn) + '-reduction' + str(opt.reduction)
            if 'avg' in opt.pool_types:
                model_opt = model_opt + 'avg'
            if 'max' in opt.pool_types:
                model_opt = model_opt + 'max'
        
        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        if self.perceptual_loss and self.is_train:
            self.loss_name = [
                'll_content_loss', 'll_style_loss', 'll_loss', 
                'high_content_loss', 'high_style_loss', 'high_loss',
                'content_loss', 'style_loss', 'total_loss'
            ]
        else:
            self.loss_name = [
                'll_loss', 'high_loss', 'total_loss'
            ]

        self.model_names = ['net', 'norm']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_model(opt).to(self.device)

        # Define SWTForward and SWTInverse
        self.swt_lv = opt.swt_lv
        self.swt = SWTForward(J=opt.swt_lv, wave=opt.wavelet_func).to(self.device)
        self.iswt = SWTInverse(wave=opt.wavelet_func).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()
                
            if self.perceptual_loss:
                self.wavelet_perceptual_loss = WaveletPerceptualLoss(opt)

            self.ll_weight = opt.ll_weight
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

            self.save_spectrum = opt.save_spectrum
            self.add_img_loss = opt.img_loss

        self.nc = opt.n_channels
        swt_num_channels = (3 * opt.swt_lv + 1) * opt.n_channels
        self.norm = Norm2d_(swt_num_channels).to(self.device)
        self.mse_criterion = nn.MSELoss()
        self.training = True

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)
        # print("x.shape:", self.x.shape)

    def forward(self):
        # self.norm.train()
        x = self.swt(self.x)       # swt
        x = serialize_swt(x)  # Serialize
        std_x = self.norm(x, update_stat=self.training) # Normalize
        self.swt_std_out = self.net(std_x)

        # self.norm.eval()
        out = self.norm(self.swt_std_out, inverse=True)
        out = unserialize_swt(out, J=self.swt_lv, C=self.nc) # out = (ll, swt_coeffs)
        self.out = self.iswt(out)

    def test(self):
        with torch.no_grad():
            self.training = False
            self.forward()
            self.training = True

    def backward(self):
        # self.norm.eval()
        swt_target = self.swt(self.target)
        swt_target = serialize_swt(swt_target)

        # Get noramlized (standarized) wavelet 
        std_target = self.norm(swt_target)
        std_out = self.swt_std_out

        if self.perceptual_loss:
            self.ll_content_loss, self.ll_style_loss, self.high_content_loss, self.high_style_loss = \
                self.wavelet_perceptual_loss(std_target, std_out)
            self.ll_loss = self.ll_content_loss + self.ll_style_loss
            self.high_loss = self.high_content_loss + self.high_style_loss
        else:
            ll_target = std_target[:, :self.nc]
            ll_out  = std_out[:, :self.nc]
            high_target = std_target[:, self.nc:]
            high_out = std_out[:, self.nc:]

            self.ll_loss = self.content_loss_criterion(ll_target, ll_out)
            self.high_loss = self.content_loss_criterion(high_target, high_out)

        self.wavelet_loss =  self.ll_weight * self.ll_loss + (1 - self.ll_weight) * self.high_loss
        
        if self.add_img_loss:
            self.img_loss = self.content_loss_criterion(self.target, self.out)
            self.loss = self.wavelet_loss + self.img_loss
        else:
            self.loss = self.wavelet_loss
        self.loss.backward()

        # mse_loss = self.mse_criterion(self.out, self.target)
        # Calculate for mse loss and PSNR
        mse_loss = self.mse_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()

        # save_spectrum is recoreded only during training
        # optimize_parameters is not used during validation or test
        if self.save_spectrum:
            print_subband_loss(self.opt, 'spectrum.txt', self.target.detach(), self.out.detach(), self.swt)

        self.backward()
        self.optimizer.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        if self.perceptual_loss:
            print("High Content Loss: {:.5f}, High Style Loss: {:.5f}, LL Content Loss: {:.5f}, LL Style Loss:{:.5f}".format(
                self.high_content_loss, self.high_style_loss, self.ll_content_loss, self.ll_style_loss)
            )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Total Loss: {:.5f} High Loss: {:.5f} LL Loss: {:.5f} PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.high_loss.item(), self.ll_loss.item(), self.psnr.item())
        )

"""
    Define the network here
"""
def create_model(opt):
    return WaveletDL2Model(opt)

class WaveletDL2Model(nn.Module):
    def __init__(self, opt):
        super(WaveletDL2Model, self).__init__()

        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = opt.kernel_size
        self.in_skip = opt.in_skip
        self.global_skip = opt.global_skip

        swt_num_channels = (3 * opt.swt_lv + 1) * opt.n_channels
        in_channels = swt_num_channels
        out_channels = swt_num_channels
        bn = opt.bn
        bias = not bn
        
        act = nn.ReLU(True)
        conv = common.default_conv

        ResBlock = common.ResBlock
        if opt.attn: CBAM = attention.CBAM

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]

        if opt.attn:
            m_body = []
            for i in range(n_resblocks):
                m_body.append(
                    ResBlock(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale)
                )

                if (i + 1)% opt.each_attn == 0:
                    m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
                    m_body.append(
                        CBAM(n_feats, reduction_ratio=opt.reduction, pool_types=opt.pool_types, spatial_attn=opt.spatial_attn)
                    )
        else : 
            m_body = [
                ResBlock(
                    conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
                ) for _ in range(n_resblocks)
            ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
                conv(n_feats, out_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        global_res = x

        x = self.head(x)
        res = self.body(x)
        if self.in_skip: res += x

        x = self.tail(res)
        self.global_skip: x = x + global_res

        return x 