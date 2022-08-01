"""
This code implements wavelet domain convolutional neural network based on edsr codes
Add perceptual loss and GAN loss
by Wonjin Kim
"""
import os
import datetime

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from models.common import networks
from models.loss.perceptual_loss import parse_wavelet_perceptual_loss, WaveletPerceptualLoss

from models.convs.wavelet import serialize_swt, unserialize_swt
from models.convs.wavelet import SWTForward, SWTInverse
from models.convs.norm import Norm2d, Norm2d_
from utils.utils import print_subband_loss

from .base_model import BaseModel

class WaveletGANP(BaseModel):
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
        parser.add_argument('--res_scale', type=float, default=1,
            help='residual scaling')

        # U-Net
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')

        # GAN options
        parser.add_argument('--netD', type=str, default='basic',
            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--ndf', type=int, default=64,
            help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3,
            help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
            choices=['instance', 'batch', 'none'],
            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
            choices=['normal', 'xavier', 'kaming', 'orthogonal'],
            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
            help='scaling factor for normal, xavier and orthogonal.')
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--gan_mode', type=str, default='lsgan',
            choices=['vanilla', 'lsgan', 'wganp],'],
            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'
        )

        parser.add_argument('--perceptual_loss', type=str, default='perceptual_loss',
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss',
                    'ganp1', 'ganp2', 'ganp3', 'ganp4', 'ganp5', 'ganp6',
                    'ganp7', 'ganp8', 'ganp9', 'ganp10', 'ganp11', 'ganp12'],
            help='specity loss_type')
        parser.add_argument('--backbone', type=str, default='waveletdl2',
            choices=['waveletdl2', 'waveletunet'],
            help='model to be trained.'
        )

        if is_train:
            # Loss
            parser.add_argument('--ll_weight', type=float, default=0.2,
                help='weight of LL loss to high loss')
            parser = parse_wavelet_perceptual_loss(parser)
            parser.add_argument('--img_loss', default=False, action='store_true',
                help='include image loss')
            parser.add_argument('--img_w', type=float, default=0.2,
                help='weight of image loss')
            parser.add_argument('--gan_w', type=float, default=0.001,
                help='weight of image loss')
        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date
        model_opt = model_opt + "-" + opt.model  + "-backbone_" + opt.backbone
        model_opt = model_opt + "-patch" + str(opt.patch_size) + "-n_resblocks" + str(opt.n_resblocks)
        model_opt = model_opt + "-n_feats" + str(opt.n_feats) + "-bn" +str(opt.bn)
        model_opt = model_opt + "-swt_" + opt.wavelet_func + "_lv" + str(opt.swt_lv)
        model_opt = model_opt + '-img_loss' + str(opt.img_loss) + '-gan_w' + str(opt.gan_w)
        model_opt = model_opt + '-p_loss' + '-' + opt.perceptual_loss
        
        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_type = opt.perceptual_loss

        if self.is_train:
            self.model_names = ['net', 'netDL', 'netDH', 'norm']
        else:
            self.model_names = ['net', 'norm']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.nc = opt.n_channels
        swt_num_channels = (3 * opt.swt_lv + 1) * opt.n_channels
    
        self.net = create_model(opt)
        self.norm = Norm2d_(swt_num_channels)

        if opt.is_train:
            if opt.backbone == 'waveletdl2' and opt.datasets[0] == 'mayo':
                pretrain_pth = '../../data/denoising/checkpoints/mayo-best-20200720-1814-waveletdl2-patch80-n_resblocks32-n_feats96-bnFalse-swt_haar_lv2-cl_l1/pth/epoch_best_n0143_loss0.00013746_psnr39.3901.pth'
            elif opt.backbone == 'waveletdl2' and opt.datasets[0] == 'phantom':
                pretrain_pth = '../../data/denoising/checkpoints/phantom-20200919-0941-waveletdl2-patch80-n_resblocks32-n_feats96-bnFalse-swt_haar_lv2-cl_l1-img_lossFalse/pth/epoch_best_n0163_loss0.00007354_psnr47.8273.pth'
            elif opt.backbone == 'waveletunet':
                pretrain_pth = '../../data/denoising/checkpoints/mayo-20200909-0326-waveletunet/pth/epoch_best_n0188_loss0.00015318_psnr39.4208.pth'

            checkpoint = torch.load(pretrain_pth)
            state_dict_net = checkpoint['net']
            state_dict_norm = checkpoint['norm']

            self.net.load_state_dict(state_dict_net)
            self.norm.load_state_dict(state_dict_norm)

        self.net = self.net.to(self.device)
        self.norm = self.norm.to(self.device)


        # Define SWTForward and SWTInverse
        self.swt_lv = opt.swt_lv
        self.swt = SWTForward(J=opt.swt_lv, wave=opt.wavelet_func).to(self.device)
        self.iswt = SWTInverse(wave=opt.wavelet_func).to(self.device)

        self.norm = Norm2d_(swt_num_channels).to(self.device)
        self.mse_criterion = nn.MSELoss()
        self.training = True
        
        # Define losses and optimizers
        if self.is_train:
            self.netDL = networks.define_D(self.nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDH = networks.define_D(swt_num_channels - self.nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netDL = self.netDL.to(self.device)
            self.netDH = self.netDH.to(self.device)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if opt.content_loss == 'l1':
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()
                
            self.wavelet_perceptual_loss = WaveletPerceptualLoss(opt)

            self.ll_weight = opt.ll_weight
            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            self.optimizer_G = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                self.netDL.parameters(), self.netDH.parameters()),
                lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0
            )
            

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.add_img_loss = opt.img_loss
            self.img_w = opt.img_w
            self.gan_w = opt.gan_w

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)
        # print("x.shape:", self.x.shape)

    def forward(self):
        # self.norm.train()
        x = self.swt(self.x)       # swt
        x = serialize_swt(x)  # Serialize
        std_x = self.norm(x, update_stat=False) # Normalize
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

    def backward_D(self):
        swt_target = self.swt(self.target)
        swt_target = serialize_swt(swt_target)
        real_l = swt_target[:, :self.nc]
        real_h = swt_target[:, self.nc:]

        fake_l = self.swt_std_out[:, :self.nc]
        fake_h = self.swt_std_out[:, self.nc:]

        pred_fake_l = self.netDL(fake_l.detach())
        pred_fake_h = self.netDH(fake_h.detach())

        self.loss_DL_fake = self.criterionGAN(pred_fake_l, False)
        self.loss_DH_fake = self.criterionGAN(pred_fake_h, False)

        pred_real_l = self.netDL(real_l)
        pred_real_h = self.netDH(real_h)

        self.loss_DL_real = self.criterionGAN(pred_real_l, True)
        self.loss_DH_real = self.criterionGAN(pred_real_h, True)

        self.loss_DL = (self.loss_DL_fake + self.loss_DL_real) * 0.5
        self.loss_DH = (self.loss_DH_fake + self.loss_DH_real) * 0.5

        self.loss_DL.backward()
        self.loss_DH.backward() 

    def backward_G(self):
        # self.norm.eval()
        swt_target = self.swt(self.target)
        swt_target = serialize_swt(swt_target)
        std_target = self.norm(swt_target)
        std_out = self.swt_std_out

        fake_l = std_out[:, :self.nc]
        fake_h = std_out[:, self.nc:]

        pred_fake_l = self.netDL(fake_l)
        pred_fake_h = self.netDH(fake_h)

        self.loss_GL_GAN = self.criterionGAN(pred_fake_l, True)
        self.loss_GH_GAN = self.criterionGAN(pred_fake_h, True)

        self.ll_content_loss, self.ll_style_loss, self.high_content_loss, self.high_style_loss = \
            self.wavelet_perceptual_loss(std_target, std_out)

        
        self.ll_loss = self.ll_content_loss + self.ll_style_loss + self.gan_w * self.loss_GL_GAN
        self.high_loss = self.high_content_loss + self.high_style_loss + self.gan_w * self.loss_GH_GAN

        self.wavelet_loss =  self.ll_weight * self.ll_loss + (1 - self.ll_weight) * self.high_loss
        
        if self.add_img_loss:
            self.img_loss = self.content_loss_criterion(self.target, self.out)
            self.loss_G = (1 - self.img_w) * self.wavelet_loss + self.img_w * self.img_loss
        else:
            self.loss_G = self.wavelet_loss
        self.loss_G.backward()

        # mse_loss = self.mse_criterion(self.out, self.target)
        # Calculate for mse loss and PSNR
        mse_loss = self.mse_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        self.loss = mse_loss

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netDL, self.netDH], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad([self.netDL, self.netDH], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("Loss_GAN_L: {:.5f}, Loss_GAN_H: {:.5f}, Loss_DL: {:.5f}, Loss_DH:{:.5f}".format(
            self.loss_GL_GAN, self.loss_GH_GAN, self.loss_DL, self.loss_DH)
        )
        print("High Content Loss: {:.5f}, High Style Loss: {:.5f}, LL Content Loss: {:.5f}, LL Style Loss:{:.5f}".format(
            self.high_content_loss, self.high_style_loss, self.ll_content_loss, self.ll_style_loss)
        )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): G Loss: {:.5f} High Loss: {:.5f} LL Loss: {:.5f} PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss_G.item(), self.high_loss.item(), self.ll_loss.item(), self.psnr.item())
        )

"""
    Define the network here
"""
def create_model(opt):
    if opt.backbone == 'waveletdl2':
        model = WaveletDL2Model(opt)
    elif opt.backbone == 'waveletunet':
        model = UNetModel(opt)

    return model

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

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]
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


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(single_conv, self).__init__()
        m_body = []
        m_body.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if bn: m_body.append(nn.BatchNorm2d(out_ch))
        m_body.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*m_body)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            single_conv(in_channels, out_channels, bn=bn),
            single_conv(out_channels, out_channels, bn=bn)
        )
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(self, opt):
        super(UNetModel, self).__init__()
        swt_num_channels = (3 * opt.swt_lv + 1) * opt.n_channels
        n_channels = swt_num_channels

        bilinear = opt.bilinear

        # self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)

        self.convs = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 128),
        )

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

        # self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def forward(self, x):
        # x = self.sub_mean(x)
        res = x
        x1 = self.inc(x)
        # print('x1.shape:', x1.shape)
        x2 = self.down1(x1)
        # print('x2.shape:', x2.shape)
        x3 = self.down2(x2)
        # print('x3.shape:', x3.shape)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.convs(x3)
        # print('x.shape:', x.shape)

        x = self.up1(x, x2)
        # print('up1 x.shape:', x.shape)
        x = self.up2(x, x1)
        # print('up2 x.shape:', x.shape)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x = self.outc(x)
        # print('outc x.shape:', x.shape)
        out = x + res
        # out = self.add_mean(out)
        return out