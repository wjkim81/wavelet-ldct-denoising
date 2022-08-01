
"""
Enhanced Deep Residual Networks for Single Image Super-Resolution
https://github.com/thstkdgus35/EDSR-PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime

from models.convs import common
from models.common import networks
from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss

from .base_model import BaseModel

class GANP(BaseModel):
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
        parser.add_argument('--bn', action='store_true',
            help='do batch normalization')
        parser.add_argument('--res_scale', type=float, default=1,
            help='residual scaling')

        # U-Net
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--perceptual_loss', type=str, default='perceptual_loss',
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss', 'edsrganp'],
            help='specity loss_type')

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
        parser.add_argument('--backbone', type=str, default='edsr',
            choices=['edsr', 'unet'],
            help='model to be trained.'
        )

        if is_train:
            parser = parse_perceptual_loss(parser)
            parser.add_argument('--gan_w', type=float, default=0.001,
                help='weight of image loss')
            parser.add_argument('--use_pretrain', action='store_true',
                help='use pre-trained model')
                
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)
            
        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.backbone + opt.model
        model_opt = model_opt + '-p_loss' + '-' + opt.perceptual_loss + "-gan_w" + str(opt.gan_w)
        
        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir

    def __init__(self, opt):
        # super(WaveletDL, self).__init__()
        BaseModel.__init__(self, opt)

        self.var_name = ['x', 'out', 'target' ]

        # Create model
        self.netG = create_model(opt)

        if opt.backbone == 'edsr':
            pretrain_pth = '../../data/denoising/checkpoints/mayo-20200713-2004-edsr-patch80-n_resblocks32-n_feats96-bnFalse-in_skipTrue-global_skipTrue-cl_l1/pth/epoch_best_n0189_loss0.00014403_psnr39.4527.pth'
        elif opt.backbone == 'unet':
            pretrain_pth = '../../data/denoising/checkpoints/mayo-20200909-1146-unet/pth/epoch_best_n0133_loss0.00015243_psnr39.2240.pth'
        
        if self.is_train and opt.use_pretrain:
            print('Use pretrained network\n{}'.format(pretrain_pth))
            checkpoint = torch.load(pretrain_pth)
            state_dict_net = checkpoint['net']
            self.netG.load_state_dict(state_dict_net)

        self.netG = self.netG.to(self.device)

        self.nc = opt.n_channels
        self.mse_criterion = nn.MSELoss()

        # Define losses and optimizers
        if self.is_train:
            self.model_names = ['netG', 'netD']

            self.netD = networks.define_D(self.nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = self.netD.to(self.device)

            if opt.content_loss == 'l1':
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()
                
            self.perceptual_loss_criterion = PerceptualLoss(opt)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.mse_loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.gan_w = opt.gan_w
        else:
            self.model_names = ['netG']

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out = self.netG(self.x)

    def backward_D(self):
        fake = self.out
        pred_fake = self.netD(fake.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.target)
        loss_d_real = self.criterionGAN(pred_real, True)
    
        self.loss_d = (loss_d_real + loss_d_fake) * 0.5
        self.loss_d.backward()

    def backward_G(self):
        fake = self.out
        pred_fake = self.netD(fake)
        self.loss_g = self.criterionGAN(pred_fake, True)

        self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.target, self.out)
        self.loss_c = self.content_loss + self.style_loss

        self.loss = self.loss_c + self.gan_w * self.loss_g 
        self.loss.backward()

        mse_loss = self.mse_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("Content Loss: {:.5f}, Style Loss: {:.5f},".format(
            self.content_loss, self.style_loss)
        )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.5f}, loss_D: {:5f}, loss_G: {:5f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.loss_d.item(), self.loss_g.item(), self.psnr.item())
        )


def create_model(opt):
    if opt.backbone == 'edsr':
        model = EDSRModel(opt)
    elif opt.backbone == 'unet':
        model = UNetModel(opt)

    return model

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

        # define head module
        m_head = [conv(n_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
            ) for _ in range(n_resblocks)
        ]
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
        n_channels = opt.n_channels
        bilinear = opt.bilinear

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)

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

        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
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
        out = self.add_mean(out)
        return out
