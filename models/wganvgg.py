"""
WGAN-VGG: Low-dose CT image denoising using a generative adversarial network with Wasserstein distance and perceptual loss

https://github.com/SSinyu/WGAN_VGG

"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F

from models.convs import common
from .base_model import BaseModel

# url = {
#     'data-mayof32g32b8l16': 'https://www.dropbox.com/s/q5bjbvm0tzdy4vk/epoch_best_n0124_loss0.00014028_psnr39.5489.pth?dl=1'
# }

class WGANVGG(BaseModel):
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
        parser.add_argument('--n_d_train', type=int, default=4,
            help='number of discriminator training')
        parser.add_argument('--generator', type=str, default='unet',
            help='generator model [unet | original]')
        parser.add_argument('--perceptual', dest='perceptual', action='store_true',
            help='use perceptual loss')
        parser.add_argument('--mse', dest='perceptual', action='store_false',
            help='use MSE loss')

        if is_train:
            parser.set_defaults(perceptual=True)
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)


        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_name = ['loss_d', 'loss_g', 'loss_p', 'loss']
        self.var_name = ['x', 'out', 'target']

        # Create model
        # self. = create_model(opt).to(self.device)
        self.net_G = create_G(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            self.net_D = create_D(opt).to(self.device)
            self.feature_extractor = create_vgg().to(self.device)
            self.model_names = ['net_G', 'net_D', 'feature_extractor']
            
            self.perceptual_criterion = nn.MSELoss()
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
            self.model_names = ['net_G']
            

        if opt.url:
            datasetname = ''
            for d in opt.datasets:
                datasetname = datasetname + d
            url_name = 'data-{}f{}'.format(datasetname, opt.n_feats)
            if url_name in url:
                self.url = url[url_name]
            else:
                print('url_name:', url_name)
                raise('Set model configurations correctly to load model from url')

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

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
        fake = x.repeat(1, 3, 1, 1)
        real = y.repeat(1, 3, 1, 1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.perceptual_criterion(fake_feature, real_feature)
        return loss

    def backward_D(self):
        fake = self.out
        d_real = self.net_D(self.target)
        d_fake = self.net_D(fake)
        loss = -torch.mean(d_real) + torch.mean(d_fake)
        loss_gp = self.gp_loss(self.target, fake) if self.gp else 0

        self.loss_d = loss + loss_gp
        self.loss_d.backward()
        # return (loss, gp_loss) if return_gp else loss
        # d_loss, gp_loss = self.WGANVGG.d_loss(x, y, gp=True, return_gp=True)

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
            # self.net_D.zero_grad()
            self.forward()
            self.backward_D()
            self.optimizer_D.step()

        # self.loss_d = torch.zeros([1])

        self.set_requires_grad([self.net_D], False)
        self.optimizer_G.zero_grad()
        # self.net_G.zero_grad()
        self.forward()
        self.backward_G()
        self.optimizer_G.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss_G: {:.8f}, Loss_D: {:.8f}, MSE_loss: {:.8f}, PSNR: {:.8f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss_g.item(), self.loss_d.item(), self.loss.item(), self.psnr.item())
        )

# Create mwcnn model

def create_G(opt):
    if opt.generator == 'unet':
        generator = UNetModel(opt)
    elif opt.generator == 'original':
        generator = WGAN_VGG_generator(opt)
    return generator

def create_D(opt):
    return WGAN_VGG_discriminator(opt)

def create_vgg():
    return WGAN_VGG_FeatureExtractor()

"""
Define WGAN-VGG model
"""
class WGAN_VGG_generator(nn.Module):
    def __init__(self, opt):
        super(WGAN_VGG_generator, self).__init__()
        n_channels = opt.n_channels
        layers = [nn.Conv2d(n_channels,32,3,1,1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32,32,3,1,1))
            layers.append(nn.ReLU())
        layers.extend([nn.Conv2d(32,n_channels,3,1,1), nn.ReLU()])
        self.net = nn.Sequential(*layers)

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        out = self.net(x)
        out = self.add_mean(out)
        return out


class WGAN_VGG_discriminator(nn.Module):
    def __init__(self, opt):
        super(WGAN_VGG_discriminator, self).__init__()
        input_size = opt.patch_size
        n_channels = opt.n_channels
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        ch_stride_set = [(n_channels,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        network = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()
        for param in network.parameters():
            param.requires_grad = False

        self.feature_extractor = network

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        super(WGAN_VGG, self).__init__()
        input_size = opt.patch_size
        self.generator = WGAN_VGG_generator(opt)
        self.discriminator = WGAN_VGG_discriminator(opt)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()
        

    def d_loss(self, x, y, gp=True, return_gp=False):
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty





def create_model(opt):
    return UNetModel(opt)

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
        bilinear = True

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
