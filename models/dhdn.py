"""
Densely Connected Hierarchical Network for Image Denoising
https://github.com/BumjunPark/DHDN
"""
import torch
import torch.nn as nn

from models.convs import common
from .base_model import BaseModel

class DHDN(BaseModel):
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
        parser.add_argument('--pad', type=str, default='reflect',
            help='pad type of networks')
        parser.add_argument('--norm', type=str, default='none',
            help='normalization type of networks')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']

        # Create model
        self.net = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.L1Loss()
            self.mse_loss_criterion = nn.MSELoss()
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

def create_model(opt):
    return DHCNModel(opt)

class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_1(x))

        conc = torch.cat([x, out], 1)

        out = self.relu2(self.conv_2(conc))

        conc = torch.cat([conc, out], 1)

        out = self.relu3(self.conv_3(conc))

        out = torch.add(out, residual)

        return out


class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out


class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))

        out = self.subpixel(out)

        return out

class DHCNModel(nn.Module):
    def __init__(self, opt):
        super(DHCNModel, self).__init__()
        n_channels = opt.n_channels

        self.conv_i = nn.Conv2d(in_channels=n_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, 128)
        self.DCR_block12 = self.make_layer(_DCR_block, 128)
        self.down1 = self.make_layer(_down, 128)
        self.DCR_block21 = self.make_layer(_DCR_block, 256)
        self.DCR_block22 = self.make_layer(_DCR_block, 256)
        self.down2 = self.make_layer(_down, 256)
        self.DCR_block31 = self.make_layer(_DCR_block, 512)
        self.DCR_block32 = self.make_layer(_DCR_block, 512)
        self.down3 = self.make_layer(_down, 512)
        self.DCR_block41 = self.make_layer(_DCR_block, 1024)
        self.DCR_block42 = self.make_layer(_DCR_block, 1024)
        self.up3 = self.make_layer(_up, 2048)
        self.DCR_block33 = self.make_layer(_DCR_block, 1024)
        self.DCR_block34 = self.make_layer(_DCR_block, 1024)
        self.up2 = self.make_layer(_up, 1024)
        self.DCR_block23 = self.make_layer(_DCR_block, 512)
        self.DCR_block24 = self.make_layer(_DCR_block, 512)
        self.up1 = self.make_layer(_up, 512)
        self.DCR_block13 = self.make_layer(_DCR_block, 256)
        self.DCR_block14 = self.make_layer(_DCR_block, 256)
        self.conv_f = nn.Conv2d(in_channels=256, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sub_mean(x)
        residual = x

        out = self.relu1(self.conv_i(x))
        out = self.DCR_block11(out)
        conc1 = self.DCR_block12(out)
        out = self.down1(conc1)
        out = self.DCR_block21(out)
        conc2 = self.DCR_block22(out)
        out = self.down2(conc2)
        out = self.DCR_block31(out)
        conc3 = self.DCR_block32(out)
        conc4 = self.down3(conc3)
        out = self.DCR_block41(conc4)
        out = self.DCR_block42(out)
        out = torch.cat([conc4, out], 1)
        out = self.up3(out)
        out = torch.cat([conc3, out], 1)
        out = self.DCR_block33(out)
        out = self.DCR_block34(out)
        out = self.up2(out)
        out = torch.cat([conc2, out], 1)
        out = self.DCR_block23(out)
        out = self.DCR_block24(out)
        out = self.up1(out)
        out = torch.cat([conc1, out], 1)
        out = self.DCR_block13(out)
        out = self.DCR_block14(out)
        out = self.relu2(self.conv_f(out))
        out = torch.add(residual, out)

        out = self.add_mean(out)
        return out