"""
Toward convolutional blind denoising of real pho-tographs
https://github.com/IDKiro/CBDNet-pytorch
train_real.py
"""
# import os
# import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from .base_model import BaseModel

class CBDNet(BaseModel):
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
        return parser

    # @staticmethod
    # def set_savedir(opt):
    #     # # Customize opt parameters
    #     # dt = datetime.datetime.now()
    #     # date = dt.strftime("%Y%m%d-%H%M")
    #     # dataset_name = ''
    #     # for d in opt.datasets:
    #     #     dataset_name = dataset_name + d

    #     # model_opt = dataset_name  + "-" + date + "-" + opt.model

    #     # if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
    #     # if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
    #     # save_dir = os.path.join(opt.checkpoints_dir, model_opt)
    #     # return save_dir
    #     super.set_save(opt)

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target', 'noise_level_est']

        # Create model
        self.net = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            self.mse_loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out, self.noise_level_est = self.net(self.x)

    def backward(self):
        f_loss = fixed_loss()
        self.loss = f_loss(self.out, self.target, self.noise_level_est, 0, 0)
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

def create_model(opt):
    return CBDNetModel(opt)

class CBDNetModel(nn.Module):
    def __init__(self, opt):
        super(CBDNetModel, self).__init__()
        n_channels = opt.n_channels
        self.fcn = FCN(opt)
        self.unet = UNet(opt)
        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    
    def forward(self, x):
        x = self.sub_mean(x)
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        out = self.add_mean(out)
        return noise_level, out


class FCN(nn.Module):
    def __init__(self, opt):
        super(FCN, self).__init__()
        c = opt.n_channels
        self.inc = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Sequential(
            nn.Conv2d(32, c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        conv1 = self.inc(x)
        conv2 = self.conv(conv1)
        conv3 = self.conv(conv2)
        conv4 = self.conv(conv3)
        conv5 = self.outc(conv4)
        return conv5


class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()
        c = opt.n_channels
        
        self.inc = nn.Sequential(
            single_conv(c*2, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, c)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


#안넣음
def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = torch.mean(torch.pow((out_image - gt_image), 2)) + \
                if_asym * 0.5 * torch.mean(torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))) + \
                0.05 * tvloss
        return loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]