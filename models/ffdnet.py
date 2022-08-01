"""
FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising
gitgub : https://github.com/cszn/KAIR
This model requires noise level in the input and mostly assumes that additive white gaussian noise (AWGN).
Thus, this model does not fit to LDCT, which shows totally different noise from AWGN.
Of course, real noise does not look like AWGN as well.
To use this model for real noise, we need to estimate or measure noise level.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from .base_model import BaseModel

from collections import OrderedDict

class FFDNet(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--sigma',type=int, default=[0,75],
                            help='[min_sigma, max_sigma], default [0,75]')
        parser.add_argument('--sigma_test', type=int, default=25,
                            help='15,25,50 for test')

        parser.set_defaults(noise=50)
        return parser
    
    # @staticmethod
    # def set_savedir(opt):
    #     dt = datetime.datetime.now()
    #     date = dt.strftime("%Y%m%d-%H%M")
    #     dataset_name = ''
    #     for d in opt.datasets:
    #         dataset_name = dataset_name + d

    #     model_opt = dataset_name  + "-" + date + "-" + opt.model

    #     if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
    #     if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
    #     save_dir = os.path.join(opt.checkpoints_dir, model_opt)
    #     return save_dir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target', 'sigma']
        self.sigma_min = opt.sigma[0]
        self.sigma_max = opt.sigma[1]
        self.sigma_test = opt.sigma_test
        self.add_noise = opt.add_noise
        self.noise = opt.noise

        #create model
        self.net = create_model(opt).to(self.device)

        #Define losses and optimizers
        if self.is_train:
            self.mse_loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas = (opt.b1, opt.b2), eps=1e-8,weight_decay=0)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        if self.add_noise:
            x_origin = input['target'].to(self.device)
        else:
            x_origin = input['x'].to(self.device)
        
        self.x = x_origin.clone().to(self.device)
        self.sigma = []
        
        # if self.is_train:
        for i in range(len(self.x)):
        #     noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0
        #     noise = torch.randn(self.x[i].size()).mul_(noise_level).float().to(self.device)
        #     self.x[i].add_(noise)
        #     self.sigma.append(noise_level.unsqueeze(1).unsqueeze(1).unsqueeze(1))
            if self.noise == 0: # random noise
                noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0
            else:
                assert self.noise > 10
                noise_level = torch.FloatTensor([np.random.uniform(self.noise - 10, self.noise + 10)])/255.0
            noise = torch.randn(self.x[i].size()).mul_(noise_level).float().to(self.device)
            self.sigma.append(noise_level.unsqueeze(1).unsqueeze(1).unsqueeze(1))
            if self.add_noise:
                self.x[i].add_(noise)
        # else:
        #     for i in range(len(self.x)):
        #         np.random.seed(seed=0)
        #         self.x[i] += np.random.normal(0, self.sigma_test/255.0, self.x.shape).to(self.device)
        #         noise_level = torch.FloatTensor([self.sigma_test/255.0])
        #         self.sigma.append(noise_level.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        
        self.sigma = torch.cat(self.sigma, dim=0).to(self.device)
        # print('set_input) self.sigma.shape: ', self.sigma.shape)
            
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out = self.net(self.x, self.sigma)

    
    def backward(self):
        self.loss = self.mse_loss_criterion(self.out, self.target)
        self.psnr = 10 * torch.log10(1 / self.loss)
        
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.5f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )


def create_model(opt):
    return FFDNetModel(opt, in_nc=opt.n_channels, out_nc=opt.n_channels, nc=64, nb=15, act_mode='R')


# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)



class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)



def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)



class FFDNetModel(nn.Module):
    def __init__(self, opt, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNetModel, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = PixelUnShuffle(upscale_factor=sf)

        m_head = conv(in_nc*sf*sf+1, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc*sf*sf, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

        n_channels = opt.n_channels

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)


    def forward(self, x, sigma):
        x = self.sub_mean(x)

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        # print('sigma.shape: ',sigma.shape)
        # print('m.shape: ',m.shape)
        # print('x.shape: ',x.shape)
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]

        x = self.add_mean(x)
        return x


# if __name__ == '__main__':
#     from utils import utils_model
#     model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
#     print(utils_model.describe_model(model))

#     x = torch.randn((2,1,240,240))
#     sigma = torch.randn(2,1,1,1)
#     x = model(x, sigma)
#     print(x.shape)

    #  run models/network_ffdnet.py