"""
Residual Dense Network for Image Super-Resolution
https://github.com/yulunzhang/RDN
"""
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from .base_model import BaseModel

url = {
    'data-mayof32g32b8l16': 'https://www.dropbox.com/s/q5bjbvm0tzdy4vk/epoch_best_n0124_loss0.00014028_psnr39.5489.pth?dl=1'
}

class RDN(BaseModel):
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
        parser.add_argument('--n_feats', type=int, default=32,
            help='number of features')
        parser.add_argument('--growth_rate', type=int, default=32,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_blocks', type=int, default=16,
            help='number of dense blocks')
        parser.add_argument('--n_layers', type=int, default=8,
            help='number of layers of dense blocks')
        if is_train:
            parser.add_argument('--loss', type=str, choices=['l1', 'l2'], default='l1',
                help='loss function (l1, l2)')

        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.model

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_model(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.L1Loss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

        if opt.url:
            datasetname = ''
            for d in opt.datasets:
                datasetname = datasetname + d
            url_name = 'data-{}f{}g{}b{}l{}'.format(datasetname, opt.n_feats, opt.growth_rate, opt.n_layers, opt.n_blocks)
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
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.5f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )





def create_model(opt):
    return RDNModel(opt)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        # print('x.shape:', x.shape)
        # print('self.layers(x).shape:', self.layers(x).shape)
        # print('self.lff(self.layers(x)).shape', self.lff(self.layers(x)).shape)
        return x + self.lff(self.layers(x))  # local residual learning


class RDNModel(nn.Module):
    # def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
    def __init__(self, opt):
        super(RDNModel, self).__init__()

        num_channels = opt.n_channels
        num_features = opt.n_feats
        growth_rate = opt.growth_rate
        num_blocks = opt.n_blocks
        num_layers = opt.n_layers

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        pix_range = 1.0
        self.sub_mean = common.MeanShift(pix_range, n_channels=num_channels)
        self.add_mean = common.MeanShift(pix_range, n_channels=num_channels, sign=1)

        # up-sampling
        # assert 2 <= scale_factor <= 4
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
        #                              nn.PixelShuffle(2)])
        #     self.upscale = nn.Sequential(*self.upscale)
        # else:
        #     self.upscale = nn.Sequential(
        #         nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
        #         nn.PixelShuffle(scale_factor)
        #     )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        x = self.sub_mean(x)
        global_res = x
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x) + global_res

        x = self.add_mean(x)

        return x