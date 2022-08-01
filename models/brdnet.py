"""
Image denoising using deep CNN with batch renormalization（BRDNet）
BRDNET(keras) : https://github.com/hellloxiaotian/BRDNet/blob/master/colorimage/batch_renorm.py
"""
# import os
# import datetime

import torch
import torch.nn as nn

from models.convs import common

from models.convs.batchrenorm import BatchRenorm2d
from .base_model import BaseModel

class BRDNet(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--kernel_size', type=int, default=3,
            help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=64,
            help='number of feature maps')
 
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
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_model(opt).to(self.device)
        
        if opt.is_train:
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
        self.loss = self.mse_loss_criterion(self.out, self.target)
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
    return BRDNetModel(opt)

class BRDNetModel(nn.Module):
    def __init__(self, opt):
        super(BRDNetModel, self).__init__()

        kernel_size = opt.kernel_size

        #default 64
        n_feats = opt.n_feats

        pix_range = 1.0
        self.sub_mean = common.MeanShift(pix_range, n_channels=opt.n_channels)
        self.add_mean = common.MeanShift(pix_range, n_channels=opt.n_channels, sign=1)
        in_channels = opt.n_channels

        def _create_bn(opt, n_feats):

            BRN = BatchRenorm2d(n_feats)
            return BRN
        act = nn.ReLU(True)
        conv = common.default_conv
        dlconv = common.dilated_conv

        layers1 = []
        layers2 = []
        layers3 = []

        #conv layer
        layers1.append(conv(in_channels, n_feats, kernel_size))
        layers1.append(_create_bn(opt, n_feats))
        layers1.append(act)

        for _ in range(15):
            layers1.append(conv(n_feats, n_feats, kernel_size))
            layers1.append(_create_bn(opt, n_feats))
            layers1.append(act)

        layers1.append(conv(n_feats, in_channels, kernel_size))

        #dilated conv layer
        layers2.append(conv(in_channels, n_feats, kernel_size))
        layers2.append(_create_bn(opt, n_feats))
        layers2.append(act)

        for _ in range(7):
            layers2.append(dlconv(n_feats, n_feats, kernel_size))
            layers2.append(act)

        layers2.append(conv(n_feats, n_feats, kernel_size))
        layers2.append(_create_bn(opt, n_feats))
        layers2.append(act)

        for _ in range(6):
            layers2.append(dlconv(n_feats, n_feats, kernel_size))
            layers2.append(act)

        layers2.append(conv(n_feats, n_feats, kernel_size))
        layers2.append(_create_bn(opt, n_feats))
        layers2.append(act)
        layers2.append(conv(n_feats, in_channels, kernel_size))

        #concat layer
        layers3.append(conv(in_channels*2, in_channels, kernel_size))

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)


    def forward(self, inp):
        inp = self.sub_mean(inp)

        x = self.layers1(inp)
        x = inp - x

        y = self.layers2(inp)
        y = inp - y

        z = torch.cat((x, y), 1)
        z = self.layers3(z)
        z = inp - z

        z = self.add_mean(z)

        return z
