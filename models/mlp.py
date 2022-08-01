"""
Image denoising: Can plain neural networks compete with BM3D?
f(x) =b3 + W3 * tanh(b2 + W2 * tanh(b1 + W1 * x)).
The architectureof  an  MLP  is  definedby the number of hidden layers and by the layer sizes.
Forinstance, a (256, 2000, 1000, 10)-MLP has two hidden layers.
"""

import os
import datetime

import torch
import torch.nn as nn

from models.convs import common
from .base_model import BaseModel

class MLP(BaseModel):
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
            self.loss_criterion = nn.MSELoss()
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
        self.loss.backward()
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

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
    return MLPModel(opt)

class MLPModel(nn.Module):
    def __init__(self, opt):
        super(MLPModel, self).__init__()
        n_channels = opt.n_channels
        patch_size = opt.patch_size
        n_dim = n_channels * patch_size * patch_size

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)
        self.mlp = nn.Sequential(*[
            nn.Linear(n_dim, 2000),
            nn.Tanh(),
            nn.Linear(2000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 10),
            nn.Tanh(),
            nn.Linear(10, n_dim)
            ]
        )
    
    def forward(self, x):
        _, c, h, w = x.shape
        # x = self.sub_mean(x)
        x = x.view(-1, c * h * w)
        x = self.mlp(x)
        out = x.view(-1, c, h, w)
        # out = self.add_mean(out)

        return out