"""
Learning Deep CNN Denoiser Prior for Image Restoration
https://arxiv.org/abs/1704.03264
"""

import os
import datetime
import torch
import torch.nn as nn

from .base_model import BaseModel


class IRCNN(BaseModel):
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
        parser.add_argument('--n_feats', type=int, default=64,
            help='number of feature maps')
        return parser

    @staticmethod
    def set_savedir(opt):
        # Customize opt parameters
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
        self.out = self.net(self.x)

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
    return IRCNNModel(opt)


# Define IRCNN
class IRCNNModel(nn.Module):
    def __init__(self, opt):
        super(IRCNNModel,self).__init__()
        kernel_size = 3
        features = opt.n_feats
        dilate = [1, 2, 3, 4, 3, 2, 1]

        n_channels = opt.n_channels

        m_head = [
            nn.Conv2d(n_channels, features, kernel_size, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True)
        ]

        m_body = []
        for i in range(1, len(dilate) - 1):
            padding = (kernel_size  + 2 * (dilate[i] - 1)) // 2
            m_body.append(nn.Conv2d(features, features, kernel_size, padding=padding, dilation=dilate[i], bias=False))
            m_body.append(nn.BatchNorm2d(features))
            m_body.append(nn.ReLU(inplace=True))

        m_tail = [
            nn.Conv2d(features, n_channels, kernel_size, padding=kernel_size//2, bias=False),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        res = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x - res
